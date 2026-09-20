#!/bin/bash
# it-65: условный монитор точки принятия решения v2 (заменяет it65_chain.sh-ожидание цели).
# v2 (2026-09-20, после ревью надёжности): готовность = НЕ «6 файлов на диске», а
# «все 7 шаговых строк OK в логах обеих цепочек» И финальные строки цепочек. Причина:
# mmaud_sahi_full.py и session_vis_probe пишут CSV инкрементально — при FAIL шага на диске
# остаётся ЧАСТИЧНЫЙ файл, и прежний критерий «файл существует» дал бы ВЕРДИКТ-ГОТОВ с
# неполным E5/E4. События: ВЕРДИКТ-ГОТОВ / ШАГ-FAIL / ШАГ-OK(каждый раз) / ВНИМАНИЕ-ЧАСТИЧНЫЙ /
# ТРЕНИРОВКА-СТОП / прогресс-эпоха.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
RUN=$MD/train/runs/visual/uav-yolov8s-bg
L1=$RUN/it65_chain.log
L2=$RUN/it65_chain2.log
declare -A SEEN

STEPS=("шаг 1" "шаг 2" "шаг 3" "шаг 4a" "шаг 4b" "шаг 5" "шаг 6")

steps_all_ok() {
  grep -q "цепочка it-65 завершена" "$L1" 2>/dev/null || return 1
  grep -q "цепочка 2 завершена" "$L2" 2>/dev/null || return 1
  for s in "${STEPS[@]}"; do
    grep -qF "$s OK" "$L1" "$L2" 2>/dev/null || return 1
  done
  return 0
}

partial_risk() {  # шаг упал, но его инкрементальный CSV уже на диске — сигнал «не доверять E4/E5»
  grep -qF "шаг 3 FAIL" "$L1" 2>/dev/null && [ -f "$ROOT/research/mmaud_sahi_full_new.csv" ] \
    && echo "mmaud_sahi_full_new.csv частичный (SAHI упал) — E5 нечитать"
  grep -qF "шаг 5 FAIL" "$L2" 2>/dev/null && [ -f "$ROOT/research/session_vis_probe_new-yolov8s-bg.csv" ] \
    && echo "session_vis_probe_new.csv частичный (probe упал) — E4 нечитать"
}

emit() {  # emit <ключ> <текст>
  [ -n "${SEEN[$1]:-}" ] && return
  SEEN[$1]=1
  echo "[$(date '+%F %T')] $2"
}

while true; do
  if steps_all_ok; then
    emit ready "ВЕРДИКТ-ГОТОВ: все 7 шагов OK (строгий критерий v2) — запускать research/.venv/bin/python research/it65_verdict.py"
    exit 0
  fi
  while read -r step; do emit "fail:$step" "ШАГ-FAIL: $step"; done \
    < <(grep -hoE "шаг [0-9]+[ab]? FAIL" "$L1" "$L2" 2>/dev/null | sort -u)
  while read -r step; do emit "ok:$step" "шаг OK: $step"; done \
    < <(grep -hoE "шаг [0-9]+[ab]? OK" "$L1" "$L2" 2>/dev/null | sort -u)
  while read -r warn; do emit "partial" "ВНИМАНИЕ-ЧАСТИЧНЫЙ: $warn"; done < <(partial_risk)
  if ! pgrep -f "it65_resume_train|uavtrain.cli train|it65_chain.sh" >/dev/null; then
    emit dead "ТРЕНИРОВКА-СТОП: ни тренировки, ни вотчеров нет; результатов мало ($(wc -l < "$RUN/results.csv" 2>/dev/null || echo 0) строк) — нужен ручной разбор"
    exit 1
  fi
  n=$(wc -l < "$RUN/results.csv" 2>/dev/null || echo 0)
  emit "epoch:$n" "прогресс: записано $((n - 1)) эпох; цепочка ещё не даёт вердикт"
  sleep 600
done
