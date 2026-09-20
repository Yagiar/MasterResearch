#!/bin/bash
# it-65: условный монитор точки принятия решения (заменяет холостые опросы цели).
# Печатает СТРОКУ-СОБЫТИЕ только когда состояние изменилось:
#   ВЕРДИКТ-ГОТОВ — все артефакты цепочек на месте (можно запускать it65_verdict.py)
#   ШАГ-FAIL — любой шаг цепочки упал
#   ТРЕНИРОВКА-СТОП — процесс обучения исчез, а результатов нет (зависание/убийство)
# 7 запланированных событий вместо десятков idle-тиканий.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
RUN=$MD/train/runs/visual/uav-yolov8s-bg
L1=$RUN/it65_chain.log
L2=$RUN/it65_chain2.log
READY=0
declare -A SEEN

has_all_artifacts() {
  for f in "$MD/train/runs/eval/visual-bg-new/metrics.json" \
           "$MD/train/runs/eval/visual-new-dut600/metrics.json" \
           "$MD/train/runs/eval/visual-new-hf600/metrics.json" \
           "$ROOT/research/coco_bg_fp_new-yolov8s-bg.csv" \
           "$ROOT/research/session_vis_probe_new-yolov8s-bg.csv" \
           "$ROOT/research/mmaud_sahi_full_new.csv"; do
    [ -f "$f" ] || return 1
  done
  return 0
}

emit() {  # emit <ключ> <текст>
  [ -n "${SEEN[$1]:-}" ] && return
  SEEN[$1]=1
  echo "[$(date '+%F %T')] $2"
}

while true; do
  if has_all_artifacts; then
    emit ready "ВЕРДИКТ-ГОТОВ: все 6 артефактов на месте — запускать research/.venv/bin/python research/it65_verdict.py"
    exit 0
  fi
  while read -r step; do emit "fail:$step" "ШАГ-FAIL: $step"; done \
    < <(grep -hoE "шаг [0-9]+[ab]? FAIL" "$L1" "$L2" 2>/dev/null | sort -u)
  if ! pgrep -f "it65_resume_train|uavtrain.cli train|it65_chain.sh" >/dev/null; then
    emit dead "ТРЕНИРОВКА-СТОП: ни тренировки, ни вотчеров нет; результатов мало ($(wc -l < "$RUN/results.csv" 2>/dev/null || echo 0) строк) — нужен ручной разбор"
    exit 1
  fi
  n=$(wc -l < "$RUN/results.csv" 2>/dev/null || echo 0)
  emit "epoch:$n" "прогресс: записано $((n - 1)) эпох; цепочка ещё не даёт вердикт"
  sleep 600
done
