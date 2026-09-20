#!/bin/bash
# it-66: пересчёт fusion-контура НОВЫМИ весами (протокол зафиксирован в
# research/iterations/it-66-fusion-new-weights-PLANNED.md; критерии T1–T5 предрегистрированы).
# Запускать ТОЛЬКО после зелёного вердикта it-65 (шаг экспорт завершён) — скрипт проверяет
# предпосылки и отказывается работать иначе. Нагрузка: шаг 1 = CPU-инференс 73 с видео
# (2 порога × 2 imgsz) — не совмещать с GPU-цепочками it-65.
# Логи: research/it66_run.log (рядом со скриптом, не /tmp).
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
PY="$ROOT/research/.venv/bin/python"
WEIGHTS="${1:-$MD/models/visual/yolov8s-uav.pt}"
LOG="$ROOT/research/it66_run.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# предпосылки: вердикт it-65 зелёный (артефакты цепочек есть) и веса — экспорт it-65
[ -f "$ROOT/research/coco_bg_fp_new-yolov8s-bg.csv" ] || { say "ОТКАЗ: нет артефактов it-65 (coco_bg_fp_new) — сначала цепочки+вердикт"; exit 1; }
[ -f "$ROOT/research/mmaud_sahi_full_new.csv" ] || { say "ОТКАЗ: нет SAHI-замера new — цепочка 1 не завершена"; exit 1; }
sha_now=$(sha256sum "$WEIGHTS" | cut -d' ' -f1)
sha_best=$(sha256sum "$MD/train/runs/visual/uav-yolov8s-bg/weights/best.pt" | cut -d' ' -f1)
[ "$sha_now" = "$sha_best" ] || say "ВНИМАНИЕ: $WEIGHTS (sha ${sha_now:0:8}) ≠ best.pt (sha ${sha_best:0:8}) — экспорт ещё не выполнен или файл менялся после экспорта; продолжаем с указанными весами, в отчёт внести фактический sha"
say "it-66 run start; weights=$WEIGHTS sha256=${sha_now:0:12}"

say "шаг 1: yolo_frames_eval новыми весами (CPU)"
"$PY" "$ROOT/research/yolo_frames_eval.py" --weights "$WEIGHTS" \
  --out "$ROOT/research/yolo_sandbox_frames_new.csv" >> "$LOG" 2>&1 \
  && say "шаг 1 OK" || say "шаг 1 FAIL"

say "шаг 2: пять fusion-скриптов (--suffix=-new)"
for s in fusion_sim_full stress_sim event_metrics threshold_calibration_v2 bootstrap_delta_v2; do
  "$PY" "$ROOT/research/$s.py" --yolo-csv research/yolo_sandbox_frames_new.csv --suffix=-new >> "$LOG" 2>&1 \
    && say "шаг 2 $s OK" || say "шаг 2 $s FAIL"
done

say "шаг 3: перегенерация манифеста (7 ключей *-new пре-зарегистрированы)"
"$PY" "$ROOT/research/make_manifest.py" >> "$LOG" 2>&1 \
  && say "шаг 3 OK" || say "шаг 3 FAIL"

say "цепочка it-66 завершена — сверка T1–T5 по таблице протокола (ручной шаг: читать it66_run.log и *-new.csv)"
