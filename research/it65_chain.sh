#!/bin/bash
# it-65: цепочка оценки после завершения тренировки YOLOv8s (watcher, detached).
# Ждёт конца тренировки (results.csv >= 31 строки ИЛИ процесс пропал), затем по очереди:
#   1) eval-visual новой модели на test (GPU)
#   2) FP на COCO-фонах test новыми весами (GPU)
#   3) полный SAHI-прогон MMAUD новыми весами (GPU)
# Экспорт весов и финальный отчёт — сознательно НЕ здесь: решение после просмотра метрик.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
RUN=$MD/train/runs/visual/uav-yolov8s-bg
LOG=$RUN/it65_chain.log   # не /tmp — переживает перезагрузку
BEST=$RUN/weights/best.pt

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "watcher запущен; жду завершения тренировки"
while true; do
  if ! pgrep -f "uavtrain.cli|it65_resume_train" > /dev/null; then
    say "процесс тренировки не найден — считаем завершённой"
    break
  fi
  n=$(wc -l < "$RUN/results.csv" 2>/dev/null || echo 0)
  if [ "$n" -ge 31 ]; then
    say "results.csv: $n строк (30 эпох + заголовок) — завершено"
    break
  fi
  sleep 300
done

cd "$MD" || { say "нет каталога MD"; exit 1; }
say "шаг 1: eval-visual новой модели на test"
./venv/bin/python -m uavtrain.cli eval-visual \
  --weights "$BEST" --data train/data/_prepared/visual/data.yaml \
  --imgsz 640 --device 0 --name bg-new >> "$LOG" 2>&1 \
  && say "шаг 1 OK" || say "шаг 1 FAIL"

say "шаг 2: FP на COCO-фонах test (new)"
"$ROOT/research/.venv/bin/python" "$ROOT/research/coco_bg_fp_eval.py" \
  --weights "$BEST" --name new-yolov8s-bg --device 0 >> "$LOG" 2>&1 \
  && say "шаг 2 OK" || say "шаг 2 FAIL"

say "шаг 3: SAHI MMAUD полными весами (new)"
"$ROOT/research/.venv/bin/python" "$ROOT/research/mmaud_sahi_full.py" \
  --weights "$BEST" --out "$ROOT/research/mmaud_sahi_full_new.csv" >> "$LOG" 2>&1 \
  && say "шаг 3 OK" || say "шаг 3 FAIL"

say "цепочка it-65 завершена (см. метрики в логе; экспорт и отчёт — ручным шагом)"
