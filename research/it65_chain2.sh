#!/bin/bash
# it-65, цепочка 2: догоняющие GPU-замеры НОВЫМИ весами после основной it65_chain.sh.
# Ждёт финальную строку в логе первой цепочки, затем:
#   4) eval новой модели на подвыборках DUT/HF-600 (прямое сравнение со старой, baseline 0,720/0,867)
#   5) контроль «стоящий дрон» на негативной сессии (baseline: 18/18 сек, conf 0,83–0,86)
#   6) перегенерация манифеста (все CSV цепочек на месте)
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
RUN=$MD/train/runs/visual/uav-yolov8s-bg
LOG=$RUN/it65_chain2.log
CHAIN1=$RUN/it65_chain.log
BEST=$RUN/weights/best.pt

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "watcher2 запущен; жду завершения основной цепочки"
until grep -q "цепочка it-65 завершена" "$CHAIN1" 2>/dev/null; do
  sleep 120
done
say "основная цепочка завершена — старт догоняющих замеров"

cd "$MD" || { say "нет каталога MD"; exit 1; }
say "шаг 4a: eval new на DUT-test600"
./venv/bin/python -m uavtrain.cli eval-visual \
  --weights "$BEST" --data train/data/_prepared/visual-dut-test600/data.yaml \
  --imgsz 640 --device 0 --name new-dut600 >> "$LOG" 2>&1 \
  && say "шаг 4a OK" || say "шаг 4a FAIL"

say "шаг 4b: eval new на HF-test600"
./venv/bin/python -m uavtrain.cli eval-visual \
  --weights "$BEST" --data train/data/_prepared/visual-hf-test600/data.yaml \
  --imgsz 640 --device 0 --name new-hf600 >> "$LOG" 2>&1 \
  && say "шаг 4b OK" || say "шаг 4b FAIL"

say "шаг 5: session_vis_probe (стоящий дрон) новыми весами"
"$ROOT/research/.venv/bin/python" "$ROOT/research/session_vis_probe.py" \
  --weights "$BEST" --name new-yolov8s-bg --device 0 >> "$LOG" 2>&1 \
  && say "шаг 5 OK" || say "шаг 5 FAIL"

say "шаг 6: перегенерация манифеста"
"$ROOT/research/.venv/bin/python" "$ROOT/research/make_manifest.py" >> "$LOG" 2>&1 \
  && say "шаг 6 OK" || say "шаг 6 FAIL"

say "цепочка 2 завершена"
