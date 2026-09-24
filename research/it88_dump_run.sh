#!/usr/bin/env bash
# it-88 — 4 GPU-дампа тайловой агрегации (строго после it-85; см. it-88-tile-aware-sahi-PLANNED.md).
# bg400 — FP-ось (400 HD-фонов OI), mmaud stride=10 — recall-guardrail; ряды old/new.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
cd "$ROOT"
PY=research/.venv/bin/python
OLD=MasterDiploma/models/visual/yolov8s-uav.pt
NEW=MasterDiploma/models/visual/uav-yolov8s-bg-best.pt
[ -f "$OLD" ] && [ -f "$NEW" ] || { echo "ОТКАЗ: нет весов"; exit 1; }
[ -f research/it85_sweep_summary.txt ] || { echo "ОТКАЗ: it-85 не закрыт (нет свода)"; exit 1; }

run() {  # $1..: аргументы harness'а
  echo "=== it88 dump $* $(date '+%F %T') ==="
  "$PY" research/it88_sahi_tiled.py "$@" || { echo "ОТКАЗ: $*"; exit 1; }
}

run --source bg400 --weights "$OLD" --out research/it88_bg400_old.csv --device cuda
run --source bg400 --weights "$NEW" --out research/it88_bg400_new.csv --device cuda
run --source mmaud --weights "$OLD" --out research/it88_mmaud_old.csv --device cuda --stride 10
run --source mmaud --weights "$NEW" --out research/it88_mmaud_new.csv --device cuda --stride 10

for f in research/it88_bg400_old.csv research/it88_bg400_new.csv research/it88_mmaud_old.csv research/it88_mmaud_new.csv; do
  n=$(( $(wc -l < "$f") - 1 ))
  echo "[it88] $f: $n строк"
done
awk 'END{if (NR-1 != 400) {exit 1}}' research/it88_bg400_old.csv || { echo "ОТКАЗ: bg400_old != 400"; exit 1; }
awk 'END{if (NR-1 != 400) {exit 1}}' research/it88_bg400_new.csv || { echo "ОТКАЗ: bg400_new != 400"; exit 1; }

echo "=== дампы готовы; разбор: $PY research/it88_tile_aggregate.py ==="
