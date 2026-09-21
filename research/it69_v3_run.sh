#!/bin/bash
# it-69 V3: контур it-66-протоколом на НОВЫХ весах с детекторным гейтом 0,4.
# Протокол и критерии — research/iterations/it-69-detector-gate-recalibration-PLANNED.md
# (предрегистрированы 2026-09-21 до запуска; уточнение плеч — там же, блок «Уточнение протокола ДО прогонов»).
# Механика: per-frame max_conf новых весов (гейт инференса 0,05, yolo_sandbox_frames_new.csv)
# предфильтруется гейтом 0,4 (max_conf<0,4 -> 0, n_det->0) — это ровно то, что делает
# conf_threshold в visual-detector; пять fusion-скриптов гоняются по протоколу it-66.
# Тихо: CPU, новые прогоны инференса НЕ выполняются. Логи: research/it69_v3_run.log.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
PY="$ROOT/research/.venv/bin/python"
LOG="$ROOT/research/it69_v3_run.log"
GATE=0.4
WEIGHTS="$MD/train/runs/visual/uav-yolov8s-bg/weights/best.pt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -f "$ROOT/research/yolo_sandbox_frames_new.csv" ] || { say "ОТКАЗ: нет yolo_sandbox_frames_new.csv (шаг 1 it-66) — V3 не на чем считать"; exit 1; }
say "V3 start; weights=$WEIGHTS sha256=$(sha256sum "$WEIGHTS" | cut -d' ' -f1 | cut -c1-12); gate=$GATE"

say "шаг 1: гейт-предфильтр CSV (max_conf<$GATE -> 0)"
"$PY" - "$GATE" >> "$LOG" 2>&1 <<'EOF' || { say "шаг 1 FAIL"; exit 1; }
import csv, sys
gate = float(sys.argv[1])
src = "/home/otrix/code/GeneralFolderMasterDiploma/research/yolo_sandbox_frames_new.csv"
dst = "/home/otrix/code/GeneralFolderMasterDiploma/research/yolo_sandbox_frames_new_gate04.csv"
rows = list(csv.DictReader(open(src)))
n = 0
for r in rows:
    if float(r["max_conf"]) < gate:
        r["max_conf"] = "0.0"; r["n_det"] = "0"; n += 1
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"гейт {gate}: обнулено {n}/{len(rows)} строк (обе imgsz); файл {dst}")
EOF
say "шаг 1 OK"

say "шаг 2: пять fusion-скриптов (--suffix=-new-gate04)"
for s in fusion_sim_full stress_sim event_metrics threshold_calibration_v2 bootstrap_delta_v2; do
  "$PY" "$ROOT/research/$s.py" --yolo-csv "research/yolo_sandbox_frames_new_gate04.csv" --suffix=-new-gate04 >> "$LOG" 2>&1 \
    && say "шаг 2 $s OK" || say "шаг 2 $s FAIL"
done

say "цепочка V3 завершена — сверка критериев V3 (recall ≥ old−0,02; F1 поздн. ≥ 0,95; задержка ≤ old+1,0 с) по *-new-gate04.csv против канона и it-66"
