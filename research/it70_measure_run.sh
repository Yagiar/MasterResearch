#!/bin/bash
# it-70: цепочка замеров E1–E8 после завершения тренировки uav-yolov8s-bg70 (watcher, detached).
# Критерии предрегистрированы: research/iterations/it-70-corpus-negative-dilution-PLANNED.md.
# Ждёт финальный маркер тренировки ("Results saved" в it70_train.out — покрывает и early
# stopping), затем: GPU-шаги (eval600, FP-400, SAHI, sandbox-кадры), CPU-шаги (session probe,
# гейт-предфильтр 0,25 по образцу it-69 V3, 5 fusion-скриптов), манифест, вердикт-машинка.
# Логи — в каталоге прогона (НЕ /tmp). Никаких авто-экспортов: решение только за вердиктом + автором.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
RUN=$MD/train/runs/visual/uav-yolov8s-bg70
LOG=$RUN/it70_measure.log
TLOG=$MD/train/runs/visual/it70_train.out
BEST=$RUN/weights/best.pt
PY="$ROOT/research/.venv/bin/python"
GATE=0.25
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "watcher запущен; жду 'Results saved' в it70_train.out"
until grep -q "Results saved" "$TLOG" 2>/dev/null; do sleep 300; done
sleep 60
[ -f "$BEST" ] || { say "ОТКАЗ: нет $BEST"; exit 1; }
say "start; best sha256=$(sha256sum "$BEST" | cut -d' ' -f1 | cut -c1-12); gate=$GATE"

cd "$MD" || { say "нет каталога MD"; exit 1; }
say "E2: eval bg70 на DUT-test600"
./venv/bin/python -m uavtrain.cli eval-visual --weights "$BEST" \
  --data train/data/_prepared/visual-dut-test600/data.yaml --imgsz 640 --device 0 --name bg70-dut600 >> "$LOG" 2>&1 \
  && say "E2 OK" || say "E2 FAIL"

say "E3: eval bg70 на HF-test600"
./venv/bin/python -m uavtrain.cli eval-visual --weights "$BEST" \
  --data train/data/_prepared/visual-hf-test600/data.yaml --imgsz 640 --device 0 --name bg70-hf600 >> "$LOG" 2>&1 \
  && say "E3 OK" || say "E3 FAIL"

say "E1: FP на независимых 400 фонах (coco-bg-v1)"
"$PY" "$ROOT/research/coco_bg_fp_eval.py" --weights "$BEST" --name it70v1 \
  --images-dir "$MD/train/data/_prepared/coco-bg-v1" --pattern '*.jpg' --device 0 >> "$LOG" 2>&1 \
  && say "E1 OK" || say "E1 FAIL"

say "E5: SAHI MMAUD полными весами bg70"
"$PY" "$ROOT/research/mmaud_sahi_full.py" --weights "$BEST" \
  --out "$ROOT/research/mmaud_sahi_full_bg70.csv" >> "$LOG" 2>&1 \
  && say "E5 OK" || say "E5 FAIL"

say "E6-8 шаг a: sandbox-кадры новыми весами"
"$PY" "$ROOT/research/yolo_frames_eval.py" --weights "$BEST" \
  --out "$ROOT/research/yolo_sandbox_frames_bg70.csv" >> "$LOG" 2>&1 \
  && say "E6-8a OK" || say "E6-8a FAIL"

say "E4: session probe (стоящий дрон, CPU)"
"$PY" "$ROOT/research/session_vis_probe.py" --weights "$BEST" --name bg70 >> "$LOG" 2>&1 \
  && say "E4 OK" || say "E4 FAIL"

say "E6-8 шаг b: гейт-предфильтр $GATE (как в проде visual-detector)"
"$PY" - "$GATE" >> "$LOG" 2>&1 <<EOF || { say "E6-8b FAIL"; exit 1; }
import csv, sys
gate = float(sys.argv[1])
src = "$ROOT/research/yolo_sandbox_frames_bg70.csv"
dst = "$ROOT/research/yolo_sandbox_frames_bg70_gate025.csv"
rows = list(csv.DictReader(open(src)))
n = 0
for r in rows:
    if float(r["max_conf"]) < gate:
        r["max_conf"] = "0.0"; r["n_det"] = "0"; n += 1
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"гейт {gate}: обнулено {n}/{len(rows)} строк; файл {dst}")
EOF
say "E6-8b OK"

cd "$ROOT" || { say "нет ROOT для fusion"; exit 1; }
say "E6-8 шаг c: пять fusion-скриптов (--suffix=-bg70)"
for s in fusion_sim_full stress_sim event_metrics threshold_calibration_v2 bootstrap_delta_v2; do
  "$PY" "$ROOT/research/$s.py" --yolo-csv "research/yolo_sandbox_frames_bg70_gate025.csv" --suffix=-bg70 >> "$LOG" 2>&1 \
    && say "E6-8c $s OK" || say "E6-8c $s FAIL"
done

say "манифест + вердикт"
"$PY" "$ROOT/research/make_manifest.py" >> "$LOG" 2>&1 || say "манифест FAIL (предупреждение про visual-new-dut600-cpu ожидаемо)"
"$PY" "$ROOT/research/it70_verdict.py" >> "$LOG" 2>&1
say "ВЕРДИКТ exit=$?"
say "цепочка it-70 завершена (экспорт — только ручным решением автора)"
