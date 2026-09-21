#!/bin/bash
# it-70 плечо S: ПОЛНЫЙ протокол E1–E8 для финалиста полки (после MMAUD-арбитража).
# Все шаги CPU (GPU — под учёбой автора). E2/E3 (eval-visual), E4 (session), E5 уже посчитаны
# скринингом/арбитром для dut/hf/sahi — здесь пересобираем их в вердикт-формат и добавляем
# недостающее: E1 per-image CSV, E4, sandbox-кадры → гейт 0,25 → 5 fusion-скриптов → вердикт.
# Запуск: bash research/it70_shelf_fullprotocol.sh <файл_весов_в_shelf_models> <prefix-метка>
#   пример: bash research/it70_shelf_fullprotocol.sh iris_v8s.pt iris-v8s
#   ВАЖНО: prefix = метка БЕЗ «shelf-» (как в выходах арбитра mmaud_sahi_full_<label>.csv),
#   иначе вердикт не найдёт E5.
# Логи: research/shelf_fullprotocol_<prefix>.log. Никаких авто-экспортов.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
PY="$ROOT/research/.venv/bin/python"
PT="${1:?usage: $0 <pt-файл> <prefix>}"
PREFIX="${2:?usage: $0 <pt-файл> <prefix>}"
W="$ROOT/research/shelf_models/$PT"
LOG="$ROOT/research/shelf_fullprotocol_${PREFIX}.log"
GATE=0.25
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
[ -f "$W" ] || { say "ОТКАЗ: нет $W"; exit 1; }
say "старт; weights=$W sha=$(sha256sum "$W" | cut -d' ' -f1 | cut -c1-12) prefix=$PREFIX"

say "E1: FP per-image на 400 независимых фонах (coco-bg-v1, CPU)"
"$PY" "$ROOT/research/coco_bg_fp_eval.py" --weights "$W" --name "$PREFIX" \
  --images-dir "$MD/train/data/_prepared/coco-bg-v1" --pattern '*.jpg' --device cpu >> "$LOG" 2>&1 \
  && say "E1 OK" || say "E1 FAIL"

say "E4: session probe (стоящий дрон, CPU)"
"$PY" "$ROOT/research/session_vis_probe.py" --weights "$W" --name "$PREFIX" --device cpu >> "$LOG" 2>&1 \
  && say "E4 OK" || say "E4 FAIL"

say "sandbox-кадры (full+SAHI по стробоскоп-клипу, CPU)"
"$PY" "$ROOT/research/yolo_frames_eval.py" --weights "$W" \
  --out "$ROOT/research/yolo_sandbox_frames_${PREFIX}.csv" >> "$LOG" 2>&1 \
  && say "sandbox OK" || say "sandbox FAIL"

say "гейт-предфильтр $GATE (урок V3: max_conf<$GATE -> 0)"
"$PY" - "$PREFIX" "$GATE" >> "$LOG" 2>&1 <<'EOF' || { say "гейт FAIL"; exit 1; }
import csv, sys
prefix, gate = sys.argv[1], float(sys.argv[2])
root = "/home/otrix/code/GeneralFolderMasterDiploma/research"
src, dst = f"{root}/yolo_sandbox_frames_{prefix}.csv", f"{root}/yolo_sandbox_frames_{prefix}_gate025.csv"
rows = list(csv.DictReader(open(src)))
n = 0
for r in rows:
    if float(r["max_conf"]) < gate:
        r["max_conf"] = "0.0"; r["n_det"] = "0"; n += 1
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"гейт {gate}: обнулено {n}/{len(rows)}; {dst}")
EOF
say "гейт OK"

say "контур E6–E8: пять fusion-скриптов (--suffix=-$PREFIX)"
for s in fusion_sim_full stress_sim event_metrics threshold_calibration_v2 bootstrap_delta_v2; do
  "$PY" "$ROOT/research/$s.py" --yolo-csv "research/yolo_sandbox_frames_${PREFIX}_gate025.csv" --suffix=-"$PREFIX" >> "$LOG" 2>&1 \
    && say "контур $s OK" || say "контур $s FAIL"
done

say "вердикт E1–E8 (предрег. it-70)"
"$PY" "$ROOT/research/it70_verdict.py" --prefix "$PREFIX" --e1 "coco_bg_fp_${PREFIX}.csv" | tee -a "$LOG"
say "ПОЛНЫЙ ПРОТОКОЛ ПЛЕЧА S ЗАВЕРШЁН ($PREFIX) — E2/E3 в вердикте ЖДЁТ по именам eval-каталогов;"
say "числа dut/hf — в shelf_screen_results.csv (тот же val test600), перенести вручную в отчёт"
