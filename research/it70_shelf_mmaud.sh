#!/bin/bash
# it-70 плечо S, арбитр: независимая MMAUD-проверка (E5-линейка, SAHI 640/0.2, conf>=0.5)
# для ЛИДЕРОВ shelf-скрининга. Критерий лидера (предрег. в плане it-70, блок плеча S):
# в топ-3 по скрининг-сумме (dut600+hf600 mAP50 при FP@0,5 < old 27,5%).
# CPU принципиально (GPU — под учёбой автора); --full1920-csv none (conf_union shelf-моделей
# не сравниваем — только conf_sahi, урок it-65 про смесь весов).
# Запуск: bash research/it70_shelf_mmaud.sh            (после "SHELF SCREEN DONE")
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
PY="$ROOT/research/.venv/bin/python"
CSV="$ROOT/research/shelf_screen_results.csv"
LOG="$ROOT/research/shelf_mmaud.log"
MAX_MODELS=3
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

grep -q 'SHELF SCREEN DONE' "$ROOT/research/shelf_screen.log" || { say "ОТКАЗ: скрининг не завершён"; exit 1; }

say "выбор лидеров из $CSV"
"$PY" - <<'EOF' | tee -a "$LOG"
import csv
rows = [r for r in csv.DictReader(open("/home/otrix/code/GeneralFolderMasterDiploma/research/shelf_screen_results.csv"))]
def num(x):
    try: return float(x)
    except ValueError: return None
old = next(r for r in rows if r["model"] == "old-uav-yolov8s")
cands = []
for r in rows:
    if not r["model"].startswith("shelf-"):
        continue
    d, h, fp = num(r["dut600_map50"]), num(r["hf600_map50"]), num(r["fp@0.5"])
    if d is None or h is None:
        continue
    cands.append((d + h, r["model"], d, h, fp))
cands.sort(reverse=True)
top = [c for c in cands if c[4] is not None and c[4] < float(old["fp@0.5"])][:3]
if not top:  # никто не прошёл фильтр FP — берём 2 лучших по сумме с пометкой
    top = [c for c in cands if c[4] is not None][:2]
    if top:
        print("NOTE\tни одна полочная модель не прошла FP-фильтр — берём 2 лучших по сумме (с пометкой)")
if not top:
    print("NOTE\tлидеров нет вовсе (все строки shelf с пропусками) — арбитр нечего запускать")
for s, m, d, h, fp in top:
    print(f"LEADER\t{m}\t{d:.4f}\t{h:.4f}\t{fp:.3f}")
EOF
n_lead=$(grep -c '^LEADER' "$LOG" || true)
[ "$n_lead" -eq 0 ] && { say "лидеров нет — завершаюсь (см. NOTE в логе)"; exit 0; }

awk -F'\t' '$1=="LEADER"{ if (!seen[$2]++) print $2 }' "$LOG" | tail -n "$MAX_MODELS" | while read -r label; do
  pt="${label#shelf-}"
  case "$pt" in
    doguilmak-v11x) F=doguilmak_v11x.pt;; doguilmak-v8x) F=doguilmak_v8x.pt;;
    skyguard-v11) F=skyguard_v11.pt;; ruju-v12) F=ruju_v12.pt;;
    danivelikova-v26n) F=danivelikova_v26n.pt;; iris-v8s) F=iris_v8s.pt;; noah-v8s) F=noah_v8s.pt;;
    *) say "неизвестная модель $pt — пропуск"; continue;;
  esac
  OUT="$ROOT/research/mmaud_sahi_full_${pt}.csv"
  [ -f "$OUT" ] && { say "$pt: $OUT уже есть — пропуск"; continue; }
  say "$pt: старт MMAUD SAHI (CPU, ~3-5 ч)"
  "$PY" "$ROOT/research/mmaud_sahi_full.py" --weights "$ROOT/research/shelf_models/$F" \
    --out "$OUT" --device cpu --full1920-csv none >> "$LOG" 2>&1 \
    && say "$pt: OK (см. сводку recall выше)" || say "$pt: FAIL"
done
say "MMAUD-арбитр плеча S завершён"
