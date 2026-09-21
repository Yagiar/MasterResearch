#!/bin/bash
# it-70 плечо S, финалисты: ждёт завершения MMAUD-арбитра, затем для КАЖДОГО лидера с
# независимым E5 (SAHI-полёт MMAUD recall @0,5) ≥ 0,895 (предрег. порог it-70) автозапускает
# полный CPU-протокол E1–E8 (it70_shelf_fullprotocol.sh). Лидеры ниже E5 не удостаиваются
# контура — критерий отсечения предрегистрирован, авто-решение соответствует протоколу.
# Идемпотентно: выход вердикта (shelf_fullprotocol_<label>.log с "ЗАВЕРШЁН") — уже сделано.
# Запуск (detached): setsid nohup bash research/it70_shelf_finalists.sh </dev/null & disown
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
PY="$ROOT/research/.venv/bin/python"
LOG="$ROOT/research/shelf_finalists.log"
E5=0.895
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "вотчер финалистов: жду «MMAUD-арбитр плеча S завершён» в shelf_mmaud.log"
until grep -q "MMAUD-арбитр плеча S завершён" "$ROOT/research/shelf_mmaud.log" 2>/dev/null; do sleep 300; done
say "арбитраж завершён; считаю E5 по mmaud_sahi_full_<label>.csv лидеров"

"$PY" - "$E5" >> "$LOG" 2>&1 <<'EOF'
import bisect, csv, glob, sys
from pathlib import Path
import numpy as np
root = Path("/home/otrix/code/GeneralFolderMasterDiploma")
md = root / "MasterDiploma"
e5 = float(sys.argv[1])
gt_files = sorted(glob.glob(str(md / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt_z = [float(np.load(g)[2]) for g in gt_files]
for p in sorted(glob.glob(str(root / "research/mmaud_sahi_full_*.csv"))):
    label = Path(p).stem.removeprefix("mmaud_sahi_full_")
    if label in ("", "new", "bg70") or "gate" in label:
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    flight = [r for r in rows if gt_z[min(len(gt_z) - 1, bisect.bisect_left(gt_ts, float(r["img"])))] > 1.0]
    rec = sum(1 for r in flight if float(r["conf_sahi"]) >= 0.5) / max(1, len(flight))
    print(f"FINALIST\t{label}\t{rec:.4f}\t{'yes' if rec >= e5 else 'no'}")
EOF

grep -P '^FINALIST\t.*\tyes$' "$LOG" | awk -F'\t' '!seen[$2]++{print $2"\t"$3}' | while IFS=$'\t' read -r label rec; do
  case "$label" in
    doguilmak-v11x) PT=doguilmak_v11x.pt;; doguilmak-v8x) PT=doguilmak_v8x.pt;;
    skyguard-v11) PT=skyguard_v11.pt;; ruju-v12) PT=ruju_v12.pt;;
    danivelikova-v26n) PT=danivelikova_v26n.pt;; iris-v8s) PT=iris_v8s.pt;;
    noah-v8s) PT=noah_v8s.pt;;
    *) say "нет маппинга для $label — пропуск (вручную)"; continue;;
  esac
  if grep -q "ПОЛНЫЙ ПРОТОКОЛ ПЛЕЧА S ЗАВЕРШЁН" "$ROOT/research/shelf_fullprotocol_${label}.log" 2>/dev/null; then
    say "$label (E5=$rec): протокол уже завершён — пропуск"; continue
  fi
  say "$label (E5=$rec ≥ $E5): старт полного протокола E1–E8"
  bash "$ROOT/research/it70_shelf_fullprotocol.sh" "$PT" "$label" >> "$LOG" 2>&1 \
    && say "$label: протокол OK" || say "$label: протокол FAIL"
done
say "Финалисты плеча S обработаны — сверка: research/it70_shelf_report.py + вердикты в shelf_fullprotocol_*.log"
