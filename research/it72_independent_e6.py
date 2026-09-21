#!/usr/bin/env python3
"""it-72: analogue E6-контура на независимой сессии MMAUD — НОЛЬ инференса.

1-сек сетка; pv = max conf_sahi в бине (SAHI-протокол E5); гейт g: pv<g→0;
состояния — лидарный GT (airborne z>1, ground z<0.5). Критерии C1–C4
предрегистрированы в it-72-...-PLANNED.md до запуска.

Запуск: research/.venv/bin/python research/it72_independent_e6.py
"""
import bisect
import csv
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"

gt_files = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt = [np.load(g) for g in gt_files]


def z_at(t: float) -> float:
    return float(gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))][2])


def bins_from(csv_path: str) -> dict[int, float]:
    """unix-секунда → max conf_sahi в бине."""
    out: dict[int, float] = {}
    for r in csv.DictReader(open(ROOT / csv_path, encoding="utf-8")):
        s = int(float(r["img"]))
        out[s] = max(out.get(s, 0.0), float(r["conf_sahi"]))
    return out


WEIGHTS = {"old": "research/mmaud_sahi_full.csv", "new": "research/mmaud_sahi_full_new.csv"}
GATES = (0.25, 0.4, 0.5)

# состояние бина по крышному GT (без заимств conf)
all_bins: dict[int, float] = {}
for path in WEIGHTS.values():
    for s, c in bins_from(path).items():
        all_bins[s] = max(all_bins.get(s, 0.0), c)
t0s, t1s = min(all_bins), max(all_bins)
state = {s: ("air" if z_at(s) > 1.0 else "ground" if z_at(s) < 0.5 else "trans") for s in range(t0s, t1s + 1)}
air = [s for s in sorted(state) if state[s] == "air"]
grnd = [s for s in sorted(state) if state[s] == "ground"]
# airborne-события: непрерывные участки
events, cur = [], None
for s in sorted(state):
    if state[s] == "air":
        cur = [s, s] if cur is None else [cur[0], s]
    elif cur is not None:
        if s - cur[1] > 1:  # разрыв ≤1 с не считаем новым событием
            events.append(tuple(cur))
        else:
            cur[1] = s
if cur is not None:
    events.append(tuple(cur))
print(f"бинов {t0s}–{t1s}: airborne {len(air)}, ground {len(grnd)}, события {events}")

print(f"\n{'веса':>5} {'g':>5} {'rec(air)':>9} {'FN':>4} {'FP(ground)':>11} {'задержка med':>13}")
res = {}
for wtag, path in WEIGHTS.items():
    bins = bins_from(path)
    for g in GATES:
        pos = {s for s in state if bins.get(s, 0.0) >= g and bins.get(s, 0.0) != 0.0}
        tp = sum(1 for s in air if s in pos)
        fn = len(air) - tp
        fp = sum(1 for s in grnd if s in pos)
        delays = []
        for es, ee in events:
            hits = [s for s in sorted(pos) if es <= s <= ee + 2]
            if hits:
                delays.append(hits[0] - es)
        med = sorted(delays)[len(delays) // 2] if delays else float("nan")
        rec, fpr = tp / max(1, len(air)), fp / max(1, len(grnd))
        res[(wtag, g)] = (rec, fpr, med, fn, fp, len(delays))
        print(f"{wtag:>5} {g:>5} {rec:>8.1%} {fn:>4} {fpr:>6.1%} ({fp:>2}) {med:>9.1f} с ({len(delays)}/{len(events)})")

# справочно: late D0 = 0.5·pv_g + 0.5·pa ≥ 0.5 (аудио-канал mmaud не переносится, it-57)
pa_bin: dict[int, float] = {}
for r in csv.DictReader(open(ROOT / "research/mmaud_acoustic_eval.csv", encoding="utf-8")):
    sb = int(float(r["t_unix"]))
    pa_bin[sb] = max(pa_bin.get(sb, 0.0), float(r["p_drone"]))
for wtag, g in (("old", 0.25), ("new", 0.4)):
    bins = bins_from(WEIGHTS[wtag])
    pos = {s for s in state if 0.5 * max(bins.get(s, 0.0), 0.0 if bins.get(s, 0.0) < g else bins.get(s, 0.0))
           + 0.5 * pa_bin.get(s, 0.0) >= 0.5}
    tp = sum(1 for s in air if s in pos); fp = sum(1 for s in grnd if s in pos)
    print(f"  [справочно] lateD0 {wtag}@{g}: rec={tp / max(1, len(air)):.1%} FP={fp / max(1, len(grnd)):.1%}")

o25, n40 = res[("old", 0.25)], res[("new", 0.4)]
print(f"\nC1 (new@0,25 < old@0,25−0,02): {res[('new', 0.25)][0]:.1%} vs {o25[0]:.1%} → "
      f"{'ПОДТВЕРЖДЁН (потеря обобщается)' if res[('new', 0.25)][0] < o25[0] - 0.02 else 'НЕ подтверждён'}")
print(f"C2 (new@0,4 ≥ old@0,25−0,02):  {n40[0]:.1%} ≥ {o25[0] - 0.02:.1%} → {'OK' if n40[0] >= o25[0] - 0.02 else 'КРАСНЫЙ'}")
print(f"C3 (FP new@0,4 ≤ FP old@0,25+0,05): {n40[1]:.1%} ≤ {o25[1] + 0.05:.1%} → {'OK' if n40[1] <= o25[1] + 0.05 else 'КРАСНЫЙ'}")
print(f"C4 (задержка new@0,4 ≤ old@0,25+1,0 с): {n40[2]:.1f} ≤ {o25[2] + 1.0:.1f} → {'OK' if n40[2] <= o25[2] + 1.0 else 'КРАСНЫЙ'}")
verdict = all([n40[0] >= o25[0] - 0.02, n40[1] <= o25[1] + 0.05, n40[2] <= o25[2] + 1.0])
print(f"\nВЕРДИТ: пара (new, 0,4) {'ПОДТВЕРЖДЕНА' if verdict else 'НЕ подтверждена'} "
      f"на независимой видеосессии (экспорт по-прежнему не разрешён; см. ограничения в PLANNED)")
