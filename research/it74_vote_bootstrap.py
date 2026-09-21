#!/usr/bin/env python3
"""it-74 дополнение: интервальные запасы основной ячейки AND@0,40 (ноль инференса).

Критерии НЕ меняются (G1/G2 уже вынесены по точечным оценкам). Цель — честный
интервал: полёт MMAUD автокоррелирован по времени → блочный бутстрап (блок 100 кадров
в хронологическом порядке, 2000 реплик, сид 20260921); фоны — 400 независимых кадров →
интервал Уилсона 95 %. Выход: research/it74_vote_bootstrap.txt.

Запуск: research/.venv/bin/python research/it74_vote_bootstrap.py
"""
import bisect
import csv
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED, NREP, BLK = 20260921, 2000, 100
TAU = 0.40

gt_files = sorted(glob.glob(str(ROOT / "MasterDiploma/train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt = [np.load(g) for g in gt_files]


def gt_at(t: float) -> np.ndarray:
    return gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))]


def load_sahi(path: Path) -> dict[str, float]:
    return {r["img"]: float(r["conf_sahi"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


old_m = load_sahi(ROOT / "research/mmaud_sahi_full.csv")
new_m = load_sahi(ROOT / "research/mmaud_sahi_full_new.csv")
keys = sorted(set(old_m) & set(new_m), key=float)  # хронологический порядок
and_f = np.array([min(old_m[k], new_m[k]) >= TAU for k in keys if gt_at(float(k))[2] > 1.0], dtype=float)
point_f = and_f.mean()

rng = np.random.default_rng(SEED)
n = len(and_f)
nblk = math.ceil(n / BLK)
boot = np.empty(NREP)
for i in range(NREP):
    idx = rng.integers(0, nblk, size=nblk)
    pool = np.concatenate([and_f[b * BLK:(b + 1) * BLK] for b in idx])
    boot[i] = pool.mean()
lo_f, hi_f = np.percentile(boot, [2.5, 97.5])

old_b = {r["img"]: float(r["max_conf"]) for r in csv.DictReader(open(ROOT / "research/coco_bg_fp_it69v1-old.csv"))}
new_b = {r["img"]: float(r["max_conf"]) for r in csv.DictReader(open(ROOT / "research/coco_bg_fp_it69v1-new.csv"))}
bk = sorted(set(old_b) & set(new_b))
fp_arr = np.array([min(old_b[k], new_b[k]) >= TAU for k in bk], dtype=float)
point_b = fp_arr.mean()
z = 1.96
den = 1 + z * z / len(bk)
ctr = (point_b + z * z / (2 * len(bk))) / den
half = z * math.sqrt(point_b * (1 - point_b) / len(bk) + z * z / (4 * len(bk) ** 2)) / den

print(f"полёт AND@{TAU}: n={n} кадров (блоки по {BLK}), точка {point_f:.1%}, "
      f"95 % [блочный бутстрап, {NREP} реплик, сид {SEED}] = [{lo_f:.1%}; {hi_f:.1%}]")
print(f"  → нижняя граница {'≥' if lo_f >= 0.895 else '<'} 89,5 % (порог G1): "
      f"{'запас устойчив к временной кластеризации пропусков' if lo_f >= 0.895 else 'ТОЧЕЧНЫЙ зелёный не выдерживает блочного интервала'}")
print(f"фоны AND@{TAU}: n={len(bk)} независимых кадров, точка {point_b:.1%}, "
      f"95 % [Уилсон] = [{ctr - half:.1%}; {ctr + half:.1%}]")
print(f"  → верхняя граница {'≤' if ctr + half <= 0.128 else '>'} 12,8 % (порог G2): "
      f"{'запас устойчив' if ctr + half <= 0.128 else 'ТОЧЕЧНЫЙ зелёный не выдерживает интервала'}")
assert abs(point_f - 0.947) < 0.002 and abs(point_b - 0.072) < 0.002, "точки разошлись с it74_vote_analysis.txt"
print("сами-проверка: точки совпали с вердиктным прогоном (94,7 % / 7,2 %)")

OUT = ROOT / "research/it74_vote_bootstrap.txt"
with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"it-74 дополнение — интервальные запасы AND@{TAU} (критерии не меняются)\n\n")
    f.write(f"полёт: n={n}, блок={BLK}, реплик={NREP}, сид={SEED}: {point_f:.4f} [{lo_f:.4f}; {hi_f:.4f}] vs порог 0,895\n")
    f.write(f"фоны: n={len(bk)}, Уилсон 95 %: {point_b:.4f} [{ctr - half:.4f}; {ctr + half:.4f}] vs порог 0,128\n")
print(f"\nTXT: {OUT}")
