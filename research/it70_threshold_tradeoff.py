#!/usr/bin/env python3
"""Диагностика it-70 (без инференса): можно ли спасти дальний бакет одним порогом.

Для каждого веса по общей сетке T считаем пару (FP-доля на 400 фонaх coco-bg-v1 @T;
recall дальних 10–19 м на MMAUD @T) и ищем окно: дальн ≥ 80,4 % (под-порог it-70)
при FP ≤ 12,8 % (критерий E1@0,5). Это НЕ пересмотр вердикта E1–E8 — диагностика
для отчёта и для решения, что делать после перезапуска плеча T.

Запуск: research/.venv/bin/python research/it70_threshold_tradeoff.py
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


def gt_at(t: float) -> np.ndarray:
    return gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))]


def far_confs(sahi_csv: str, band: tuple[float, float] | None) -> list[float]:
    """conf_sahi на летящих кадрах MMAUD; band=None — все «летит» (E5), иначе дистанционное окно."""
    out = []
    for r in csv.DictReader(open(ROOT / sahi_csv, encoding="utf-8")):
        x, y, z = gt_at(float(r["img"]))
        if z <= 1.0:
            continue
        d = math.hypot(x, y)
        if band is None or band[0] <= d < band[1]:
            out.append(float(r["conf_sahi"]))
    return out


def bg_maxconf(fp_csv: str) -> list[float]:
    return [float(r["max_conf"]) for r in csv.DictReader(open(ROOT / fp_csv, encoding="utf-8"))]


MODELS = [
    ("old (боевой)", "research/mmaud_sahi_full.csv", "research/coco_bg_fp_it69v1-old.csv"),
    ("new (it-65)", "research/mmaud_sahi_full_new.csv", "research/coco_bg_fp_it69v1-new.csv"),
    ("skyguard-v11", "research/mmaud_sahi_full_skyguard-v11.csv", "research/coco_bg_fp_skyguard-v11.csv"),
    ("doguilmak-v8x", "research/mmaud_sahi_full_doguilmak-v8x.csv", "research/coco_bg_fp_doguilmak-v8x.csv"),
]
GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print(f"{'модель':>14} {'N дальн':>7} | " + " ".join(f"T={t:.2f}" for t in GRID))
for tag, sahi_csv, fp_csv in MODELS:
    far = far_confs(sahi_csv, (10.0, 20.0))
    fly = far_confs(sahi_csv, None)
    bg = bg_maxconf(fp_csv)
    rec = " ".join(f"{sum(c >= t for c in far) / max(1, len(far)):5.1%}" for t in GRID)
    flyr = " ".join(f"{sum(c >= t for c in fly) / max(1, len(fly)):5.1%}" for t in GRID)
    fp = " ".join(f"{sum(c >= t for c in bg) / max(1, len(bg)):5.1%}" for t in GRID)
    print(f"{tag:>14} {len(far):>7} | полёт(E5):  " + flyr)
    print(f"{'':>14} {'':>7} | дальн:    " + rec)
    print(f"{'':>14} {len(bg):>7} | FP:       " + fp)
    ok = [t for t in GRID
          if sum(c >= t for c in fly) / max(1, len(fly)) >= 0.895
          and sum(c >= t for c in far) / max(1, len(far)) >= 0.804
          and sum(c >= t for c in bg) / max(1, len(bg)) <= 0.128]
    print(f"{'':>14} {'':>7} | окно (E5≥89,5 % и дальн≥80,4 % и FP≤12,8 %): {ok or 'НЕТ на сетке'}")
