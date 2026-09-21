#!/usr/bin/env python3
"""Recall полёта MMAUD по диапазонам дистанции (лидарным GT x,y) для имеющихся SAHI-CSV.

Микрорезультат it-70 (21.09): деградация E5 у новых весов точится на дальних 10–19 м.
Замер без нового инференса — только парсинг CSV + лидарный GT.
Запуск: research/.venv/bin/python research/mmaud_recall_by_distance.py [csv:key:tag ...]
"""
import bisect
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
gt_files = sorted(glob.glob(str(ROOT / "MasterDiploma/train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt = [np.load(g) for g in gt_files]


def gt_at(t: float) -> np.ndarray:
    i = min(len(gt) - 1, bisect.bisect_left(gt_ts, t))
    return gt[i]


def analyze(path: Path, key: str) -> dict[int, tuple[int, int]]:
    bins: dict[int, tuple[int, int]] = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        x, y, z = gt_at(float(r["img"]))
        if z <= 1.0:  # сегмент «летит» (z > 1 м), как в E5
            continue
        b = int(math.hypot(x, y) // 10) * 10
        tp, n = bins.get(b, (0, 0))
        bins[b] = (tp + (float(r[key]) >= 0.5), n + 1)
    return bins


def main() -> None:
    specs = sys.argv[1:] or [
        "research/mmaud_sahi_full.csv:conf_sahi:old",
        "research/mmaud_sahi_full_new.csv:conf_sahi:new",
        "research/mmaud_sahi_full.csv:conf_union:old+full1920",
    ]
    for sp in specs:
        rel, key, tag = sp.split(":")
        p = ROOT / rel
        if not p.exists():
            print(f"{tag:>18}: (нет файла {rel})")
            continue
        b = analyze(p, key)
        print(f"{tag:>18}: " + "  ".join(f"{k}–{k + 9}м:{v[0] / v[1]:.1%}({v[1]})" for k, v in sorted(b.items())))


if __name__ == "__main__":
    main()
