#!/usr/bin/env python3
"""Бэклог it65-union-new-weights: чистый conf_union для НОВЫХ весов + разбор по дистанции.

full@1920 новыми весами (`mmaud_imgsz1920_new.csv`, GPU-прогон 21.09) мержится с уже
имеющимся SAHI-прогоном new (`mmaud_sahi_full_new.csv`) по кадрам; считается recall
«летит»/дальних 10–19 м при conf≥0,5 для SAHI, full@1920 и union=max — против старых
весов (канон E5 94,5 %, дальн 89,7 %, union-дальн 91,5 %). Справка для отчёта; вердикты
E1–E8 предрегистрированы по SAHI-протоколу и этим замером не меняются.

Запуск (после окончания GPU-прогона): research/.venv/bin/python research/it65_union_analysis.py
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


def load_sahi(path: Path) -> dict[str, float]:
    return {r["img"]: float(r["conf_sahi"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def load_full(path: Path) -> dict[str, float]:
    return {r["img"]: float(r["conf"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def report(tag: str, sahi: dict[str, float], full: dict[str, float]) -> None:
    keys = sorted(set(sahi) & set(full))
    seg: dict[str, list[float]] = {"полёт": [], "дальн10-19": [], "ближн0-9": []}
    for k in keys:
        x, y, z = gt_at(float(k))
        if z <= 1.0:
            continue
        d = math.hypot(x, y)
        u = max(sahi[k], full[k])
        for name, cond in (("полёт", True), ("дальн10-19", 10 <= d < 20), ("ближн0-9", d < 10)):
            if cond:
                seg[name].append((sahi[k], full[k], u))
    print(f"\n{tag}: кадров пересечения {len(keys)}")
    for name in ("полёт", "ближн0-9", "дальн10-19"):
        rows = seg[name]
        n = max(1, len(rows))
        rs = sum(r[0] >= 0.5 for r in rows) / n
        rf = sum(r[1] >= 0.5 for r in rows) / n
        ru = sum(r[2] >= 0.5 for r in rows) / n
        print(f"  {name:>10} (n={len(rows)}): SAHI={rs:.1%}  full@1920={rf:.1%}  union={ru:.1%}")


new_sahi = load_sahi(ROOT / "research/mmaud_sahi_full_new.csv")
new_full = load_full(ROOT / "research/mmaud_imgsz1920_new.csv")
report("NEW (uav-yolov8s-bg)", new_sahi, new_full)

old = {r["img"]: (float(r["conf_sahi"]), float(r["conf_full1920"]))
       for r in csv.DictReader(open(ROOT / "research/mmaud_sahi_full.csv", encoding="utf-8"))}
report("OLD (боевой, контроль)", {k: v[0] for k, v in old.items()}, {k: v[1] for k, v in old.items()})
