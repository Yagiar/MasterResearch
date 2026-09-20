#!/usr/bin/env python3
"""Статистика визуального корпуса it-65 по доменам и сплитам (для отчёта).

Считает по YOLO-разметке (_prepared/visual): кадры, боксы, долю негативных кадров,
распределение относительных размеров объектов (small <32^2 px эквивалент по площади
доли от кадра). Дешёвый проход по txt-файлам, CPU не нагружает тренировку.
Запуск: research/.venv/bin/python research/corpus_stats.py
"""
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "MasterDiploma/train/data/_prepared/visual"

def domain(stem: str) -> str:
    if stem.startswith("dut-anti-uav"):
        return "DUT"
    if stem.startswith("coco-background"):
        return "COCO-bg"
    return "HF"

stats = defaultdict(lambda: {"frames": 0, "boxes": 0, "neg": 0, "w_rel": []})
for split in ("train", "val", "test"):
    labs = (VIS / "labels" / split)
    for lf in sorted(labs.glob("*.txt")):
        d = domain(lf.stem)
        s = stats[(split, d)]
        s["frames"] += 1
        rows = [ln.split() for ln in lf.read_text().splitlines() if ln.strip()]
        if not rows:
            s["neg"] += 1
        for r in rows:
            s["boxes"] += 1
            if len(r) >= 4:
                s["w_rel"].append(float(r[2]) * float(r[3]))  # rel площадь бокса

print(f"{'сплит':6} {'домен':9} {'кадров':>7} {'боксам':>7} {'негат.':>7} {'%нег':>6} {'площадь: медиана':>17} {'p10':>7} {'p90':>7}")
tot = defaultdict(lambda: [0, 0, 0])
for (split, d), s in sorted(stats.items()):
    ws = sorted(s["w_rel"]) or [0.0]
    p10, med, p90 = ws[len(ws) // 10], statistics.median(ws), ws[int(len(ws) * 0.9)]
    pct = s["neg"] / s["frames"] if s["frames"] else 0
    print(f"{split:6} {d:9} {s['frames']:>7} {s['boxes']:>7} {s['neg']:>7} {pct:>5.1%} "
          f"{med:>15.5f} {p10:>6.5f} {p90:>6.5f}")
    t = tot[split]
    t[0] += s["frames"]; t[1] += s["boxes"]; t[2] += s["neg"]
print()
for split, (f, b, n) in sorted(tot.items()):
    print(f"{split}: кадров {f}, боксов {b}, негативов {n} ({n / f:.1%})")
