#!/usr/bin/env python3
"""Полный прогон MMAUD Mavic3 @imgsz 1920 + SAHI-проба (it-58).

It-56: recall «летит» 18.3% @960 → 64.5% @1920 (подвыборка 1/10). Здесь:
  1) полный прогон всех 5091 кадров @1920 — точная кривая recall по состояниям и времени;
  2) SAHI-проба (нарезка 640 с перекрытием) на подвыборке «полёт» — потолок восстановления;
  3) кривая p_v по секундам (где модель видит/теряет).
Запуск: research/.venv/bin/python research/mmaud_imgsz1920_eval.py
  [--weights ПТЬ.ВЕСОВ] [--out ПТЬ.CSV]   # дефолты: канон it-58 (yolov8s-uav.pt, mmaud_imgsz1920.csv);
  замеры новых весов (union-бэклог it-65) — явным --out, чтобы не затирать baseline.
"""
import argparse
import csv
import glob
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(MD / "models/visual/yolov8s-uav.pt"))
ap.add_argument("--out", default=None, help="путь CSV (по умолчанию research/mmaud_imgsz1920.csv)")
args = ap.parse_args()
IMGS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/image/*.png")))
GTS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in GTS]
gt_z = [float(np.load(g)[2]) for g in GTS]

def z_of(p):
    ts = float(Path(p).stem)
    return gt_z[min(len(gt_z) - 1, bisect_left(gt_ts, ts))]

from ultralytics import YOLO  # noqa: E402

model = YOLO(args.weights)

OUT = Path(args.out) if args.out else ROOT / "research/mmaud_imgsz1920.csv"
rows = []
for i, img_path in enumerate(IMGS):
    res = model.predict(img_path, imgsz=1920, conf=0.01, verbose=False)[0]
    conf = float(res.boxes.conf.max()) if res.boxes is not None and len(res.boxes) else 0.0
    rows.append(dict(img=Path(img_path).stem, z=round(z_of(img_path), 3), conf=round(conf, 4)))
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(IMGS)}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["img", "z", "conf"])
    w.writeheader(); w.writerows(rows)

ground = [r for r in rows if r["z"] < 0.5]
flight = [r for r in rows if r["z"] > 1.0]
print(f"\n=== полный прогон @1920: кадров {len(rows)} (стоит {len(ground)}, летит {len(flight)}) ===")
for name, seg in (("стоит", ground), ("летит", flight)):
    rec = sum(1 for r in seg if r["conf"] >= 0.5)
    c = sorted(r["conf"] for r in seg)
    print(f"  {name}: recall(≥0.5)={rec / max(1, len(seg)):.1%}  conf медиана={c[len(c) // 2]:.3f}  p90={c[int(0.9 * (len(c) - 1))]:.3f}")
# recall по третям полёта (начало/середина/конец)
fl = sorted(flight, key=lambda r: r["img"])
for j, part in enumerate((fl[:len(fl)//3], fl[len(fl)//3:2*len(fl)//3], fl[2*len(fl)//3:]), 1):
    rec = sum(1 for r in part if r["conf"] >= 0.5)
    print(f"  полёт треть {j}: recall={rec / max(1, len(part)):.1%} (n={len(part)})")

# --- SAHI-проба: нарезка 640/0.2 на подвыборке полёта, где @1920 не увидел (conf<0.5) ---
print("\n=== SAHI-проба (slices 640, overlap 0.2) на 60 кадрах, где @1920 conf<0.5 ===")
missed = [r for r in flight if r["conf"] < 0.5][::max(1, len([r for r in flight if r["conf"] < 0.5]) // 60)]
import cv2  # noqa: E402

recovered = 0
for r in missed:
    img = cv2.imread(str(MD / "train/data/mmaud/Mavic3/image" / f"{r['img']}.png"))
    H, W = img.shape[:2]
    sl, ov = 640, int(640 * 0.2)
    best = 0.0
    for y in range(0, H, sl - ov):
        for x in range(0, W, sl - ov):
            tile = img[y:y + sl, x:x + sl]
            if tile.size == 0:
                continue
            res = model.predict(tile, imgsz=640, conf=0.01, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                best = max(best, float(res.boxes.conf.max()))
    recovered += (best >= 0.5)
print(f"  восстановлено SAHI: {recovered}/{len(missed)} = {recovered / max(1, len(missed)):.1%} кадров с conf ≥ 0.5")
print(f"\nCSV: {OUT}")
