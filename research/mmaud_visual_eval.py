#!/usr/bin/env python3
"""Независимая визуальная валидация на MMAUD V1 Mavic3 (it-56).

Первый прогон обученной модели (yolov8s-uav.pt) на НЕЗАВИСИМОМ материале (ревью §10):
5091 кадр 2560×960, ~30 fps, 169.8 с; GT — 3D-позиция дрона (833 замера, lidar).
Метрики:
  1) recall присутствия по состояниям (GT: z<0.5 м — стоит, z>1 м — летит);
  2) распределения p_v (стоит vs летит);
  3) re-test признака движения it-52/53 на реальном полёте: скорость центра,
     нормированная на ширину bbox (speed_rel), распределения по состояниям.
Запуск: research/.venv/bin/python research/mmaud_visual_eval.py
"""
import csv
import glob
import json
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
IMGS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/image/*.png")))
GTS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
OUT = ROOT / "research/mmaud_visual_eval.csv"

gt_ts = [float(Path(g).stem) for g in GTS]
gt_pos = [np.load(g) for g in GTS]
gt_z = [float(p[2]) for p in gt_pos]

from ultralytics import YOLO  # noqa: E402

model = YOLO(str(MD / "models/visual/yolov8s-uav.pt"))

rows = []
prev = None  # (ts, cx, cy, bbox_w) предыдущего кадра с детекцией
import cv2  # noqa: E402

for i, img_path in enumerate(IMGS):
    ts = float(Path(img_path).stem)
    res = model.predict(img_path, imgsz=960, conf=0.01, verbose=False)[0]
    j = min(len(gt_ts) - 1, bisect_left(gt_ts, ts))
    z = gt_z[j]
    conf, bw, cx, cy = 0.0, None, None, None
    if res.boxes is not None and len(res.boxes):
        k = int(res.boxes.conf.argmax())
        conf = float(res.boxes.conf[k])
        x1, y1, x2, y2 = [float(v) for v in res.boxes.xyxy[k]]
        bw = x2 - x1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    speed_rel = None
    if conf > 0 and prev is not None and prev[3]:
        dt = ts - prev[0]
        if dt > 1e-3:
            speed = ((cx - prev[1]) ** 2 + (cy - prev[2]) ** 2) ** 0.5 / dt
            speed_rel = min(1.0, speed / max(1.0, prev[3]))  # it-53: нормировка на bbox
    prev = (ts, cx, cy, bw) if conf > 0 else prev
    rows.append(dict(ts=round(ts, 2), z=round(z, 3), conf=round(conf, 4),
                     bbox_w=round(bw, 1) if bw else None,
                     speed_rel=round(speed_rel, 4) if speed_rel is not None else None))
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(IMGS)}", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ts", "z", "conf", "bbox_w", "speed_rel"])
    w.writeheader()
    w.writerows(rows)

ground = [r for r in rows if r["z"] < 0.5]
flight = [r for r in rows if r["z"] > 1.0]
def med(v):
    v = sorted(v); return v[len(v) // 2] if v else float("nan")
def p90(v):
    v = sorted(v); return v[int(0.9 * (len(v) - 1))] if v else float("nan")

print(f"\nкадров: {len(rows)} (стоит: {len(ground)}, летит: {len(flight)})")
print("recoll присутствия (conf ≥ 0.5):")
for name, seg in (("стоит", ground), ("летит", flight)):
    det = sum(1 for r in seg if r["conf"] >= 0.5)
    print(f"  {name}: {det}/{len(seg)} = {det / max(1, len(seg)):.1%}")
print("\np_v (макс conf) по состояниям:")
for name, seg in (("стоит", ground), ("летит", flight)):
    c = [r["conf"] for r in seg]
    print(f"  {name}: медиана={med(c):.3f} p90={p90(c):.3f}")
print("\nRE-TEST it-53: speed_rel (bbox-норм. скорость) по состояниям на реальном полёте:")
for name, seg in (("стоит", [r for r in ground if r["speed_rel"] is not None]),
                  ("летит", [r for r in flight if r["speed_rel"] is not None])):
    sr = [r["speed_rel"] for r in seg]
    print(f"  {name}: n={len(sr)} медиана={med(sr):.3f} p90={p90(sr):.3f} max={max(sr):.3f}")
for floor in (0.05, 0.1, 0.2, 0.3):
    fp = sum(1 for r in ground if (r["speed_rel"] or 0) >= floor) / max(1, len(ground))
    tp = sum(1 for r in flight if (r["speed_rel"] or 0) >= floor) / max(1, len(flight))
    print(f"  floor={floor:.2f}: статика проходит {fp:.1%}, полёт проходит {tp:.1%}")
print(f"\nCSV: {OUT}")
