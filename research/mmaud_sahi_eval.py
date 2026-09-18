#!/usr/bin/env python3
"""Офлайн-оценка SAHI-детекции на выборке MMAUD (it-59): baseline @1920 vs SAHI 640/0.2.

Каждый k-й кадр корпуса: (а) полный кадр @1920; (б) нарезка 640/0.2 @640 (10 срезов на кадр).
Итог: recall присутствия (GT: дрон присутствует во всех кадрах) и сравнение с it-56/58.
Запуск: research/.venv/bin/python research/mmaud_sahi_eval.py [stride]
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
IMGS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/image/*.png")))
stride = int(sys.argv[1]) if len(sys.argv) > 1 else 10
sample = IMGS[::stride]
print(f"выборка: {len(sample)} из {len(IMGS)} кадров (stride={stride})")

from ultralytics import YOLO  # noqa: E402
import cv2  # noqa: E402

model = YOLO(str(MD / "models/visual/yolov8s-uav.pt"))

def sahi_detect(img, slice_px=640, overlap=0.2):
    H, W = img.shape[:2]
    step = max(1, int(slice_px * (1 - overlap)))
    best = 0.0
    for y in range(0, max(1, H - slice_px + 1) or 1, step):
        for x in range(0, max(1, W - slice_px + 1) or 1, step):
            sw, sh = min(slice_px, W - x), min(slice_px, H - y)
            tile = img[y:y + sh, x:x + sw]
            r = model.predict(tile, imgsz=640, conf=0.01, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                best = max(best, float(r.boxes.conf.max()))
            if y + sh >= H and x + sw >= W:
                break
        if y + slice_px >= H:
            break
    return best

rows = []
for i, p in enumerate(sample):
    img = cv2.imread(p)
    r_full = model.predict(img, imgsz=1920, conf=0.01, verbose=False)[0]
    conf_full = float(r_full.boxes.conf.max()) if r_full.boxes is not None and len(r_full.boxes) else 0.0
    conf_sahi = sahi_detect(img)
    rows.append(dict(img=Path(p).stem, conf_full1920=round(conf_full, 4), conf_sahi=round(conf_sahi, 4)))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(sample)}", flush=True)

with open(ROOT / "research/mmaud_sahi_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["img", "conf_full1920", "conf_sahi"])
    w.writeheader(); w.writerows(rows)

def recall(rs, key):
    return sum(1 for r in rs if r[key] >= 0.5) / max(1, len(rs))
def union(rs):
    return sum(1 for r in rs if r["conf_full1920"] >= 0.5 or r["conf_sahi"] >= 0.5) / max(1, len(rs))

print(f"\nrecall присутствия (n={len(rows)}):")
print(f"  full @1920:      {recall(rows, 'conf_full1920'):.1%}")
print(f"  SAHI 640/0.2 @640: {recall(rows, 'conf_sahi'):.1%}")
print(f"  union (либо-либо): {union(rows):.1%}")
