#!/usr/bin/env python3
"""Re-test признака движения (speed_rel) в SAHI-режиме на MMAUD (it-60).

Гипотеза (it-59 §5): при SAHI-кропе цель крупнее в пикселях среза → джиттер детектора
относительно ширины bbox меньше → скорость центра (нормированная на bbox) разделяет
«стоит (z<0.5) / летит (z>1)» — в отличие от it-52/53 (полный кадр).

Метод: SAHI-детекция (640/0.2, best drone conf) на кадрах; speed_rel = |Δцентр|/Δt / bbox_w;
два набора: (а) кадры 0..300 ПОДРЯД (нативный dt≈33 мс: 49 статичных + начало полёта);
(б) полёт каждые 30 кадров (dt≈1 с, дальняя фаза). Распределения + ROC-разделимость.
Запуск: research/.venv/bin/python research/mmaud_sahi_motion.py
"""
import csv
import glob
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
IMGS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/image/*.png")))
GTS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in GTS]
gt_z = [float(np.load(g)[2]) for g in GTS]

def z_of(p):
    ts = float(Path(p).stem)
    return gt_z[min(len(gt_z) - 1, bisect_left(gt_ts, ts))]

from ultralytics import YOLO  # noqa: E402
import cv2  # noqa: E402

model = YOLO(str(MD / "models/visual/yolov8s-uav.pt"))

def _nms(dets, iou_thr=0.45):
    """NMS-мердж (как в detector._nms_merge): устраняет кросс-срезовые дубли с обрезанными боксами."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[0]):
        c, (cx, cy, bw) = d
        dup = False
        for kc, (kcx, kcy, kbw) in keep:
            ix = max(0, min(cx + bw / 2, kcx + kbw / 2) - max(cx - bw / 2, kcx - kbw / 2))
            iy = max(0, min(cy + bw / 2, kcy + kbw / 2) - max(cy - bw / 2, kcy - kbw / 2))
            inter = ix * iy
            union = bw * bw + kbw * kbw - inter
            if union > 0 and inter / union >= iou_thr:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def sahi_best(img, slice_px=640, overlap=0.2):
    """Лучший drone-conf и его центр/bbox: SAHI-нарезка + NMS-мердж (как в detector).

    Без мерджа argmax прыгал между срезами из-за обрезанных на границах боксов —
    скорость насыщалась (артефакт методологии первой попытки it-60).
    """
    H, W = img.shape[:2]
    step = max(1, int(slice_px * (1 - overlap)))
    dets = []
    ys = list(range(0, max(1, H - slice_px + 1), step)) or [0]
    if ys[-1] != max(0, H - slice_px):
        ys.append(max(0, H - slice_px))
    xs = list(range(0, max(1, W - slice_px + 1), step)) or [0]
    if xs[-1] != max(0, W - slice_px):
        xs.append(max(0, W - slice_px))
    for y in ys:
        for x in xs:
            sh, sw = min(slice_px, H - y), min(slice_px, W - x)
            tile = img[y:y + sh, x:x + sw]
            r = model.predict(tile, imgsz=640, conf=0.05, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            k = int(r.boxes.conf.argmax())
            x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[k]]
            dets.append((float(r.boxes.conf[k]),
                         (x + (x1 + x2) / 2, y + (y1 + y2) / 2, x2 - x1)))
    if not dets:
        return 0.0, None
    merged = _nms(dets)
    c, box = max(merged, key=lambda d: d[0])
    return c, box

def speed_rel_series(frame_range):
    """speed_rel по последовательности кадров (consecutive по dt реального времени)."""
    out = []
    prev = None  # (ts, cx, cy, bw)
    for idx in frame_range:
        p = IMGS[idx]
        ts = float(Path(p).stem)
        conf, box = sahi_best(cv2.imread(p))
        if box is None:
            prev = None
            continue
        cx, cy, bw = box
        if prev is not None:
            dt = ts - prev[0]
            if dt > 1e-3:
                speed = ((cx - prev[1]) ** 2 + (cy - prev[2]) ** 2) ** 0.5 / dt
                out.append(dict(idx=idx, dt=round(dt, 3),
                                speed_rel=round(min(1.0, speed / max(1.0, min(prev[3], bw))), 4),
                                z=z_of(p), conf=round(conf, 3)))
        prev = (ts, cx, cy, bw)
    return out

rows_a = speed_rel_series(range(0, 300))                    # подряд: статика + ранний полёт
flight_idx = list(range(300, len(IMGS), 30))                # дальняя фаза, dt≈1 с
rows_b = speed_rel_series(flight_idx)
rows = rows_a + rows_b

with open(ROOT / "research/mmaud_sahi_motion.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "dt", "speed_rel", "z", "conf"])
    w.writeheader(); w.writerows(rows)

static = [r["speed_rel"] for r in rows if r["z"] < 0.5]
flight = [r["speed_rel"] for r in rows if r["z"] > 1.0]
def med(v):
    v = sorted(v); return v[len(v) // 2] if v else float("nan")
def p90(v):
    v = sorted(v); return v[int(0.9 * (len(v) - 1))] if v else float("nan")

print(f"замеров: {len(rows)} (стоит {len(static)}, летит {len(flight)})")
for name, seg in (("СТОИТ", static), ("ЛЕТИТ", flight)):
    if seg:
        print(f"  {name}: speed_rel медиана={med(seg):.3f} p90={p90(seg):.3f} max={max(seg):.3f}")
print("\nразделимость по motion_floor (доля «активен»):")
for floor in (0.02, 0.05, 0.1, 0.2, 0.4):
    fp = sum(1 for v in static if v >= floor) / max(1, len(static))
    tp = sum(1 for v in flight if v >= floor) / max(1, len(flight))
    print(f"  floor={floor:.2f}: статика проходит {fp:.1%} (FP), полёт проходит {tp:.1%} (recall)")
print(f"\nCSV: {ROOT / 'research/mmaud_sahi_motion.csv'}")
