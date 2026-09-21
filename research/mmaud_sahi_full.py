#!/usr/bin/env python3
"""Полный SAHI-прогон всего корпуса MMAUD (5091 кадров) — it-63.

Завершает it-58: SAHI-проба была на 62 пропущенных кадрах; здесь SAHI (640/0.2)
на ВСЕХ кадрах + мерж с уже измеренным full@1920 (mmaud_imgsz1920.csv) → точные
recall/union на независимом материале.
Запуск: research/.venv/bin/python research/mmaud_sahi_full.py

ОСТОРОЖНО (it-65): mmaud_imgsz1920.csv измерен СТАРЫМИ весами (it-58). При других --weights
колонки conf_full1920/conf_union в выходе — смесь моделей, для old↔new сравнений непригодны
(сравнивать только conf_sahi). Для прогона другими весами передавать `--full1920-csv none`
(или файл full@1920, измеренный теми же весами).
"""
import argparse
import csv
import bisect
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
IMGS = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/image/*.png")))

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(MD / "models/visual/yolov8s-uav.pt"))
ap.add_argument("--out", default=str(ROOT / "research/mmaud_sahi_full.csv"))
ap.add_argument("--full1920-csv", default=str(ROOT / "research/mmaud_imgsz1920.csv"),
                help="предыдущий замер full@1920 ТЕМИ ЖЕ весами; 'none' — отключить мерж (честный conf_union)")
ap.add_argument("--device", default=None, help="устройство инференса ('cpu' для фоновых прогонов без гонки за GPU; по умолчанию — как решит ultralytics)")
args = ap.parse_args()

# уже измеренный full@1920 (it-58) — только если он относится к тем же весам
full1920 = ({} if args.full1920_csv == "none" else
            {r["img"]: float(r["conf"]) for r in csv.DictReader(open(args.full1920_csv, encoding="utf-8"))})

from ultralytics import YOLO  # noqa: E402
import cv2  # noqa: E402

model = YOLO(args.weights)

OUT = Path(args.out)
slice_px, overlap = 640, 0.2
step = max(1, int(slice_px * (1 - overlap)))

def sahi_conf(img):
    H, W = img.shape[:2]
    ys = list(range(0, max(1, H - slice_px + 1), step)) or [0]
    if ys[-1] != max(0, H - slice_px):
        ys.append(max(0, H - slice_px))
    xs = list(range(0, max(1, W - slice_px + 1), step)) or [0]
    if xs[-1] != max(0, W - slice_px):
        xs.append(max(0, W - slice_px))
    best = 0.0
    for y in ys:
        for x in xs:
            tile = img[y:y + min(slice_px, H - y), x:x + min(slice_px, W - x)]
            r = model.predict(tile, imgsz=640, conf=0.05, verbose=False, device=args.device)[0]
            if r.boxes is not None and len(r.boxes):
                best = max(best, float(r.boxes.conf.max()))
    return best

rows = []
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["img", "z", "conf_full1920", "conf_sahi", "conf_union"])
    w.writeheader()
    for i, p in enumerate(IMGS):
        stem = Path(p).stem
        img = cv2.imread(p)
        cs = sahi_conf(img)
        cf = full1920.get(stem, 0.0)
        union = max(cs, cf)
        w.writerow(dict(img=stem, conf_full1920=round(cf, 4), conf_sahi=round(cs, 4), conf_union=round(union, 4)))
        rows.append(dict(img=stem, cf=cf, cs=cs, union=union))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(IMGS)}", flush=True)

def recall(rs, key):
    return sum(1 for r in rs if r[key] >= 0.5) / max(1, len(rs))

gt_files = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt_z = [float(np.load(g)[2]) for g in gt_files]


ground = [r for r in rows if gt_z[min(len(gt_z) - 1, bisect.bisect_left(gt_ts, float(r["img"])))] < 0.5]
flight = [r for r in rows if gt_z[min(len(gt_z) - 1, bisect.bisect_left(gt_ts, float(r["img"])))] > 1.0]

print(f"\n=== ПОЛНЫЙ корпус {len(rows)} кадров (стоит {len(ground)}, летит {len(flight)}) ===")
for name, seg in (("стоит", ground), ("летит", flight)):
    print(f"  {name}: full@1920={recall(seg, 'cf'):.1%}  SAHI={recall(seg, 'cs'):.1%}  union={recall(seg, 'union'):.1%}")
print(f"\nCSV: {OUT} ({len(rows)} строк)")
