#!/usr/bin/env python3
"""it-73 S2: FP SAHI-режима на 400 независимых фонах coco-bg-v1 (выборка V1 it-69).

Тот же tile-протокол, что it73_sahi_sandbox.py (640/0,2, инференс-срез 640, conf 0,05,
max по тайлам — дрон в кадре отсутствует по построению, любое срабатывание = FP).
Формат CSV — как coco_bg_fp_eval.py (img,n_boxes,max_conf,fp@…).
Запуск: research/.venv/bin/python research/it73_coco_sahi_fp.py \
  --weights <.pt> --out research/coco_bg_sahi_fp_it73_<old|new>.csv [--device cuda]
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.25, 0.4, 0.5, 0.7]

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--images-dir", default=str(ROOT / "MasterDiploma/train/data/_prepared/coco-bg-v1"))
ap.add_argument("--pattern", default="*.jpg")
ap.add_argument("--file-list", default=None,
                help="файл со списком имён (по одному в строке, relative to --images-dir); перекрывает --pattern")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

if args.file_list:
    imgs = [Path(args.images_dir) / l.strip() for l in open(args.file_list) if l.strip()]
else:
    imgs = sorted(Path(args.images_dir).glob(args.pattern))
if not imgs:
    raise SystemExit(f"нет кадров {args.pattern} в {args.images_dir}")

from ultralytics import YOLO  # noqa: E402
import cv2  # noqa: E402

model = YOLO(args.weights)
SLICE, OVERLAP = 640, 0.2
STEP = max(1, int(SLICE * (1 - OVERLAP)))


def axis(n_px: int) -> list[int]:
    pos = list(range(0, max(1, n_px - SLICE + 1), STEP)) or [0]
    last = max(0, n_px - SLICE)
    if pos[-1] != last:
        pos.append(last)
    return pos


def sahi_max(path: Path) -> tuple[int, float]:
    img = cv2.imread(str(path))
    H, W = img.shape[:2]
    n_det, best = 0, 0.0
    for y in axis(H):
        for x in axis(W):
            tile = img[y:y + min(SLICE, H - y), x:x + min(SLICE, W - x)]
            r = model.predict(tile, imgsz=640, conf=0.05, verbose=False, device=args.device)[0]
            if r.boxes is not None and len(r.boxes):
                n_det += len(r.boxes)
                best = max(best, float(r.boxes.conf.max()))
    return n_det, best

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
TMP = out.parent / (out.name + ".tmp")
n = 0
with open(TMP, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["img", "n_boxes", "max_conf"] + [f"fp@{t}" for t in THRESHOLDS])
    for p in imgs:
        n_det, best = sahi_max(p)
        w.writerow([p.name, n_det, round(best, 6)] + [int(best >= t) for t in THRESHOLDS])
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{len(imgs)}", flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
TMP.replace(out)

print(f"\nSAHI FP, n={n} фонов, веса {Path(args.weights).name}:")
rows = list(csv.DictReader(open(out)))
for t in THRESHOLDS:
    k = sum(int(r[f"fp@{t}"]) for r in rows)
    print(f"  FP@{t} = {k / n:.1%} ({k}/{n})")
print(f"CSV: {out}")
