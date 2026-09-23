#!/usr/bin/env python3
"""it-88 harness: per-tile дампы SAHI (640/0,2) + full-frame corroboration — GPU.

Тот же tile-протокол, что it73_coco_sahi_fp.py / mmaud_sahi_full.py (SLICE=640,
OVERLAP=0.2 → STEP=512, инференс-срез 640, conf=0.05), НО вместо одного max на кадр
сохраняет СПИСОК conf по тайлам (и count боксов на тайл) + отдельный full-frame@640 conf.
Это даёт нулевой-инференс анализатору it88_tile_aggregate.py пересобирать любые
агрегации (MAX / CONS-k / NULL-процентиль / full-frame corroboration) поверх дампа.

Две оси за один прогон (аргумент --source):
  bg400  : строгие 400 HD-фонов Open Images (strict400/*.jpg) — FP-ось (кадр без дрона);
  mmaud  : flight-кадры MMAUD через подмножество --stride (recall-guardrail, GT z>1.0).

Формат CSV: одна строка на (img), поля:
  img, n_tiles, tile_confs («;»-разделённый float-ряд по тайлам, порядок row-major),
  tile_nboxes («;»-разделённый int-ряд), full_conf640, w, h.
Запуск (GPU, строго ПОСЛЕ закрытия it-85):
  research/.venv/bin/python research/it88_sahi_tiled.py --source bg400 --device cuda \
      --weights MasterDiploma/models/visual/yolov8s-uav.pt  --out research/it88_bg400_old.csv
  research/.venv/bin/python research/it88_sahi_tiled.py --source bg400 --device cuda \
      --weights MasterDiploma/models/visual/uav-yolov8s-bg-best.pt --out research/it88_bg400_new.csv
  research/.venv/bin/python research/it88_sahi_tiled.py --source mmaud --stride 10 --device cuda \
      --weights MasterDiploma/models/visual/yolov8s-uav.pt  --out research/it88_mmaud_old.csv
  ... --weights MasterDiploma/models/visual/uav-yolov8s-bg-best.pt --out research/it88_mmaud_new.csv
"""
import argparse
import bisect
import csv
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
STRICT400 = MD / "train/data/_prepared/hi-res-bg-v1/strict400"
MMAUD_IMG = MD / "train/data/mmaud/Mavic3/image"
MMAUD_GT = MD / "train/data/mmaud/Mavic3/ground_truth"

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["bg400", "mmaud"])
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--stride", type=int, default=10, help="mmaud: каждый N-й flight-кадр (по sorted frame-id)")
args = ap.parse_args()

SLICE, OVERLAP = 640, 0.2
STEP = max(1, int(SLICE * (1 - OVERLAP)))


def axis(n_px: int) -> list[int]:
    pos = list(range(0, max(1, n_px - SLICE + 1), STEP)) or [0]
    last = max(0, n_px - SLICE)
    if pos[-1] != last:
        pos.append(last)
    return pos


def gt_z_at(t: float, gt_ts: list[float], gt: list[np.ndarray]) -> float:
    return float(gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))][2])


# ---- список кадров + размерности ----
if args.source == "bg400":
    imgs = sorted(STRICT400.glob("*.jpg"))
    assert len(imgs) == 400, f"strict400: {len(imgs)} кадров (ожидалось 400)"
    subset = imgs
else:
    gt_files = sorted(glob.glob(str(MMAUD_GT / "*.npy")))
    gt_ts = [float(Path(g).stem) for g in gt_files]
    gt = [np.load(g) for g in gt_files]
    all_frames = sorted(MMAUD_IMG.glob("*.png"), key=lambda p: float(p.stem))
    flight = [p for p in all_frames if gt_z_at(float(p.stem), gt_ts, gt) > 1.0]
    subset = flight[::max(1, args.stride)]
    print(f"mmaud: flight={len(flight)}, stride={args.stride} → subset={len(subset)}")

from ultralytics import YOLO  # noqa: E402
import cv2  # noqa: E402

model = YOLO(args.weights)


def tile_confs(img):
    H, W = img.shape[:2]
    confs, nboxes = [], []
    for y in axis(H):
        for x in axis(W):
            tile = img[y:y + min(SLICE, H - y), x:x + min(SLICE, W - x)]
            r = model.predict(tile, imgsz=SLICE, conf=0.05, verbose=False, device=args.device)[0]
            if r.boxes is not None and len(r.boxes):
                confs.append(float(r.boxes.conf.max()))
                nboxes.append(len(r.boxes))
            else:
                confs.append(0.0)
                nboxes.append(0)
    return confs, nboxes


def full_conf640(img):
    r = model.predict(img, imgsz=SLICE, conf=0.05, verbose=False, device=args.device)[0]
    return float(r.boxes.conf.max()) if (r.boxes is not None and len(r.boxes)) else 0.0


out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
TMP = out.parent / (out.name + ".tmp")
n = 0
with open(TMP, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["img", "n_tiles", "tile_confs", "tile_nboxes", "full_conf640", "w", "h"])
    for p in subset:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        confs, nboxes = tile_confs(img)
        w.writerow([p.name, len(confs),
                    ";".join(f"{c:.6f}" for c in confs),
                    ";".join(str(b) for b in nboxes),
                    round(full_conf640(img), 6), W, H])
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{len(subset)}", flush=True)
TMP.replace(out)
print(f"\n{args.source}: {n} кадров, веса {Path(args.weights).name} → {out}")
