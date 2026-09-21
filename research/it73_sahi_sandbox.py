#!/usr/bin/env python3
"""it-73 S1: SAHI-pv по 73 sandbox-кадрам (640/0,2, инференс-срез 640) — вход контура.

Формат выхода — как yolo_sandbox_frames.csv (imgsz="sahi640"), чтобы каркас it-71
грузил его без правок. n_det — суммарное число боксов (conf≥0,05) по всем тайлам.
Запуск: research/.venv/bin/python research/it73_sahi_sandbox.py \
  --weights <.pt> --out research/yolo_sandbox_frames_<old|new>_sahi640.csv
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "research/sandbox_frames/full"

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default=None, help="'cuda' / 'cpu'; по умолчанию — как решит ultralytics")
args = ap.parse_args()

GT = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(ROOT / "research/gt_sandbox_video.csv"))}

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


rows = []
for sec in sorted(GT):
    img = cv2.imread(str(FRAMES / f"f_{sec + 1:03d}.jpg"))
    H, W = img.shape[:2]
    n_det, best = 0, 0.0
    for y in axis(H):
        for x in axis(W):
            tile = img[y:y + min(SLICE, H - y), x:x + min(SLICE, W - x)]
            r = model.predict(tile, imgsz=640, conf=0.05, verbose=False, device=args.device)[0]
            if r.boxes is not None and len(r.boxes):
                n_det += len(r.boxes)
                best = max(best, float(r.boxes.conf.max()))
    rows.append(dict(imgsz="sahi640", second=sec, visible=GT[sec][0], airborne=GT[sec][1],
                     n_det=n_det, max_conf=round(best, 6)))

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

air = [r["max_conf"] for r in rows if r["airborne"]]
gnd = [r["max_conf"] for r in rows if not r["airborne"]]
import statistics as st  # noqa: E402
print(f"{out.name}: airborne n={len(air)} медиана {st.median(air):.3f} (мин {min(air):.3f}) | "
      f"ground n={len(gnd)} медиана {st.median(gnd):.3f} (макс {max(gnd):.3f})")
for thr in (0.25, 0.4, 0.5):
    rec = sum(1 for c in air if c >= thr) / len(air)
    fp = sum(1 for c in gnd if c >= thr)
    print(f"  thr={thr}: recall airborne={rec:.1%}  ground-секунд ≥thr: {fp}")
print(f"CSV: {out} ({len(rows)} строк)")
