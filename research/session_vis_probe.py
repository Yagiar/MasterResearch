#!/usr/bin/env python3
"""Контроль «стоящий дрон» на негативной сессии (it-65, шаг 3).

Семантика (it-51): негативная сессия — наземные сегменты sandbox-клипа, дрон СТОИТ
в кадре, моторы выключены: airborne=0, но presence (drone_visible) = 1 все 17 с.
Для видео-ветки это не FP-тест, а TP-контроль: модель, переученная с фонами и DUT,
не должна потерять маленького наземного дрона. Метрика — доля секунд, где есть
детект с conf >= порога (max по 5 кадрам/с).

Запуск: research/.venv/bin/python research/session_vis_probe.py --weights <файл.pt> --name <метка>
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.25, 0.4, 0.5, 0.7]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None, help="метка для CSV (по умолчанию — имя весов)")
    ap.add_argument("--video", default=str(ROOT / "MasterDiploma/sandboxDataForSimulator/negative-session.mp4"))
    ap.add_argument("--fps", type=float, default=5.0, help="частота забора кадров")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="cpu по умолчанию — не конкурировать с тренировкой")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"не открыл видео: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(src_fps / args.fps))

    from ultralytics import YOLO

    model = YOLO(args.weights)
    label = args.name or Path(args.weights).stem

    per_sec: dict[int, list[float]] = defaultdict(list)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            res = model.predict(frame, imgsz=args.imgsz, conf=0.01, verbose=False, device=args.device)[0]
            confs = res.boxes.conf.cpu().tolist() if res.boxes is not None else []
            per_sec[int(idx // src_fps)].append(max(confs) if confs else 0.0)
        idx += 1
    cap.release()

    out = ROOT / f"research/session_vis_probe_{label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["second", "n_frames", "max_conf"] + [f"det@{t}" for t in THRESHOLDS])
        for s in sorted(per_sec):
            mx = max(per_sec[s])
            w.writerow([s, len(per_sec[s]), round(mx, 4)] + [int(mx >= t) for t in THRESHOLDS])

    n = len(per_sec)
    print(f"=== {label}: {idx} кадров видео, взято {sum(len(v) for v in per_sec.values())} "
          f"({n} сек), imgsz={args.imgsz}, device={args.device} ===")
    print(f"{'порог':>8} {'секунд с детектом (TP по presence)':>36}")
    for t in THRESHOLDS:
        k = sum(int(max(v) >= t) for v in per_sec.values())
        print(f"{t:>8.2f} {k:>12} / {n}  ({k / n:.1%})")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
