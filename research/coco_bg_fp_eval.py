#!/usr/bin/env python3
"""Замер FP видео-ветки на независимых фонах без дрона (it-65, шаг 3).

Кадры `coco-background` из test-сплита подготовленного корпуса (COCO val2017,
в кадре дрона нет по построению). Модель обязана молчать: любое детект-окно
выше порога — ложная тревога. Это прямой ответ на дефект it-64 (0,002% негативов
в обучении → «всегда-положительное» видео).

Запуск: research/.venv/bin/python research/coco_bg_fp_eval.py --weights <файл.pt> [--name метка]
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "MasterDiploma/train/data/_prepared/visual/images/test"
THRESHOLDS = [0.25, 0.4, 0.5, 0.7]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None, help="метка для CSV (по умолчанию — имя весов)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="cpu по умолчанию — не конкурировать с тренировкой")
    ap.add_argument("--images-dir", default=None,
                    help="каталог фонов (по умолчанию test-сплит подготовленного корпуса; it-69 V1 — свежая выборка val2017)")
    ap.add_argument("--pattern", default="coco-background_*.jpg")
    args = ap.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else IMAGES_DIR
    imgs = sorted(images_dir.glob(args.pattern))
    if not imgs:
        raise SystemExit(f"нет кадров {args.pattern} в {images_dir}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    label = args.name or Path(args.weights).stem

    rows = []
    for i, img in enumerate(imgs, 1):
        res = model.predict(str(img), imgsz=args.imgsz, conf=0.01, verbose=False, device=args.device)[0]
        confs = res.boxes.conf.cpu().tolist() if res.boxes is not None else []
        max_conf = max(confs) if confs else 0.0
        row = dict(img=img.name, n_boxes=len(confs), max_conf=round(max_conf, 4))
        for t in THRESHOLDS:
            row[f"fp@{t}"] = int(any(c >= t for c in confs))
        rows.append(row)
        if i % 20 == 0:
            print(f"{i}/{len(imgs)}", flush=True)

    out = ROOT / f"research/coco_bg_fp_{label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"\n=== {label}: {n} фоновых кадров COCO-test, imgsz={args.imgsz} ===")
    print(f"{'порог':>8} {'кадров с FP':>12} {'доля':>8}")
    for t in THRESHOLDS:
        k = sum(r[f"fp@{t}"] for r in rows)
        print(f"{t:>8.2f} {k:>12} {k / n:>7.1%}")
    print(f"медиана max_conf: {sorted(r['max_conf'] for r in rows)[n // 2]:.4f}")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
