#!/usr/bin/env python3
"""it-70 поисковое плечо (просьба автора 2026-09-21): готовые модели с полки против той же линейки.

БЕЗ обучения: для каждого чекпойнта — mAP50 на DUT-test600 и HF-test600 (split=test, imgsz640,
CPU, чтобы не мешать тренировке bg70) + FP на 400 независимых фонах coco-bg-v1 (it-69 V1).
Калибровка линейки: первым идёт старый боевой вес (ожидаем ≈0,720 DUT / ≈0,867 HF / FP@0,5 27,5 %±).
CSV инкрементальный (partial-file протокол it-65): строка печатается сразу после модели.
Запуск: research/.venv/bin/python research/shelf_screen.py
"""
import csv
import sys
from pathlib import Path

import torch  # noqa: F401  (ultralytics сам разберётся; импорт для явности CPU-режима)

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
PREP = MD / "train/data/_prepared"
BG_DIR = PREP / "coco-bg-v1"
OUT = ROOT / "research/shelf_screen_results.csv"
MODELS = [
    ("old-uav-yolov8s", MD / "models/visual/yolov8s-uav.pt", "боевой майский (калибровка линейки)"),
    ("shelf-doguilmak-v11x", ROOT / "research/shelf_models/doguilmak_v11x.pt", "MIT, HF doguilmak/Drone-Detection-YOLOv11x"),
    ("shelf-doguilmak-v8x", ROOT / "research/shelf_models/doguilmak_v8x.pt", "HF doguilmak/Drone-Detection-YOLOv8x"),
    ("shelf-skyguard-v11", ROOT / "research/shelf_models/skyguard_v11.pt", "MIT, HF SkyGuardAI/drone-detection-yolov11"),
    ("shelf-ruju-v12", ROOT / "research/shelf_models/ruju_v12.pt", "AGPL-3.0(!), HF rujutashashikanjoshi/yolo12-..."),
    ("shelf-danivelikova-v26n", ROOT / "research/shelf_models/danivelikova_v26n.pt", "HF danivelikova/...-yolo26n"),
    ("shelf-iris-v8s", ROOT / "research/shelf_models/iris_v8s.pt", "CC-BY-NC(!), IRIS EO benchmark baseline"),
    ("shelf-noah-v8s", ROOT / "research/shelf_models/noah_v8s.pt", "HF Noah-Yohannes/Drone_Detection"),
]
FIELDS = ["model", "dut600_map50", "hf600_map50", "fp@0.25", "fp@0.4", "fp@0.5", "med_maxconf_bg", "note"]


def bg_fp(model) -> tuple[float, float, float, float]:
    imgs = sorted(BG_DIR.glob("*.jpg"))
    assert len(imgs) == 400, f"ожидал 400 независимых фонов, нашёл {len(imgs)}"
    maxes = []
    for i, img in enumerate(imgs, 1):
        r = model.predict(str(img), imgsz=640, conf=0.01, verbose=False, device="cpu")[0]
        confs = r.boxes.conf.cpu().tolist() if r.boxes is not None else []
        maxes.append(max(confs) if confs else 0.0)
        if i % 100 == 0:
            print(f"  bg {i}/400", flush=True)
    n = len(maxes)
    med = sorted(maxes)[n // 2]
    return (sum(c >= 0.25 for c in maxes) / n, sum(c >= 0.4 for c in maxes) / n,
            sum(c >= 0.5 for c in maxes) / n, med)


def val_map(model, data: Path) -> float:
    res = model.val(data=str(data), imgsz=640, split="test", device="cpu", batch=1,
                    workers=0, plots=False, verbose=False)
    return float(res.box.map50)


def main() -> None:
    new_file = not OUT.exists()
    from ultralytics import YOLO

    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for label, path, note in MODELS:
            row = dict(model=label, note=note)
            try:
                model = YOLO(str(path))
                row["dut600_map50"] = f"{val_map(model, PREP / 'visual-dut-test600/data.yaml'):.4f}"
                print(f"{label}: dut600={row['dut600_map50']}", flush=True)
                row["hf600_map50"] = f"{val_map(model, PREP / 'visual-hf-test600/data.yaml'):.4f}"
                print(f"{label}: hf600={row['hf600_map50']}", flush=True)
                fp25, fp40, fp50, med = bg_fp(model)
                row.update({"fp@0.25": f"{fp25:.3f}", "fp@0.4": f"{fp40:.3f}",
                            "fp@0.5": f"{fp50:.3f}", "med_maxconf_bg": f"{med:.3f}"})
                print(f"{label}: bg FP {fp25:.1%}/{fp40:.1%}/{fp50:.1%} med {med:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001 — скрининг: падающую модель не роняем всю цепочку
                row["note"] = f"{note} | FAIL {type(e).__name__}: {str(e)[:120]}"
                print(f"{label}: FAIL {e}", flush=True)
            w.writerow(row)
            f.flush()
    print("SHELF SCREEN DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
