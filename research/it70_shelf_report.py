#!/usr/bin/env python3
"""Сводка плеча S it-70: полка HF-моделей против наших весов (скрининг + MMAUD-арбитраж).

Читает research/shelf_screen_results.csv и все появившиеся
research/mmaud_sahi_full_<shelf-label>.csv (SAHI 640/0,2, --full1920-csv none),
печатает единую таблицу с линейкой E5-recall полёта на независимом MMAUD.
Запуск: research/.venv/bin/python research/it70_shelf_report.py
"""
import bisect
import csv
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"

gt_files = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt_xyz = [np.load(g) for g in gt_files]
gt_z = [a[2] for a in gt_xyz]


def _gt_idx(t: float) -> int:
    return min(len(gt_z) - 1, bisect.bisect_left(gt_ts, t))


def flight_recall(path: Path) -> tuple[float | None, float | None]:
    """(полный E5-recall полёта, recall дальнего бакета 10–19 м) по conf_sahi ≥ 0,5."""
    if not path.exists():
        return None, None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    hits = far_tp = far_n = 0
    flight_n = 0
    for r in rows:
        i = _gt_idx(float(r["img"]))
        if gt_z[i] <= 1.0:
            continue
        flight_n += 1
        hit = float(r["conf_sahi"]) >= 0.5
        hits += hit
        if 10 <= math.hypot(*gt_xyz[i][:2]) < 20:
            far_tp += hit
            far_n += 1
    return (hits / flight_n if flight_n else None, (far_tp / far_n if far_n else None))


def main() -> None:
    ours = {
        "old-yolov8s-uav(боевой майский)": (MD / "train/runs/eval/visual-old-dut600", "visual-old-hf600",
                                            ROOT / "research/coco_bg_fp_it69v1-old.csv",
                                            ROOT / "research/mmaud_sahi_full.csv"),
        "new-yolov8s-bg(it-65)": (MD / "train/runs/eval/visual-new-dut600", "visual-new-hf600",
                                  ROOT / "research/coco_bg_fp_new-yolov8s-bg.csv",
                                  ROOT / "research/mmaud_sahi_full_new.csv"),
    }

    print(f"{'модель':<34} {'DUT600':>7} {'HF600':>7} {'FP@0,5':>7} {'SAHI-полёт':>10} {'дальн10-19':>10}  прим.")
    for name, (d_dir, h_dir, fpcsv, mmaud) in ours.items():
        d = _map50(d_dir)
        h = _map50(h_dir)
        fp = _fp(fpcsv)
        fr, far = flight_recall(mmaud)
        print(f"{name:<34} {_f(d):>7} {_f(h):>7} {_f(fp):>7} {_f(fr):>10} {_f(far):>10}  канон")
    screen = ROOT / "research/shelf_screen_results.csv"
    if screen.exists():
        for r in csv.DictReader(open(screen, encoding="utf-8")):
            m = r["model"]
            if m.startswith("old-"):
                continue
            label = m.removeprefix("shelf-")
            fr, far = flight_recall(ROOT / f"research/mmaud_sahi_full_{label}.csv")
            note = "нет арбитража" if fr is None else ""
            if fr is not None and fr < 0.895:
                note = "E5-красный" + ("; дальний бакет <80,4%" if far is not None and far < 0.804 else "")
            print(f"{m:<34} {r['dut600_map50']:>7} {r['hf600_map50']:>7} {r['fp@0.5']:>7} {_f(fr):>10} {_f(far):>10}  {note}")
    print("\nОговорки: (1) DUT600/HF600 для полки могут быть in-domain (вероятен тренировочный"
          " leakage на VisDrone-семействе) — арбитр только SAHI-полёт MMAUD (E5-линейка ≥0,895);"
          " (2) ruju-v12 — AGPL-3.0, iris-v8s — CC-BY-NC: внутреннее сравнение ок, поставка нет.")


def _map50(ev_dir) -> float | None:
    import json
    p = Path(ev_dir) / "metrics.json"
    if not p.exists():
        p = MD / "train/runs/eval" / str(ev_dir) / "metrics.json"
    return json.loads(p.read_text())["map50"] if p.exists() else None


def _fp(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    return sum(int(r["fp@0.5"]) for r in rows) / len(rows)


def _f(v):
    return "—" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
