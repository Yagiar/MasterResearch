#!/usr/bin/env python3
"""Сверка весов it-70 (uav-yolov8s-bg70) с предрегистрированными критериями E1–E8.

Критерии зафиксированы ДО метрик it-70 (iterations/it-70-…-PLANNED.md, коммит b43f782).
E1 — независимые 400 фонов COCO val2017 (тот же каталог, что V1 it-69); E2–E5 — образец it-65;
E6–E8 — контур уровня окон на песочных кадрах (урок V3 it-69).
Выход: 0 — все зелёные, 1 — есть ЖДЁТ (не полны замеры), 2 — есть красный.
Запуск: research/.venv/bin/python research/it70_verdict.py
"""
import bisect
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
R = ROOT / "research"


def fp_rates(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {k: sum(int(r[f"fp@{k}"]) for r in rows) / len(rows) for k in ("0.25", "0.4", "0.5")}


def metric(evdir: str, key: str) -> float:
    return json.loads((MD / "train/runs/eval" / evdir / "metrics.json").read_text())[key]


def sahi_flight_recall(path: Path) -> float:
    gt_files = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
    gt_ts = [float(Path(g).stem) for g in gt_files]
    gt_z = [float(np.load(g)[2]) for g in gt_files]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    flight = [r for r in rows if gt_z[min(len(gt_z) - 1, bisect.bisect_left(gt_ts, float(r["img"])))] > 1.0]
    return sum(1 for r in flight if float(r["conf_sahi"]) >= 0.5) / max(1, len(flight))


def session_share(path: Path, thr: str) -> tuple[int, int]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return sum(int(r[f"det@{thr}"]) for r in rows), len(rows)


def fusion_row(path: Path, policy: str, col: str) -> float:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return float(next(r for r in rows if r["policy"] == policy)[col])


def event_delay(path: Path, config: str) -> float:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    r = next(x for x in rows if x["config"] == config)
    return max(float(v) for v in r["delay_s"].split("–"))


def main() -> int:
    checks = []  # (id, описание, получено, ожидаем, ok|None)

    f = R / "coco_bg_fp_it70v1.csv"
    if f.exists():
        d = fp_rates(f)
        ok = d["0.25"] <= 0.25 and d["0.5"] <= 0.128
        checks.append(("E1", "FP независ. 400 фонов: @0,25≤25% и @0,5≤12,8%", f"{d['0.25']:.1%}/{d['0.5']:.1%}", "≤25/≤12,8", ok))
    else:
        checks.append(("E1", "FP независ. 400 фонов", "нет файла", "≤25/≤12,8", None))

    for cid, ev, op, thr in (("E2", "visual-bg70-dut600", ">", 0.720), ("E3", "visual-bg70-hf600", "≥", 0.837)):
        p = MD / "train/runs/eval" / ev / "metrics.json"
        if p.exists():
            m = metric(ev, "map50")
            checks.append((cid, f"mAP50 {ev.replace('visual-bg70-', '')} {op} {thr}", f"{m:.3f}", f"{op}{thr}", m > thr if cid == "E2" else m >= thr))
        else:
            checks.append((cid, f"mAP50 {ev}", "нет файла", f"{op}{thr}", None))

    f = R / "session_vis_probe_bg70.csv"
    if f.exists():
        k, n = session_share(f, "0.5")
        checks.append(("E4", "стоящий дрон: сек с детектом @0,5 ≥ 16/18", f"{k}/{n}", "≥16/18", k >= 16))
    else:
        checks.append(("E4", "session probe @0,5", "нет файла", "≥16/18", None))

    f = R / "mmaud_sahi_full_bg70.csv"
    if f.exists():
        v = sahi_flight_recall(f)
        checks.append(("E5", "SAHI MMAUD recall полёт @0,5 ≥ 89,5%", f"{v:.1%}", "≥89,5%", v >= 0.895))
    else:
        checks.append(("E5", "SAHI MMAUD recall полёт", "нет файла", "≥89,5%", None))

    f = R / "fusion_sim_results-bg70.csv"
    if f.exists():
        r6 = fusion_row(f, "video-only", "R")
        r7 = fusion_row(f, "late 0.5/0.5", "F1")
        checks.append(("E6", "контур: video-only recall airborne-окон ≥ 0,98", f"{r6:.3f}", "≥0,98", r6 >= 0.98))
        checks.append(("E7", "контур: F1 late(τ=0,5) ≥ 0,95", f"{r7:.3f}", "≥0,95", r7 >= 0.95))
    else:
        checks.append(("E6", "контур video-only R окон", "нет файла", "≥0,98", None))
        checks.append(("E7", "контур F1 late", "нет файла", "≥0,95", None))

    f = R / "event_metrics-bg70.csv"
    if f.exists():
        d8 = event_delay(f, "late τ=0.5")
        checks.append(("E8", "контур: задержка late ≤ канон(0,0)+1,0 с", f"{d8:.1f}", "≤1,0", d8 <= 1.0))
    else:
        checks.append(("E8", "контур задержка", "нет файла", "≤1,0", None))

    print(f"{'крит':<5} {'проверка':<52} {'значение':>12} {'порог':>11}  вердикт")
    ready = True
    for cid, name, val, thr, ok in checks:
        if ok is None:
            ready = False
            verdict = "ЖДЁТ"
        else:
            verdict = "OK" if ok else "КРАСНЫЙ"
        print(f"{cid:<5} {name:<52} {val:>12} {thr:>11}  {verdict}")
    if not ready:
        print("\nЕсть незавершённые замеры — решение об экспорте не принимается.")
        return 1
    all_ok = all(ok for *_, ok in checks)
    print("\nВЕРДИКТ:", "Экспортируем (все E1–E8 зелёные)" if all_ok else "НЕ экспортируем — красный критерий, честная констатация в отчёт")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
