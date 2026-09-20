#!/usr/bin/env python3
"""Сверка новой модели с предрегистрированными критериями E1–E5 (it-65).

Читает артефакты цепочек it65_chain/chain2 и baseline'ы старой модели, печатает
таблицу вердиктов; выход 0 — все зелёные (экспорт), 1 — есть красный/нет данных.
Критерии зафиксированы ДО просмотра метрик (it-65, коммит 18123aa).
Запуск: research/.venv/bin/python research/it65_verdict.py
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


def fp_rate(path: Path, thr: str) -> float:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return sum(int(r[f"fp@{thr}"]) for r in rows) / len(rows)


def metric(evdir: str, key: str) -> float:
    return json.loads((MD / "train/runs/eval" / evdir / "metrics.json").read_text())[key]


def sahi_flight_recall(path: Path) -> float:
    """Recall SAHI (conf_sahi ≥ 0.5) на летящих кадрах; сегментация по лидарному GT (z>1 м)."""
    gt_files = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
    gt_ts = [float(Path(g).stem) for g in gt_files]
    gt_z = [float(np.load(g)[2]) for g in gt_files]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    flight = [r for r in rows if gt_z[min(len(gt_z) - 1, bisect.bisect_left(gt_ts, float(r["img"])))] > 1.0]
    return sum(1 for r in flight if float(r["conf_sahi"]) >= 0.5) / max(1, len(flight))


def session_share(path: Path, thr: str) -> tuple[int, int]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return sum(int(r[f"det@{thr}"]) for r in rows), len(rows)


def main() -> int:
    checks = []  # (id, описание, получено, ожидаем, готово)

    f_new = R / "coco_bg_fp_new-yolov8s-bg.csv"
    if f_new.exists():
        v = fp_rate(f_new, "0.5")
        checks.append(("E1", f"FP COCO-фоны @0,5 ≤ 12,8% (base 25,6%)", f"{v:.1%}", "≤12,8%", v <= 0.128))
    else:
        checks.append(("E1", "FP COCO-фоны @0,5", "нет файла", "≤12,8%", None))

    for cid, ev, op, base in (("E2", "visual-new-dut600", ">", "0,720"), ("E3", "visual-new-hf600", "≥", "0,837")):
        p = MD / "train/runs/eval" / ev / "metrics.json"
        if p.exists():
            m = metric(ev, "map50")
            ok = m > 0.720 if cid == "E2" else m >= 0.837
            checks.append((cid, f"mAP50 {ev.replace('visual-new-', '')} {op} {base}", f"{m:.3f}", base, ok))
        else:
            checks.append((cid, f"mAP50 {ev}", "нет файла", base, None))

    f_s = R / "session_vis_probe_new-yolov8s-bg.csv"
    if f_s.exists():
        k, n = session_share(f_s, "0.5")
        checks.append(("E4", f"стоящий дрон: сек с детектом @0,5 ≥ 16/18", f"{k}/{n}", "≥16/18", k >= 16 and k / max(n, 1) >= 16 / 18))
    else:
        checks.append(("E4", "session probe @0,5", "нет файла", "≥16/18", None))

    f_m = R / "mmaud_sahi_full_new.csv"
    if f_m.exists():
        v = sahi_flight_recall(f_m)
        checks.append(("E5", "SAHI MMAUD recall полёт ≥ 89,5% (base 94,5%)", f"{v:.1%}", "≥89,5%", v >= 0.895))
    else:
        checks.append(("E5", "SAHI MMAUD recall полёт", "нет файла", "≥89,5%", None))

    print(f"{'крит':<5} {'проверка':<48} {'значение':>10} {'порог':>9}  вердикт")
    ready = True
    for cid, name, val, thr, ok in checks:
        if ok is None:
            ready = False
            verdict = "ЖДЁТ"
        else:
            verdict = "OK" if ok else "КРАСНЫЙ"
        print(f"{cid:<5} {name:<48} {val:>10} {thr:>9}  {verdict}")
    if not ready:
        print("\nЕсть незавершённые замеры — решение об экспорте не принимается.")
        return 1
    all_ok = all(ok for *_, ok in checks)
    print("\nВЕРДИКТ:", "Экспортируем (все E1–E5 зелёные)" if all_ok else "НЕ экспортируем — красный критерий, честная констатация в отчёт")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
