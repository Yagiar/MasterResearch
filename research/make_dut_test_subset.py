#!/usr/bin/env python3
"""Детерминированные подвыборки test-кадров по домену (it-65 контроль).

Зачем: корпус it-65 собрал три домена (hf-drone-detection, DUT, фоны); у СТАРОЙ модели
нет замеров на новых доменах. Чтобы «сравнить старую и новую» без полного val по
4915 кадрам на CPU, фиксируются сэмплы 600 кадров домена из test (symlink-каталог +
data.yaml): на них гоняются обе модели одним протоколом.
Запуск: research/.venv/bin/python research/make_dut_test_subset.py [--prefix dut-anti-uav|hf]
"""
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_IMG = ROOT / "MasterDiploma/train/data/_prepared/visual/images/test"
TEST_LAB = ROOT / "MasterDiploma/train/data/_prepared/visual/labels/test"
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="dut-anti-uav")
ap.add_argument("--tag", default="dut-test600", help="имя каталога: visual-<tag>")
ap.add_argument("-n", type=int, default=600)
ap.add_argument("--seed", type=int, default=65)
args = ap.parse_args()
OUT = ROOT / f"MasterDiploma/train/data/_prepared/visual-{args.tag}"
N, SEED = args.n, args.seed

dut = sorted(p for p in TEST_IMG.glob("*.jpg") if p.stem.startswith(args.prefix))
random.Random(SEED).shuffle(dut)
picked = dut[:N]

for sub in ("images/test", "labels/test"):
    d = OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.is_symlink() or f.is_file():
            f.unlink()
for p in picked:
    lab = TEST_LAB / (p.stem + ".txt")
    (OUT / "images/test" / p.name).symlink_to(p)
    if lab.exists():
        (OUT / "labels/test" / lab.name).symlink_to(lab)
# ultralytics check_det_dataset требует наличие всех сплитов — зеркалим test
for split in ("train", "val"):
    link = OUT / "images" / split
    if not link.is_symlink():
        link.symlink_to(OUT / "images/test")

src = (ROOT / "MasterDiploma/train/data/_prepared/visual/data.yaml").read_text()
(OUT / "data.yaml").write_text(src.replace(
    str(ROOT / "MasterDiploma/train/data/_prepared/visual"), str(OUT)))
print(f"DUT test-кадров всего: {len(dut)}, взято: {len(picked)}, seed={SEED}")
print(f"каталог: {OUT}")
