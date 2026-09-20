#!/usr/bin/env python3
"""Детерминированная подвыборка DUT-кадров из test-сплита (it-65 контроль).

Зачем: корпус it-65 собрал три домена (hf-drone-detection, DUT, фоны); у СТАРОЙ модели
нет замера на новом домене DUT. Чтобы «сравнить старую и новую» без полного val по
4915 кадрам на CPU, фиксируется сэмпл 600 DUT-кадров test (symlink-каталог + data.yaml):
на нём гоняются обе модели одних протоколом.
Запуск: research/.venv/bin/python research/make_dut_test_subset.py
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_IMG = ROOT / "MasterDiploma/train/data/_prepared/visual/images/test"
TEST_LAB = ROOT / "MasterDiploma/train/data/_prepared/visual/labels/test"
OUT = ROOT / "MasterDiploma/train/data/_prepared/visual-dut-test600"
N, SEED = 600, 65

dut = sorted(p for p in TEST_IMG.glob("*.jpg") if p.stem.startswith("dut-anti-uav"))
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
