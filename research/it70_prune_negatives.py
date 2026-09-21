#!/usr/bin/env python3
"""it-70, шаг 1: детерминированный ПРУНИНГ фоновых негативов корпуса it-65 (стратифицированный).

Единственный рычаг итерации: доля COCO-фонов в train 1,26 % -> ~0,4 %.
Keep списывается СВОИМ срезом в каждом сплите (независимый seed=20260921 — сид 1337
в dry-run показал структурную корреляцию с _split_groups: val/test-фоны попали в keep
целиком): train 720->240, val 90->30, test 90->30.
Удаляем из _prepared/visual/images|labels/{train,val,test} coco-background_*, чьи
исходные stems не в keep своего сплита. Исходники train/data/coco-background НЕ трогаем
(подготовленные картинки — жёсткие ссылки; полный корпус = prepare-visual it-65).

Запуск: research/.venv/bin/python research/it70_prune_negatives.py --apply  (без --apply — dry-run)
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "MasterDiploma/train/data"
PREPARED = DATA / "_prepared/visual"
KEEP_PER_SPLIT = {"train": 240, "val": 30, "test": 30}
SEED = 20260921
# prepared-имя негатива: "coco-background_{orig}_{i:07d}_{orig}", orig вида coco_bg_00012
NAME_RE = re.compile(r"^coco-background_(.+)_\d{7}_(.+)$")


def orig_stem(fname_stem: str) -> str:
    m = NAME_RE.match(fname_stem)
    assert m and m.group(1) == m.group(2), f"неожиданное имя: {fname_stem}"
    return m.group(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    keep: dict[str, set[str]] = {}
    per_split_origs: dict[str, list[str]] = {}
    for sub, n_keep in KEEP_PER_SPLIT.items():
        ors = sorted({orig_stem(f.stem) for f in (PREPARED / "images" / sub).glob("coco-background_*")})
        assert len(ors) == {"train": 720, "val": 90, "test": 90}[sub], f"{sub}: ожидал полный корпус it-65, нашёл {len(ors)}"
        per_split_origs[sub] = ors
        keep[sub] = set(random.Random(SEED).sample(ors, n_keep))
        assert len(keep[sub]) == n_keep

    deleted = Counter()
    for sub in KEEP_PER_SPLIT:
        for kind in ("images", "labels"):
            for f in sorted((PREPARED / kind / sub).glob("coco-background_*")):
                if orig_stem(f.stem) in keep[sub]:
                    continue
                deleted[(sub, kind)] += 1
                if args.apply:
                    f.unlink()

    print(("APPLIED" if args.apply else "DRY-RUN"), "удалено:", dict(deleted))
    if args.apply:
        out = ROOT / "research/it70_neg_keep.txt"
        out.write_text("".join(f"{sub}\t{o}\n" for sub in KEEP_PER_SPLIT for o in sorted(keep[sub])), encoding="utf-8")
        print(f"keep-list записан: {out}")
        for sub in KEEP_PER_SPLIT:
            d = PREPARED / "images" / sub
            n_bg = len(list(d.glob("coco-background_*")))
            n_all = len(list(d.iterdir()))
            print(f"{sub}: фонов {n_bg} / всего {n_all} ({n_bg / n_all:.2%})")
        # контроль целостности: у каждого оставшегося негатива есть пустой label
        bad = [str(f) for sub in KEEP_PER_SPLIT for f in (PREPARED / "images" / sub).glob("coco-background_*")
               if not (PREPARED / "labels" / sub / (f.stem + ".txt")).exists()
               or (PREPARED / "labels" / sub / (f.stem + ".txt")).stat().st_size != 0]
        print("labels-целостность:", "OK" if not bad else f"ПРОБЛЕМЫ: {bad[:5]}")


if __name__ == "__main__":
    main()
