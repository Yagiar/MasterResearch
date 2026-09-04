#!/usr/bin/env python3
"""Скоринг этапов ablation v2 по срезам decisions.jsonl против GT (research/it-30).

Использование: research/.venv/bin/python research/score_stages.py B=START:END C=... D=...
где START:END — 1-based диапазон строк jsonl этапа (из лога ablation_v2.sh).
Фаза клипа: перебор смещения t0 в [-5, +5] с шагом 0.25 с вокруг первого решения этапа
(симулятор стартует клип с t=0); выбирается фаза с максимумом F1 против GT «airborne»
(отметить в анализе: лёгкий оптимизм подгонки).
"""
import csv
import json
import sys

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"
CLIP = 72.609

gt = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(f"{ROOT}/research/gt_sandbox_video.csv"))}

rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]

def prf(preds):
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

for arg in sys.argv[1:]:
    name, rng = arg.split("=")
    a, b = (int(x) for x in rng.split(":"))
    seg = rows[a - 1:b]
    if not seg:
        print(f"{name}: пусто")
        continue
    t_first = seg[0]["ts"]
    best = (-1.0, None)
    for off100 in range(-20, 21):
        t0 = t_first - off100 * 0.25
        preds = [(int(r["decision"]), gt[int((r["ts"] - t0) % CLIP)][1]) for r in seg]
        P, R, F = prf(preds)
        if F > best[0]:
            best = (F, t0, P, R)
    F, t0, P, R = best
    # скоринг при найденной фазе: по всем модам отдельно
    print(f"\n### Этап {name}: n={len(seg)}, выравнивание t0-t_first={t0 - t_first:+.2f} с")
    for mode in sorted({r["mode"] for r in seg}):
        rs = [r for r in seg if r["mode"] == mode]
        preds = [(int(r["decision"]), gt[int((r["ts"] - t0) % CLIP)][1]) for r in rs]
        P, R, F = prf(preds)
        mixed = sum(1 for r in rs if r["contributions"]["p_a"] is not None)
        print(f"  {mode:<12} vs airborne: P={P:.3f} R={R:.3f} F1={F:.3f} (n={len(rs)}, смешанных={mixed})")
    # подмножества GT
    for gname, gix in (("visible", 0), ("airborne", 1)):
        preds = [(int(r["decision"]), gt[int((r["ts"] - t0) % CLIP)][gix]) for r in seg]
        P, R, F = prf(preds)
        print(f"  ИТОГ {mode if False else 'все':<12} vs {gname:<8}: P={P:.3f} R={R:.3f} F1={F:.3f}")
