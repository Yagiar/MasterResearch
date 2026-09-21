#!/usr/bin/env python3
"""it-78 — семейство обобщённого среднего M_p(c_old,c_new) × τ: закрытие измерения «функция голосования».

Ноль инференса: канонические ряды it-74 (MMAUD SAHI 5091, V1-фоны 400, сек-ряды sandbox для D3).
M_p = ((x^p+y^p)/2)^(1/p); границы сетки анкерятся на it-74: p→−∞ = AND, p→+∞ = OR (сами-проверки).
Критерии предрегистрированы: iterations/it-78-gmean-vote-PLANNED.md (D1 доминирование AND@0,40,
D2 Парето-карта — справка, D3 контур только при зелёном D1).
Запуск: research/.venv/bin/python research/it78_gmean_vote.py
"""
import bisect
import csv
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
PS = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TAS = [0.30, 0.35, 0.40, 0.45, 0.50]
LINES = []


def log(s=""):
    LINES.append(s)
    print(s, flush=True)


def gmean(x: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """M_p = max·((a^p+b^p)/2)^(1/p), a=x/max, b=y/max — численно устойчиво (a,b≤1).
    p<0: нуль в одном входе обнуляет пару (AND-семантика при p→−∞)."""
    if p == float("-inf"):
        return np.minimum(x, y)
    if p == float("inf"):
        return np.maximum(x, y)
    m0 = np.maximum(x, y)
    n0 = np.minimum(x, y)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if p > 0:  # scale by max: (a^p + b^p) с a,b ≤ 1 → без перебора
            a, b = x / m0, y / m0
            m = m0 * np.exp((np.log(a ** p + b ** p) - math.log(2)) / p)
        else:  # scale by min: (1 + (max/min)^p) устойчиво при p<0
            r = np.where(n0 > 0, m0 / n0, np.inf)
            m = n0 * np.exp((np.log(1.0 + r ** p) - math.log(2)) / p)
    if p < 0:
        m[(x == 0) | (y == 0)] = 0.0
    return np.nan_to_num(m, nan=0.0, posinf=1.0)


def _selftest_gmean():
    x = np.array([0.9, 0.5, 0.0, 1.0])
    y = np.array([0.3, 0.5, 0.4, 1.0])
    assert np.allclose(gmean(x, y, -1.0), np.where((x == 0) | (y == 0), 0.0, 2 * x * y / (x + y)), atol=1e-9), "p=-1 (гармоника) не совпала"
    assert np.allclose(gmean(x, y, 1.0)[1:], (x + y)[1:] / 2, atol=1e-9), "p=+1 (арифметика) не совпала"
    assert np.allclose(gmean(x, y, -1e9), np.minimum(x, y), atol=1e-6), "p→−∞ не даёт min"
    assert np.allclose(gmean(x, y, 1e9), np.maximum(x, y), atol=1e-6), "p→+∞ не даёт max"


_selftest_gmean()


# --- MMAUD полёт/дальн (конвенция it-74/it65_union: GT-высота z>1, дальн 10–19 м) ---
gt_files = sorted(glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt = [np.load(g) for g in gt_files]


def gt_at(t):
    return gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))]


old_s = {r["img"]: float(r["conf_sahi"]) for r in csv.DictReader(open(ROOT / "research/mmaud_sahi_full.csv", encoding="utf-8"))}
new_s = {r["img"]: float(r["conf_sahi"]) for r in csv.DictReader(open(ROOT / "research/mmaud_sahi_full_new.csv", encoding="utf-8"))}
keys = sorted(set(old_s) & set(new_s), key=float)
assert len(keys) == 5091, f"join MMAUD {len(keys)} != 5091"
xo = np.array([old_s[k] for k in keys])
xn = np.array([new_s[k] for k in keys])
air, far = [], []
for k in keys:
    x, y, z = gt_at(float(k))
    air.append(z > 1.0)
    far.append(z > 1.0 and 10.0 <= math.hypot(x, y) < 20.0)
air, far = np.array(air), np.array(far)

# --- V1-фоны (400) ---
fo = {r["img"]: float(r["max_conf"]) for r in csv.DictReader(open(ROOT / "research/coco_bg_fp_it69v1-old.csv", encoding="utf-8"))}
fn_ = {r["img"]: float(r["max_conf"]) for r in csv.DictReader(open(ROOT / "research/coco_bg_fp_it69v1-new.csv", encoding="utf-8"))}
fkeys = sorted(set(fo) & set(fn_))
assert len(fkeys) == 400, f"join фонов {len(fkeys)} != 400"
bo = np.array([fo[k] for k in fkeys])
bn = np.array([fn_[k] for k in fkeys])

log("it-78: M_p-сетка голосования (11 внутренних p + крайние точки AND/OR) × τ; материал it-74")
rows = []


def evaluate(p):
    m_air = gmean(xo, xn, p)
    m_bg = gmean(bo, bn, p)
    out = []
    for ta in TAS:
        rec = float((m_air[air] >= ta).mean())
        fp = float((m_bg >= ta).mean())
        rf = float((m_air[far] >= ta).mean())
        out.append((ta, rec, fp, rf))
        rows.append((p if isinstance(p, float) else float(p), *out[-1]))
    return out


res = {p: evaluate(p) for p in PS}
anch_and, anch_or = evaluate(float("-inf")), evaluate(float("inf"))

log("\n== сами-проверки крайних точек (против it-74: AND@0,40 = 94,7/7,2; OR@0,35 = 100,0/49,8) ==")
a40 = anch_and[TAS.index(0.40)]
o35 = anch_or[TAS.index(0.35)]
ok1 = abs(a40[1] * 100 - 94.7) < 0.06 and abs(a40[2] * 100 - 7.2) < 0.06
ok2 = abs(o35[1] * 100 - 100.0) < 0.06 and abs(o35[2] * 100 - 49.8) < 0.06
log(f"  AND@0,40: rec={a40[1]*100:.1f} FP={a40[2]*100:.1f} → {'OK' if ok1 else 'ПРОВАЛ'}; OR@0,35: rec={o35[1]*100:.1f} FP={o35[2]*100:.1f} → {'OK' if ok2 else 'ПРОВАЛ'}")
assert ok1 and ok2, "крайние точки не совпали с it-74 — сетка невалидна"

log("\n== D2: Парето-карта (реколл полёта %, при FP % в скобках; строки — p) ==")
hdr = "     p    " + "".join(f"τ={t:.2f}".rjust(16) for t in TAS)
log(hdr)
for p, vals in zip(["AND", *PS, "OR"], [anch_and, *[res[q] for q in PS], anch_or]):
    log(f"{str(p):>9} " + "".join(f"{r*100:>7.1f} ({f*100:>4.1f})".rjust(16) for _, r, f, _ in vals))

log("\n== D1: доминирование AND@0,40 (rec ≥ 94,7 ∧ FP ≤ 7,2 ∧ [rec ≥ 95,5 ∨ FP ≤ 6,0]) ==")
cands = []
for p_label, vals in zip(["AND", *PS, "OR"], [anch_and, *[res[q] for q in PS], anch_or]):
    for ta, rec, fp, rf in vals:
        if rec * 100 >= 94.7 - 1e-9 and fp * 100 <= 7.2 + 1e-9 and (rec * 100 >= 95.5 or fp * 100 <= 6.0):
            cands.append((p_label, ta, rec, fp, rf))
if not cands:
    log("  кандидатов нет → D1 КРАСНЫЙ: p-семейство не улучшает frontier AND@0,40;")
    log("  вывод it-74 распространяется на всё семейство средних (рамка: «исчерпано над p-семейством»), D3 не считается.")
else:
    best = max(cands, key=lambda c: (c[2], -c[3]))
    log(f"  ЗЕЛЁНЫЙ, {len(cands)} точек; лучшая (по rec, затем −FP): p={best[0]}, τ={best[1]:.2f} → rec={best[2]*100:.1f} FP={best[3]*100:.1f} дальн={best[4]*100:.1f}")
    log("  D3 (контур) считается в отдельном дополнении при подтверждении автора-цикла — см. журнал итерации.")

with open(ROOT / "research/it78_gmean_vote.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["p", "tau", "rec_flight", "fp_pct", "rec_far"])
    for p, ta, rec, fp, rf in rows:
        w.writerow([p, f"{ta:.2f}", f"{rec:.6f}", f"{fp:.6f}", f"{rf:.6f}"])
(ROOT / "research/it78_gmean_vote.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
print("\nартефакты: research/it78_gmean_vote.{csv,txt}")
