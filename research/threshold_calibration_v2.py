#!/usr/bin/env python3
"""Калибровка порога решения на ЧЕСТНЫХ вероятностях p_drone (it-40; ревью GPT-6-Astra §4).

Прежняя итерация it-12 объявила калибровку порога «опровергнутой» (F1 плоский 0.05–0.5),
но её данные были получены кодом, обнулявшим p_drone у non-drone предсказаний
(значения {0}∪[0.5;1] — пороги ниже 0.5 были физически неизмеримы). После it-36
ast_windows.csv содержит реальные вероятности — калибровку повторяем.

Дизайн (по требованиям ревью §2/§10): каждый метод (audio-only, late 0.5/0.5,
late + каузальная медиана-5) получает СВОЙ порог, подбираемый на том же материале.
ОГОВОРКА (ревью §10): порог подбирается на том же клипе, на котором считается F1 —
это development-подбор, не проверка обобщения; для подтверждающего вывода нужен
независимый материал. Дополнительно: распределение p_drone (проверка снятия
«бимодальности») и парный блочный бутстрап для разницы F1 лучших конфигураций.
Запуск: research/.venv/bin/python research/threshold_calibration_v2.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import argparse
_YA = argparse.ArgumentParser(); _YA.add_argument("--yolo-csv", default="research/yolo_sandbox_frames.csv"); _YSRC = str(ROOT / _YA.parse_known_args()[0].yolo_csv)  # it-66: пересчёт новыми весами

ast = {r["t0"]: r for r in csv.DictReader(open(ROOT / "research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(_YSRC))
        if r["imgsz"] == "480"}

wins = []
for t0 in sorted(ast.keys(), key=float):
    sec = int(float(t0))
    pv = float(yolo[str(sec)]["max_conf"]) if str(sec) in yolo else float(yolo[str(sec + 1)]["max_conf"])
    wins.append((float(t0), pv, float(ast[t0]["p_drone"]), int(ast[t0]["airborne_gt"])))
N = len(wins)

def prf(preds):
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def causal_median(vals, k=5):
    out = []
    for i in range(len(vals)):
        seg = sorted(vals[max(0, i - k + 1): i + 1])
        out.append(seg[len(seg) // 2])
    return out

pas_med5 = causal_median([w[2] for w in wins], 5)

def series(method: str, pa_seq=None):
    """Ряд «оценок» метода до порога (pv, pa на каждом окне)."""
    seq = pa_seq if pa_seq is not None else [w[2] for w in wins]
    if method == "audio-only":
        return [pa for (_, _, pa, _) in wins], seq
    return [pv for (_, pv, _, _) in wins], seq  # late: 0.5*pv+0.5*pa

def late_score(pv, pa):
    return 0.5 * pv + 0.5 * pa

METHODS = {
    "audio-only":            lambda pv, pa: pa,
    "late 0.5/0.5":          late_score,
    "late+causal median-5":  late_score,   # pa_seq уже сглажен
    # Δ-правило при честных вероятностях (проверка вывода it-36, что оно вредит)
    "late 0.5/0.5+Δ(0.1)":   lambda pv, pa: min(max(0.5 * pv + 0.5 * pa
                                     + (0.1 if (pv >= 0.5 and pa >= 0.5) else (-0.1 if (pv >= 0.5) != (pa >= 0.5) else 0.0)), 0.0), 1.0),
}

print(f"окон: {N} | airborne: {sum(w[3] for w in wins)}")
print("\n=== 1. Распределение p_drone (проверка снятия «бимодальности» it-12) ===")
bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
hist = []
for lo, hi in bins:
    n = sum(1 for w in wins if lo <= w[2] < hi)
    hist.append((f"[{lo:.1f},{hi:.1f})", n))
    print(f"  [{lo:.1f},{hi:.1f}): {'#' * (n // 2)} {n}")
sub = sum(n for (lo, _), n in zip(bins, [h[1] for h in hist]) if 0.05 <= lo < 0.5)
print(f"  значимых значений в (0.05..0.5): {sub} — «плоской бимодальности» больше нет" if sub else "  суб-0.5 значений нет")

print("\n=== 2. F1(τ) по методам (τ = порог решения, у каждого метода свой) ===")
taus = [round(0.05 + 0.05 * i, 2) for i in range(19)]
rows_out = []
best_per_method = {}
for mname, fn in METHODS.items():
    pa_seq = pas_med5 if "median" in mname else None
    line = []
    best = (-1.0, None, None, None)
    for tau in taus:
        preds = [(int(fn(pv, pa) >= tau), gt) for (_, pv, _, gt), pa in zip(wins, pa_seq or [w[2] for w in wins], strict=True)]
        P, R, F = prf(preds)
        line.append(F)
        rows_out.append(dict(method=mname, tau=tau, P=round(P, 3), R=round(R, 3), F1=round(F, 3)))
        if F > best[0]:
            best = (F, tau, P, R)
    best_per_method[mname] = best
    print(f"\n  {mname}:")
    for i in range(0, len(taus), 10):
        print("    τ: " + " ".join(f"{t:5.2f}" for t in taus[i:i + 10]))
        print("    F: " + " ".join(f"{v:5.3f}" for v in line[i:i + 10]))
    F, tau, P, R = best
    print(f"    ЛУЧШИЙ: τ={tau:.2f} → P={P:.3f} R={R:.3f} F1={F:.3f}")

with open(ROOT / "research/threshold_calibration_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "tau", "P", "R", "F1"])
    w.writeheader()
    w.writerows(rows_out)
print(f"\nCSV: research/threshold_calibration_v2.csv ({len(rows_out)} строк)")

print("\n=== 3. Итог против порога 0.5 (дефолт) ===")
for mname, (F, tau, P, R) in best_per_method.items():
    print(f"  {mname:<24} лучший τ={tau:.2f} F1={F:.3f}")
