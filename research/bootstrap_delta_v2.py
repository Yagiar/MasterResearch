#!/usr/bin/env python3
"""Парные интервальные оценки и приговор Δ-правилу (it-41; ревью GPT-6-Astra §4/§10).

(1) Парный блочный бутстрап разницы F1 между конфигурациями: блоки = последовательные
    отрезки по 5 с (10 окон) — учитывает временную автокорреляцию перекрывающихся окон,
    в отличие от пересэмплирования отдельных окон как независимых (прямое требование
    ревью: «интервалы с учётом группировки по записям»; одна запись ⇒ блоки по времени —
    диагностический максимум возможного, НЕ независимая проверка обобщения).
(2) Полный перебор Δ (delta_conf × delta_unconf) при честных вероятностях: it-36
    показал, что дефолтные ±0.1/−0.1 вредят; здесь проверяем, помогает ли Δ хоть где-то.
Запуск: research/.venv/bin/python research/bootstrap_delta_v2.py
"""
import csv
import random
from pathlib import Path
from statistics import mean, quantiles

ROOT = Path(__file__).resolve().parent.parent
import argparse
_YA = argparse.ArgumentParser(); _YA.add_argument("--yolo-csv", default="research/yolo_sandbox_frames.csv"); _YA.add_argument("--suffix", default="", help="суффикс для вых. CSV, напр. -new (it-66)"); _ARGS = _YA.parse_known_args()[0]; _YSRC = str(ROOT / _ARGS.yolo_csv); _SFX = _ARGS.suffix  # it-66: пересчёт новыми весами
B = 2000  # бутстрап-реплик
BLOCK_WINDOWS = 10  # 10 окон × 0.5 с = 5 с клипа

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

CONFIGS = {
    "audio-only τ=0.25":        [int(pa >= 0.25) for (_, _, pa, _) in wins],
    "audio-only τ=0.5":         [int(pa >= 0.5) for (_, _, pa, _) in wins],
    "late τ=0.5":               [int(0.5 * pv + 0.5 * pa >= 0.5) for (_, pv, pa, _) in wins],
    "late+med5 τ=0.55":         [int(0.5 * pv + 0.5 * pa >= 0.55) for (_, pv, _, _), pa in zip(wins, pas_med5, strict=True)],
    "video-only τ=0.5":         [int(pv >= 0.5) for (_, pv, _, _) in wins],
}

print(f"окон: {N} | airborne: {sum(w[3] for w in wins)} | блок бутстрапа: {BLOCK_WINDOWS} окон (5 с), {B} реплик")
point = {}
for name, preds in CONFIGS.items():
    P, R, F = prf([(p, gt) for p, (_, _, _, gt) in zip(preds, wins, strict=True)])
    point[name] = F
    print(f"  {name:<22} F1={F:.3f} (P={P:.3f} R={R:.3f})")

def block_bootstrap_diff(a: str, b: str):
    """Парная разница F1(a)−F1(b): ресэмплируем БЛОКИ окон, внутри блока берём оба ряда."""
    pa, pb = CONFIGS[a], CONFIGS[b]
    n_blocks = (N + BLOCK_WINDOWS - 1) // BLOCK_WINDOWS
    diffs = []
    rng = random.Random(20260905)
    for _ in range(B):
        ia, fa = [], []
        ib, fb = [], []
        for _b in range(n_blocks):
            s = rng.randrange(N)
            idx = [(s + j) % N for j in range(BLOCK_WINDOWS)]
            ia += [pa[i] for i in idx]; fa += [wins[i][3] for i in idx]
            ib += [pb[i] for i in idx]; fb += [wins[i][3] for i in idx]
        Fa = prf(list(zip(ia, fa, strict=True)))[2]
        Fb = prf(list(zip(ib, fb, strict=True)))[2]
        diffs.append(Fa - Fb)
    diffs.sort()
    lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
    return mean(diffs), lo, hi

print("\n=== 1. Парные разницы F1 (блочный бутстрап, 95% интервал) ===")
pairs = [("audio-only τ=0.25", "late+med5 τ=0.55"),
         ("audio-only τ=0.25", "late τ=0.5"),
         ("late+med5 τ=0.55", "late τ=0.5"),
         ("audio-only τ=0.5", "late+med5 τ=0.55"),
         ("audio-only τ=0.25", "audio-only τ=0.5")]
rows_out = []
for a, b in pairs:
    m, lo, hi = block_bootstrap_diff(a, b)
    verdict = "значима" if (lo > 0 or hi < 0) else "НЕ значима"
    print(f"  {a} − {b}: ΔF1 = {point[a] - point[b]:+.3f}  [{lo:+.3f}; {hi:+.3f}]  → {verdict}")
    rows_out.append(dict(a=a, b=b, dF1=round(point[a] - point[b], 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), significant=verdict))

with open(ROOT / f"research/bootstrap_pairs{_SFX}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["a", "b", "dF1", "ci_lo", "ci_hi", "significant"])
    w.writeheader(); w.writerows(rows_out)

print("\n=== 2. Приговор Δ-правилу: перебор delta_conf × delta_unconf (late, τ=0.5, честные p) ===")
best = (-1.0, None)
table = []
for dc in (0.0, 0.05, 0.1, 0.15):
    row = []
    for du in (0.0, 0.05, 0.1, 0.15):
        preds = []
        for (_, pv, pa, gt) in wins:
            s = 0.5 * pv + 0.5 * pa
            if pv >= 0.5 and pa >= 0.5:
                s += dc
            elif (pv >= 0.5) != (pa >= 0.5):
                s -= du
            preds.append((int(min(max(s, 0.0), 1.0) >= 0.5), gt))
        F = prf(preds)[2]
        row.append(F)
        table.append(dict(delta_conf=dc, delta_unconf=du, F1=round(F, 3)))
        if F > best[0]:
            best = (F, (dc, du))
    print("  δc\\δu: " + " ".join(f"{du:5.2f}" for du in (0.0, 0.05, 0.1, 0.15)))
    print("       " + " ".join(f"{v:5.3f}" for v in row) + f"   (δc={dc})")
print(f"\n  ЛУЧШИЙ Δ: δc={best[1][0]}, δu={best[1][1]} → F1={best[0]:.3f}  "
      f"(без Δ: {point['late τ=0.5']:.3f}) → Δ {'не помогает' if best[0] <= point['late τ=0.5'] + 1e-9 else 'помогает на ' + format(best[0] - point['late τ=0.5'], '+.3f')}")

with open(ROOT / f"research/delta_sweep{_SFX}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["delta_conf", "delta_unconf", "F1"])
    w.writeheader(); w.writerows(table)
print("\nCSV: research/bootstrap_pairs.csv, research/delta_sweep.csv")
