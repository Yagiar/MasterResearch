#!/usr/bin/env python3
"""it-84 — парный (exact paired replay) разбор FP-эффекта AND@0,4 на бездронном материале.

Отличие от it-82 (не paired): те же самые кадры (total ordering по media_ts, loop=false)
прогоняются в обоих плечах, сравнение попоточно-сопоставленное, а не по двум независимым
выборкам из одного распределения.

Использование:
  research/.venv/bin/python research/it84_paired_eval.py [--jsonl PATH] [--expect-frames 400]
      A=START:END B=START:END
    (срезы — 1-based строки decisions.jsonl из «SCORE_OFF» лога it84)

Метрика:
  * Единица — кадр (уникальный media_ts внутри этапа). Кадр положительный, если любое
    решение по нему имеет decision=true (union-семантика, как в it-82).
  * Комплектность (гейт C1): множества кадров A и B совпадают и |A| ≥ expect-frames
    (иначе прогон не парный → вердикт не выносится, числа фиксируются).
  * Discordant-таблица: b = A+/B− (кадры, которые AND погасил), c = A−/B+ (которые AND
    включил — на бездронном материале это было бы усилением FP).
  * McNemar exact (двусторонний биномиальный на b+c) — основной тест сдвига направления;
    ДИ Уилсона для маргинальных FP приводятся как вторичные.
  * Cluster bootstrap по кадрам (10 000 повторов, семя фикс.) для Δ = FP(B) − FP(A) —
    согласованная оценка разности долей.
Артефакты: research/it84_paired.csv, research/it84_paired_summary.txt.
"""
import json
import math
import random
import sys

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"
N_BOOT = 10_000
SEED = 8404

argv = sys.argv[1:]
expect = 400
if "--expect-frames" in argv:
    i = argv.index("--expect-frames")
    expect = int(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
if "--jsonl" in argv:
    i = argv.index("--jsonl")
    JSONL = argv[i + 1]
    argv = argv[:i] + argv[i + 2:]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Двусторонний p-value: 2*P(X ≤ min(b,c)), X~Bin(b+c, 0.5), кап 1.0."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]


def frames_of(rng: str) -> dict[float, bool]:
    a, b = (int(x) for x in rng.split(":"))
    out: dict[float, bool] = {}
    for r in rows[a - 1 : b]:
        m = r.get("media_ts")
        if m is None:
            continue
        out[m] = out.get(m, False) or bool(r.get("decision"))
    return out


stages = {}
for spec in argv:
    name, rng = spec.split("=")
    stages[name] = frames_of(rng)
    print(f"{name}: кадров {len(stages[name])}, положит {sum(stages[name].values())}")

names = list(stages)
if len(names) < 2:
    sys.exit("нужно минимум 2 этапа (A и B)")
fa, fb = stages[names[0]], stages[names[1]]
common = sorted(set(fa) & set(fb))
only_a, only_b = set(fa) - set(fb), set(fb) - set(fa)
c1 = not only_a and not only_b and len(common) >= expect
print(f"C1 комплектность: |общих| {len(common)}, только в {names[0]} {len(only_a)}, "
      f"только в {names[1]} {len(only_b)} (порог {expect}) — {'🟢' if c1 else '🔴'}")

b_ = sum(1 for m in common if fa[m] and not fb[m])   # погашено ансамблем
c_ = sum(1 for m in common if not fa[m] and fb[m])   # добавлено ансамблем
k = sum(1 for m in common if fa[m] and fb[m])
n_ = sum(1 for m in common if not fa[m] and not fb[m])
fp_a = (k + b_) / len(common)
fp_b = (k + c_) / len(common)
p = mcnemar_exact(b_, c_)
print(f"discordant: a+ b− (AND погасил) {b_} | a− b+ (AND добавил) {c_} | оба+ {k} | оба− {n_}")
print(f"FP({names[0]}) {fp_a:.4f} ДИ {wilson(k + b_, len(common))} | "
      f"FP({names[1]}) {fp_b:.4f} ДИ {wilson(k + c_, len(common))} | Δ {fp_b - fp_a:+.4f}")

rng = random.Random(SEED)
diffs = []
n = len(common)
vals = [(1 if fa[m] else 0, 1 if fb[m] else 0) for m in common]
for _ in range(N_BOOT):
    sa = sb = 0
    for _i in range(n):
        x, y = vals[rng.randrange(n)]
        sa += x
        sb += y
    diffs.append((sb - sa) / n)
diffs.sort()
lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]
print(f"cluster-bootstrap Δ(B−A): [{lo:+.4f}; {hi:+.4f}] (10k, seed {SEED}); "
      f"McNemar exact p={p:.3g}")

with open(f"{ROOT}/research/it84_paired.csv", "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"n_common,{n}\nonly_a,{len(only_a)}\nonly_b,{len(only_b)}\n")
    f.write(f"b_rescued,{b_}\nc_added,{c_}\nboth_pos,{k}\nboth_neg,{n_}\n")
    f.write(f"fp_a,{fp_a:.6f}\nfp_b,{fp_b:.6f}\ndelta,{fp_b - fp_a:+.6f}\n")
    f.write(f"mcnemar_p,{p:.6g}\nboot_lo,{lo:.6f}\nboot_hi,{hi:.6f}\n")

with open(f"{ROOT}/research/it84_paired_summary.txt", "w", encoding="utf-8") as f:
    f.write("it-84 exact paired replay (strict400 HD OI, loop=false, fps=2, join=media_ts)\n\n")
    f.write(f"этапы: {names[0]} vs {names[1]}; C1: |общих| {n} (порог {expect}), "
            f"only_A {sorted(only_a)}, only_B {sorted(only_b)} — {'OK' if c1 else 'FAIL'}\n")
    f.write(f"discordant: b(A+/B−)={b_} c(A−/B+)={c_} оба+={k} оба−={n_}\n")
    f.write(f"FP(A)={fp_a:.4f} ДИ {wilson(k + b_, n)} | FP(B)={fp_b:.4f} ДИ {wilson(k + c_, n)} | Δ={fp_b - fp_a:+.4f}\n")
    f.write(f"P1 (b>c, McNemar p≤0,05, Δ<0): {'🟢' if (b_ > c_ and p <= 0.05 and fp_b < fp_a) else '🔴'} "
            f"(p={p:.3g})\n")
    f.write(f"P2 (знак b>c и FP(A)∈[0,5;1,0]): {'🟢' if (b_ > c_ and 0.5 <= fp_a <= 1.0) else '🔴'}\n")
    f.write(f"cluster-bootstrap Δ(B−A): [{lo:+.4f}; {hi:+.4f}] (10k, seed {SEED})\n")
