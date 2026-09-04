#!/usr/bin/env python3
"""Офлайн-анализ decisions.jsonl (артефакт sink) — без ground truth.

Разрез «сегмент (прогон) × режим fusion»: доля TRUE, модальность окон,
согласованность каналов, распределения уверенностей, латентность, поля gating.
Сегменты выделяются по дырам > 120 с в отсортированном по ts потоке.
Запуск:
    python scripts/analyze_decisions.py [path/to/decisions.jsonl]
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from collections import Counter, defaultdict
from statistics import mean, median, quantiles

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/decisions/decisions.jsonl"
GAP_S = 120.0


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    qs = quantiles(vals, n=100, method="inclusive")
    return qs[max(0, min(99, int(round(p)) - 1))]


def hist(vals: list[float], bins: int = 10) -> str:
    c = Counter(min(int(v * bins), bins - 1) for v in vals if v is not None)
    total = sum(c.values()) or 1
    return "".join(str(c.get(i, 0) * 10 // total) for i in range(bins))


rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: r["ts"])
runs: list[list[dict]] = []
for r in rows:
    if runs and r["ts"] - runs[-1][-1]["ts"] <= GAP_S:
        runs[-1].append(r)
    else:
        runs.append([r])

print(f"всего решений: {len(rows)}; сегментов-прогонов: {len(runs)}\n")
gating_filled = Counter()

for i, run in enumerate(runs, 1):
    t0 = dt.datetime.fromtimestamp(run[0]["ts"]).strftime("%d.%m %H:%M")
    t1 = dt.datetime.fromtimestamp(run[-1]["ts"]).strftime("%H:%M")
    print(f"### Сегмент #{i}: {t0}..{t1}  ({run[-1]['ts'] - run[0]['ts']:.0f} c, n={len(run)})")
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in run:
        by_mode[r["mode"]].append(r)
        g = r.get("gating") or {}
        for k, v in g.items():
            if v is not None:
                gating_filled[k] += 1

    hdr = ("  mode          n     TRUE  ratio  p_f avg/med    mixed  agree/contra  "
           "conf_v  conf_a  e2e p50/p95, ms     p_a hist(10 бинов)")
    print(hdr)
    for mode, rs in sorted(by_mode.items()):
        n = len(rs)
        true_n = sum(1 for r in rs if r["decision"])
        p_f = [r["p_fused"] for r in rs]
        e2e = [r["e2e_latency_ms"] for r in rs]
        mixed = [r for r in rs if r["contributions"]["p_v"] is not None
                 and r["contributions"]["p_a"] is not None]
        pv = [r["contributions"]["p_v"] for r in rs if r["contributions"]["p_v"] is not None]
        pa = [r["contributions"]["p_a"] for r in rs if r["contributions"]["p_a"] is not None]
        agree = sum(1 for r in mixed if (r["contributions"]["p_v"] >= 0.5) == (r["contributions"]["p_a"] >= 0.5))
        print(f"  {mode:<13} {n:<5} {true_n:<5} {true_n / n:.3f}  {mean(p_f):.3f}/{median(p_f):.3f}   "
              f"{len(mixed):<6} {agree}/{len(mixed) - agree:<6}        "
              f"{mean(pv) if pv else float('nan'):.3f}   {mean(pa) if pa else float('nan'):.3f}   "
              f"{median(e2e):>5.0f}/{pct(e2e, 95):>6.0f}    [{hist(pa)}]")
    print()

print(f"заполненность полей gating (не-null): {dict(gating_filled) or '—'}")

# e2e-дрейф внутри самого длинного мультимодального сегмента (bug sink/PostgreSQL)
big = max(runs, key=len)
q = max(1, len(big) // 10)
dec = [median([r["e2e_latency_ms"] for r in big[k * q:(k + 1) * q]]) for k in range(10)]
print(f"e2e-дрейф (медианы по децилям самого большого сегмента, мс): "
      f"{' → '.join(f'{v:.0f}' for v in dec)}")
