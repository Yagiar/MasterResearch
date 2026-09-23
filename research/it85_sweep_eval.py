#!/usr/bin/env python3
"""it-85 — разбор throughput sweep: устойчивая входная частота по предрег. определению.

Использование:
  research/.venv/bin/python research/it85_sweep_eval.py A@2=START:END B@2=... [...]
    (срезы — 1-based строки decisions.jsonl из «SCORE_OFF» лога it85_sweep_run)

Метрики на этап (предрегистр, it-85-throughput-saturation-PLANNED.md):
  coverage = уникальных media_ts / 400; lag = (ts − t0) − media_ts, t0 = min(ts − media_ts);
  p50/p95/p99 лага; send_done = t0 + 400/fps; drain_s = от send_done до первого сэмпла
  (~15 с тик) с vd+fusion лагом = 0 (не нашли до конца окна этапа → цензура = провал);
  max lag за стрим; GPU util mean/p95; VRAM peak — из research/it85_samples.csv.
sustainable := drain ≤ 60 с И p95 ≤ 10 с И coverage ≥ 0,45.
Проверки: C1 coverage(A@2)∈[0,45;0,60]; P0 sustainable(A@0,5); P1 все A sustainable;
P2 sustainable_fps(B) ≤ sustainable_fps(A) и Δp95(B−A) ≥ 0 на общих точках.
Артефакты: research/it85_sweep.csv, research/it85_sweep_summary.txt.
"""
import csv
import json
import math
import sys
from pathlib import Path

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"
SAMPLES = f"{ROOT}/research/it85_samples.csv"
OUT_CSV = f"{ROOT}/research/it85_sweep.csv"
OUT_TXT = f"{ROOT}/research/it85_sweep_summary.txt"
# опциональные пути только для синтетической самопроверки парсера (дефолт — боевые артефакты)
_args = [a for a in sys.argv[1:] if a.startswith("--")]
for _a in _args:
    k, v = _a.split("=", 1)
    if k == "--jsonl":
        JSONL = v
    elif k == "--samples":
        SAMPLES = v
    elif k == "--out":
        OUT_CSV = v + ".csv"
        OUT_TXT = v + ".txt"
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith("--")]
N_FRAMES = 400
DRAIN_MAX = 60.0
P95_MAX = 10.0
COV_MIN = 0.45


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
with open(SAMPLES, encoding="utf-8") as f:
    rd = list(csv.reader(l for l in f if l.strip() and not l.startswith("epoch")))
samples = [(int(r[0]), r[1], int(r[2]), int(r[3]), float(r[4]), float(r[5])) for r in rd if len(r) >= 6]

out = []
for spec in sys.argv[1:]:
    name, rng = spec.split("=")
    a, b = (int(x) for x in rng.split(":"))
    part = rows[a - 1 : b]
    if not part:
        print(f"{name}: ПУСТОЙ СРЕЗ {rng}")
        continue
    fps = float(name.split("@")[1].replace(",", "."))
    recs = [(r["media_ts"], r["ts"]) for r in part if r.get("media_ts") is not None]
    t0 = min(ts - m for m, ts in recs)
    first_ts = min(ts for _, ts in recs)
    last_ts = max(ts for _, ts in recs)
    frames = {}
    for m, ts in recs:
        frames[m] = min(frames.get(m, ts), ts)
    lags = [ts - t0 - m for m, ts in frames.items()]
    cov = len(frames) / N_FRAMES
    p50, p95, p99 = pct(lags, 0.50), pct(lags, 0.95), pct(lags, 0.99)
    send_done = t0 + N_FRAMES / fps
    smin, smax = first_ts - 120, last_ts + 90
    st = [s for s in samples if s[1] == name and smin <= s[0] <= smax]
    drain = float("nan")
    maxlag = 0
    util = []
    vmem = 0.0
    for ep, _, vd, fu, gu, gm in st:
        maxlag = max(maxlag, vd + fu) if ep <= send_done else maxlag
        util.append(gu)
        vmem = max(vmem, gm)
        if math.isnan(drain) and ep >= send_done and vd + fu == 0:
            drain = ep - send_done
    sust = (not math.isnan(drain)) and drain <= DRAIN_MAX and p95 <= P95_MAX and cov >= COV_MIN
    out.append((name, fps, len(frames), cov, p50, p95, p99, drain, maxlag,
                sum(util) / len(util) if util else float("nan"), pct(util, 0.95), vmem, sust, len(st)))
    print(f"{name}: cov {cov:.3f} ({len(frames)}) | лаг p50 {p50:.1f} p95 {p95:.1f} p99 {p99:.1f} с | "
          f"drain {drain:.0f} с | maxlag {maxlag} | GPU {out[-1][9]:.0f}%/{out[-1][10]:.0f}% p95 | "
          f"VRAM {vmem:.0f} MiB | семплов {len(st)} | {'sustainable 🟢' if sust else 'НЕ sustainable 🔴'}")

def fps_of(name):
    return float(name.split("@")[1].replace(",", "."))


def arm(name):
    return name.split("@")[0]


def sus_fps(a):
    vals = [fps_of(o[0]) for o in out if arm(o[0]) == a and o[12]]
    return max(vals) if vals else float("nan")


got = {arm(o[0]): {} for o in out}
for o in out:
    got[arm(o[0])][fps_of(o[0])] = o

if "A" in got and 2.0 in got["A"]:
    c = got["A"][2.0]
    print(f"C1 покрытие A@2 ∈ [0,45;0,60]: {'🟢' if 0.45 <= c[3] <= 0.60 else '🔴'} ({c[3]:.3f})")
p0 = got.get("A", {}).get(0.5)
print(f"P0 A@0,5 sustainable: {'🟢' if p0 and p0[12] else '🔴'}")
a_stages = [o for o in out if arm(o[0]) == "A"]
p1 = bool(a_stages) and all(o[12] for o in a_stages)
sa, sb = sus_fps("A"), sus_fps("B")
print(f"P1 все A sustainable: {'🟢' if p1 else '🔴'} (sustainable_fps(A) = {sa}; "
      f"провалившие точки: {[o[0] for o in a_stages if not o[12]] or 'нет'})")
dp = []
for f in sorted(set(got.get("A", {})) & set(got.get("B", {}))):
    dp.append((f, got["B"][f][5] - got["A"][f][5]))
p2 = (not math.isnan(sa) and not math.isnan(sb) and sb <= sa) and all(d >= -0.5 for _, d in dp)
print(f"P2 sustainable_fps(B)≤(A): {sb} ≤ {sa}; Δp95(B−A) по точкам: "
      f"{[(f'{f:g}', round(d, 2)) for f, d in dp]} → {'🟢' if p2 else '🔴'}")

def fmt(x):
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.4f}"
    return str(x)


with open(OUT_CSV, "w", encoding="utf-8") as fh:
    fh.write("stage,fps,n_frames_covered,coverage,lag_p50,lag_p95,lag_p99,drain_s,max_backlog,"
             "gpu_util_mean,gpu_util_p95,vram_peak_mib,sustainable,n_samples\n")
    for o in out:
        fh.write(",".join(fmt(x) for x in o) + "\n")

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("it-85 throughput saturation sweep (strict400 OI, loop=false, 400 кадров/этап, "
             "post-fix media_ts; sustainable := drain≤60с ∧ p95≤10с ∧ coverage≥0,45)\n")
    for o in out:
        fh.write(f"{o[0]}: cov={o[3]:.3f} p50={o[4]:.1f} p95={o[5]:.1f} p99={o[6]:.1f} "
                 f"drain={o[7]:.0f} maxlag={o[8]} gpu={o[9]:.0f}/{o[10]:.0f} vram={o[11]:.0f} "
                 f"sust={'да' if o[12] else 'нет'}\n")
    fh.write(f"\nsustainable_fps(A)={sus_fps('A')}, sustainable_fps(B)={sus_fps('B')}\n")
    fh.write("Оговорки: стрим конечный (400 кадров) — потолок оценивается по drain/лагам внутри этапа; "
             "ingest и детектор не разделены в backlog-метрике (video.raw лаг суммарный).\n")
