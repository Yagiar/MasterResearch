#!/usr/bin/env python3
"""it-89 — zero-inference disagreement-гейт для skyguard как третьего voter.

Протокол: iterations/it-89-skyguard-third-voter-PLANNED.md (предрег. до расчётов).
Политики на τ: AND2 = min(old,new)≥τ; MAJ3 = ≥2 из трёх ≥τ; AND3 = все три ≥τ;
TIE = на D (old≠new по τ) голос skyguard, вне D — общий голос old=new.
Множества: flight (MMAUD, GT z>1,0) и фоны (COCO-400, все негативны).
Запуск: research/.venv/bin/python research/it89_skyguard_gate.py
Артефакты: research/it89_skyguard_gate.csv, research/it89_skyguard_gate.txt.
"""
import bisect
import csv
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAUS = (0.35, 0.40, 0.45, 0.50)
LOG = []


def p(line=""):
    print(line)
    LOG.append(line)


gt_files = sorted(glob.glob(str(ROOT / "MasterDiploma/train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt = [np.load(g) for g in gt_files]


def load_sahi(path):
    return {r["img"]: float(r["conf_sahi"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def load_bg(path):
    return {r["img"]: float(r["max_conf"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


old_m, new_m, sky_m = (load_sahi(ROOT / f"research/{f}") for f in
                       ("mmaud_sahi_full.csv", "mmaud_sahi_full_new.csv", "mmaud_sahi_full_skyguard-v11.csv"))
keys = sorted(set(old_m) & set(new_m) & set(sky_m))
p(f"Q0 join MMAUD: per {len(old_m)}/{len(new_m)}/{len(sky_m)}, общих {len(keys)} (ожидаем 5091)")
def gt_at(t):
    return gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))]


flight = [k for k in keys if gt_at(float(k))[2] > 1.0]
old_b, new_b, sky_b = (load_bg(ROOT / f"research/{f}") for f in
                       ("coco_bg_fp_it69v1-old.csv", "coco_bg_fp_it69v1-new.csv", "coco_bg_fp_skyguard-v11.csv"))
bk = sorted(set(old_b) & set(new_b) & set(sky_b))
p(f"Q0 join фонов: per {len(old_b)}/{len(new_b)}/{len(sky_b)}, общих {len(bk)} (ожидаем 400)")

r_old = sum(1 for k in flight if old_m[k] >= 0.5) / len(flight)
r_new = sum(1 for k in flight if new_m[k] >= 0.5) / len(flight)
r_sky = sum(1 for k in flight if sky_m[k] >= 0.5) / len(flight)
q0 = (len(keys) == 5091 and len(bk) == 400
      and abs(r_old - 0.945) < 0.002 and abs(r_new - 0.843) < 0.002 and abs(r_sky - 0.909) < 0.002)
p(f"Q0 recall@0,5: old {r_old:.1%} (канон 94,5), new {r_new:.1%} (84,3), skyguard {r_sky:.1%} (90,9) → {'🟢' if q0 else '🔴'}")

# вложенности политик на множествах (τ=0,4)
tau = 0.40
A2 = {k for k in flight if min(old_m[k], new_m[k]) >= tau}
M3 = {k for k in flight if sum(c >= tau for c in (old_m[k], new_m[k], sky_m[k])) >= 2}
A3 = {k for k in flight if min(old_m[k], new_m[k], sky_m[k]) >= tau}
nest = A3 <= A2 <= M3
p(f"Q0 вложенность flight@0,4: AND3⊆AND2⊆MAJ3 → {'🟢' if nest else '🔴'} (|A3|={len(A3)}, |A2|={len(A2)}, |M3|={len(M3)})")

rows = []
q1_hit = None
p("\nTIE (разрез расхождений old≠new голосом skyguard) против AND2 на той же τ:")
for tau in TAUS:
    d_f = [k for k in flight if (old_m[k] >= tau) != (new_m[k] >= tau)]
    d_b = [k for k in bk if (old_b[k] >= tau) != (new_b[k] >= tau)]
    a2_f = {k for k in flight if min(old_m[k], new_m[k]) >= tau}
    a2_b = {k for k in bk if min(old_b[k], new_b[k]) >= tau}
    tie_f = a2_f | {k for k in d_f if sky_m[k] >= tau}   # вне D old=new: 1 ⟺ оба ≥τ (= AND2 на них); в D — sky
    tie_b = a2_b | {k for k in d_b if sky_b[k] >= tau}
    resc = len(tie_f) - len(a2_f)
    intr = len(tie_b) - len(a2_b)
    prec = resc / (resc + intr) if resc + intr else float("nan")
    hit = resc >= 5 and prec >= 0.90 and intr <= 2
    if hit and q1_hit is None:
        q1_hit = tau
    rows.append(dict(block="tie", tau=tau, D_f=len(d_f), D_b=len(d_b), a2_f=len(a2_f), tie_f=len(tie_f),
                     rescued=resc, a2_b=len(a2_b), tie_b=len(tie_b), introduced=intr,
                     prec=round(prec, 4) if prec == prec else "nan", hit=int(hit)))
    p(f"  τ={tau:.2f}: |D_полёт|={len(d_f)} |D_фоны|={len(d_b)} | AND2 f/b={len(a2_f)}/{len(a2_b)} → "
      f"TIE f/b={len(tie_f)}/{len(tie_b)} | rescued={resc} introduced={intr} prec={prec:.2f} "
      f"{'← Q1 🟢' if hit else ''}")

p("\nMAJ3 против AND2 (действующая AND_presence_live@0,4):")
maj_rows = []
for tau in TAUS:
    a2_f = sum(1 for k in flight if min(old_m[k], new_m[k]) >= tau) / len(flight)
    a2_b = sum(1 for k in bk if min(old_b[k], new_b[k]) >= tau) / len(bk)
    m3_f = sum(1 for k in flight if sum(c >= tau for c in (old_m[k], new_m[k], sky_m[k])) >= 2) / len(flight)
    m3_b = sum(1 for k in bk if sum(c >= tau for c in (old_b[k], new_b[k], sky_b[k])) >= 2) / len(bk)
    dr, dfp = (m3_f - a2_f) * 100, (m3_b - a2_b) * 100
    ok2 = dr >= 1.0 and dfp <= 1.0
    maj_rows.append(ok2)
    rows.append(dict(block="maj3", tau=tau, a2_f=round(a2_f, 4), m3_f=round(m3_f, 4), dR_pp=round(dr, 2),
                     a2_b=round(a2_b, 4), m3_b=round(m3_b, 4), dFP_pp=round(dfp, 2), hit=int(ok2)))
    p(f"  τ={tau:.2f}: R {a2_f:.1%}→{m3_f:.1%} (Δ{dr:+.1f} п.п.) | FP {a2_b:.1%}→{m3_b:.1%} (Δ{dfp:+.1f} п.п.) "
      f"{'← Q2 🟢' if ok2 else ''}")

q1 = q1_hit is not None
q2 = any(maj_rows)
p(f"\nВЕРДИКТ-гейт: Q1 {'🟢 (τ=' + str(q1_hit) + ')' if q1 else '🔴'} | Q2 {'🟢' if q2 else '🔴'}")
p("ИТОГ: " + ("Q1∨Q2 — третий voter имеет комплементарный сигнал; планировать GPU-прогон отдельной итерацией"
              if (q1 or q2) else
              "¬Q1∧¬Q2 — комплементарного сигнала нет; it-89 закрывается отрицательно БЕЗ GPU-прогона (стоп-критерий роадмапа)"))

with open(ROOT / "research/it89_skyguard_gate.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
    wr.writeheader()
    for r in rows:
        wr.writerow({k: r.get(k, "") for k in wr.fieldnames})
with open(ROOT / "research/it89_skyguard_gate.txt", "w", encoding="utf-8") as f:
    f.write("it-89 skyguard third-voter disagreement gate — срез stdout (предрег. it-89-skyguard-third-voter-PLANNED.md)\n\n")
    f.write("\n".join(LOG) + "\n")
