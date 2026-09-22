"""it-80: однородный HD-профиль ансамбля (R — MMAUD SAHI-HD; FP — 400 HD-фонов it-79).

Ноль инференса: только закоммиченные CSV (mmaud_sahi_full{,_new}.csv, it79_old/new.csv)
и GT mmaud/Mavic3. Выход: it80_hd_profile.{csv,txt}. Критерии H0–H2 предрегистрированы
(итерация it-80 PLANNED, коммит до замера).
"""
from __future__ import annotations

import bisect
import csv
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research"

TAUS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HT1_TAUS = [0.55, 0.60, 0.65, 0.70]
FP_BAR = 12.8   # %, порог E1 (зарегистрирован для COCO-фонов; см. ограничение 2)
R_BAR = 85.0    # %, полёт
H2_DBAR = 1.0   # п.п.

lines: list[str] = []
rows: list[dict] = []
def p(s: str = "") -> None:
    lines.append(s)

gt_files = sorted(glob.glob(str(ROOT / "MasterDiploma/train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt = [np.load(g) for g in gt_files]
assert len(gt) == len(gt_ts) > 0

def gt_at(t: float) -> np.ndarray:
    return gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))]

def load_sahi(path: Path) -> dict[str, float]:
    return {r["img"]: float(r["conf_sahi"]) for r in csv.DictReader(open(path, encoding="utf-8"))}

old_m = load_sahi(OUT / "mmaud_sahi_full.csv")
new_m = load_sahi(OUT / "mmaud_sahi_full_new.csv")
keys = sorted(set(old_m) & set(new_m))
n_join = len(keys)
flight, fard = [], []
for k in keys:
    x, y, z = gt_at(float(k))
    d = (old_m[k], new_m[k])
    if z > 1.0:
        flight.append(d)
        if 10 <= math.hypot(x, y) < 20:
            fard.append(d)

def load_bg(path: Path) -> dict[str, float]:
    return {r["img"]: float(r["max_conf"]) for r in csv.DictReader(open(path, encoding="utf-8"))}

old_b = load_bg(OUT / "it79_old.csv")
new_b = load_bg(OUT / "it79_new.csv")
bk = sorted(set(old_b) & set(new_b))

p("it-80: однородный HD-профиль ансамбля (оба ряда — режим SAHI-HD)")
p(f"H0-данные: MMAUD join {n_join}/5091 | полёт {len(flight)} (канон 5018) | "
  f"дальн10-19 {len(fard)} (канон 2351) | HD-фоны join {len(bk)}/400")

# ---- сами-проверки (блокатор интерпретации) ----
checks = []
r_old05 = sum(1 for a, _ in flight if a >= 0.5) / len(flight)
r_new05 = sum(1 for _, b in flight if b >= 0.5) / len(flight)
fp_and040 = sum(1 for k in bk if min(old_b[k], new_b[k]) >= 0.40) / len(bk) * 100
checks.append(("old@0,5 (MMAUD)", r_old05 * 100, 94.5, 0.2))
checks.append(("new@0,5 (MMAUD)", r_new05 * 100, 84.3, 0.2))
checks.append(("FP(AND@0,40; HD-фоны)", fp_and040, 48.8, 0.1))
h0 = True
p("\nсами-проверки:")
for name, got, want, tol in checks:
    okflag = abs(got - want) <= tol
    h0 &= okflag
    p(f"  {name}: {got:.1f} % (ожидалось {want}±{tol}) → {'OK' if okflag else 'НАРУШЕНА'}")
for name, n, want in (("join MMAUD", n_join, 5091), ("полёт n", len(flight), 5018),
                      ("дальн n", len(fard), 2351), ("join фонов", len(bk), 400)):
    okflag = n == want
    h0 &= okflag
    p(f"  {name}: {n} (ожидалось {want}) → {'OK' if okflag else 'НАРУШЕНА'}")
if not h0:
    p("\nВЕРДИКТ: H0 нарушена → замеры не интерпретируются.")
    (OUT / "it80_hd_profile.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    raise SystemExit(1)

# ---- H3: фронтир τ × ряды ----
def series(name: str):
    if name == "old":
        return lambda ab: ab[0], lambda k: old_b[k]
    if name == "new":
        return lambda ab: ab[1], lambda k: new_b[k]
    if name == "AND":
        return lambda ab: min(ab), lambda k: min(old_b[k], new_b[k])
    return lambda ab: max(ab), lambda k: max(old_b[k], new_b[k])

p("\nФронтир τ × ряд (R_полёт / R_дальн / FP_HD), %:")
for tau in TAUS:
    cells = {}
    for name in ("old", "new", "AND", "OR"):
        fr, fb = series(name)
        r_f = sum(1 for d in flight if fr(d) >= tau) / len(flight) * 100
        r_d = sum(1 for d in fard if fr(d) >= tau) / len(fard) * 100
        fpr = sum(1 for k in bk if fb(k) >= tau) / len(bk) * 100
        cells[name] = (r_f, r_d, fpr)
        rows.append(dict(tau=f"{tau:.2f}", row=name, r_flight=round(r_f, 1),
                         r_farnear=round(r_d, 1), fp_hd=round(fpr, 1)))
    line = f"  τ={tau:.2f}: " + " | ".join(
        f"{n} {cells[n][0]:5.1f}/{cells[n][2]:5.1f}" for n in ("old", "new", "AND", "OR"))
    p(line + "   (формат R_полёт/FP_HD; R_дальн — в CSV)")

# ---- H1/H2 ----
p("")
tau_star = None
for tau in HT1_TAUS:
    fr, fb = series("AND")
    r_f = sum(1 for d in flight if fr(d) >= tau) / len(flight) * 100
    fpr = sum(1 for k in bk if fb(k) >= tau) / len(bk) * 100
    p(f"  τ={tau:.2f}: FP(AND;HD)={fpr:.1f} % {'≤' if fpr <= FP_BAR else '>'} {FP_BAR} ∧ "
      f"R(AND)={r_f:.1f} % {'≥' if r_f >= R_BAR else '<'} {R_BAR}")
    if fpr <= FP_BAR and r_f >= R_BAR and tau_star is None:
        tau_star = tau
h1 = tau_star is not None
if not h1:
    lo = None
    for tau in HT1_TAUS:
        _, fb = series("AND")
        fpr = sum(1 for k in bk if fb(k) >= tau) / len(bk) * 100
        if fpr <= FP_BAR:
            lo = tau
            break
    tau_eval = lo
    verdict_tau = f"мин τ с FP≤{FP_BAR}: {lo}" if lo is not None else f"ни в одном τ∈{{0,55…0,70}} FP≤{FP_BAR} не достигнут"
else:
    tau_eval = tau_star
    verdict_tau = f"τ*={tau_star:.2f}"
p(f"\nH1 (главный): {'ЗЕЛЁНЫЙ' if h1 else 'КРАСНЫЙ'} — {verdict_tau}")

if tau_eval is None:
    p("H2: τ с FP(AND;HD)≤12,8 % отсутствует → сравнение ансамбль/одиночные на целевом FP не определяется; "
      "фиксируется максимум достижимого подавления FP на сетке.")
    _, fb = series("AND")
    best = min((sum(1 for k in bk if fb(k) >= t) / len(bk) * 100, t) for t in HT1_TAUS)
    p(f"  минимум FP(AND) на τ∈{{0,55…0,70}}: {best[0]:.1f} % при τ={best[1]:.2f}")
    h2 = None
else:
    frA, _ = series("AND")
    rA = sum(1 for d in flight if frA(d) >= tau_eval) / len(flight) * 100
    rO = sum(1 for a, _ in flight if a >= tau_eval) / len(flight) * 100
    rN = sum(1 for _, b in flight if b >= tau_eval) / len(flight) * 100
    d_r = rA - max(rO, rN)
    h2 = d_r >= H2_DBAR
    p(f"H2: при τ={tau_eval:.2f}: R(AND)={rA:.1f} vs max(R(old),R(new))={max(rO, rN):.1f} → "
      f"ΔR={d_r:+.1f} п.п. {'≥' if h2 else '<'} +{H2_DBAR} → {'ЗЕЛЁНЫЙ' if h2 else 'КРАСНЫЙ'}")

p("")
if h1 and h2:
    p("ВЕРДИКТ: на HD есть однородная рабочая точка ансамбля (τ* из H1) с сохранённым "
      "преимуществом AND → зелёная пара it-74 переносится в HD-режим со сдвигом порога.")
elif h1:
    p("ВЕРДИКТ: однородная HD-точка существует (H1 зелёный), но выигрыш AND над лучшим "
      "одиночным рядом при ней не подтверждён (H2 красный) → на HD ансамбль не лучше "
      "одиночной модели с тем же FP: ограничение профиля (в SYNTHESIS).")
else:
    p("ВЕРДИКТ: в пределах сетки τ≤0,70 однородная HD-точка в барах E5×E1 не достигается "
      "(H1 красный) → профиль ансамбля 94,7/7,2 — свойство смешанного/640-режима; в HD "
      "требуется либо τ выше сетки (с неконтролируемым падением R), либо новая калибровка "
      "ряда (вне безобученного фронта). Регистрация ограничения в SYNTHESIS.")
txt = "\n".join(lines) + "\n"
(OUT / "it80_hd_profile.txt").write_text(txt, encoding="utf-8")
with open(OUT / "it80_hd_profile.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["tau", "row", "r_flight", "r_farnear", "fp_hd"])
    w.writeheader()
    w.writerows(rows)
print(txt)
