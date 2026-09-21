#!/usr/bin/env python3
"""it-74: голосование двух моделей (old⊕new) по conf — ноль инференса, чистый join CSV.

Правила (pre-defined): AND_τ := min(c_old,c_new) ≥ τ; OR_τ := max ≥ τ, τ∈{0,35;0,40;0,45;0,50}.
Основная ячейка — AND@0,40; критерии G1–G4 предрегистрированы в
iterations/it-74-two-model-vote-PLANNED.md (коммит 3743ed6) — до запуска менять нельзя.

Домены: полёт/дальн10-19 (MMAUD SAHI conf, сегментация как it65_union_analysis.py),
фоны (400 независимых COCO-кадров it-69 V1, full-frame max_conf),
контур (pv := min(old,new) секундный ряд imgsz480, каркас it-71, гейт 0,25, D0).

Запуск: research/.venv/bin/python research/it74_vote_analysis.py
"""
import bisect
import csv
import glob
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAUS = (0.35, 0.40, 0.45, 0.50)
rows_csv = []
LOG = []


def p(line: str = "") -> None:
    print(line)
    LOG.append(line)


def ok(name: str, cond: bool) -> None:
    p(f"  [{'OK' if cond else 'ПРОВАЛ'}] {name}")
    assert cond, name


# ---------- MMAUD: полёт / дальн10-19 ----------
gt_files = sorted(glob.glob(str(ROOT / "MasterDiploma/train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in gt_files]
gt = [np.load(g) for g in gt_files]


def gt_at(t: float) -> np.ndarray:
    return gt[min(len(gt) - 1, bisect.bisect_left(gt_ts, t))]


def load_sahi(path: Path) -> dict[str, float]:
    return {r["img"]: float(r["conf_sahi"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


old_m = load_sahi(ROOT / "research/mmaud_sahi_full.csv")
new_m = load_sahi(ROOT / "research/mmaud_sahi_full_new.csv")
keys = sorted(set(old_m) & set(new_m))
assert len(keys) == 5091 == len(old_m) == len(new_m), f"join MMAUD: {len(keys)}/{len(old_m)}/{len(new_m)}"
flight, fard = [], []
for k in keys:
    x, y, z = gt_at(float(k))
    d = (old_m[k], new_m[k])
    if z > 1.0:
        flight.append(d)
        if 10 <= math.hypot(x, y) < 20:
            fard.append(d)

p(f"MMAUD join: {len(keys)} из 5091 | полёт {len(flight)} | дальн10-19 {len(fard)}")
p("\nG1/G3 — recall (полёт; дальн10-19 справочно):")
rec = {}
for tag, rule in (("AND", min), ("OR", max)):
    for tau in TAUS:
        r_f = sum(1 for a, b in flight if rule(a, b) >= tau) / len(flight)
        r_d = sum(1 for a, b in fard if rule(a, b) >= tau) / len(fard)
        rec[(tag, tau)] = (r_f, r_d)
        p(f"  {tag}@{tau:.2f}: полёт={r_f:.1%}  дальн10-19={r_d:.1%}")
        rows_csv.append(dict(domain="mmaud", rule=tag, tau=tau, flight=round(r_f, 4), farnear=round(r_d, 4), n=len(flight)))
r_old = sum(1 for a, _ in flight if a >= 0.5) / len(flight)
r_new = sum(1 for _, b in flight if b >= 0.5) / len(flight)
p("\nсами-проверки MMAUD:")
ok(f"old@0,5 = {r_old:.1%} (канон 94,5 %)", abs(r_old - 0.945) < 0.002)
ok(f"new@0,5 = {r_new:.1%} (канон 84,3 %)", abs(r_new - 0.843) < 0.002)

# ---------- Фоны: FP на 400 независимых ----------
def load_bg(path: Path) -> dict[str, float]:
    return {r["img"]: float(r["max_conf"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


old_b = load_bg(ROOT / "research/coco_bg_fp_it69v1-old.csv")
new_b = load_bg(ROOT / "research/coco_bg_fp_it69v1-new.csv")
bk = sorted(set(old_b) & set(new_b))
assert len(bk) == 400, f"join фонов: {len(bk)}"
p(f"\nG2 — FP на {len(bk)} независимых фон:")
fp = {}
for tag, rule in (("AND", min), ("OR", max)):
    for tau in TAUS:
        f = sum(1 for k in bk if rule(old_b[k], new_b[k]) >= tau) / len(bk)
        fp[(tag, tau)] = f
        p(f"  {tag}@{tau:.2f}: FP={f:.1%}")
        rows_csv.append(dict(domain="bg400", rule=tag, tau=tau, fp=round(f, 4), n=len(bk)))
p("\nсами-проверки фоны:")
ok(f"old@0,5 = {sum(1 for k in bk if old_b[k] >= 0.5) / 400:.1%} (канон 27,5 %)",
   abs(sum(1 for k in bk if old_b[k] >= 0.5) / 400 - 0.275) < 0.002)
ok(f"new@0,5 = {sum(1 for k in bk if new_b[k] >= 0.5) / 400:.1%} (канон 6,5 %)",
   abs(sum(1 for k in bk if new_b[k] >= 0.5) / 400 - 0.065) < 0.002)

# ---------- Контур: каркас it-71, pv := min(old,new) ----------
ast = {r["t0"]: r for r in csv.DictReader(open(ROOT / "research/ast_windows.csv"))}
WINS_T = sorted(ast.keys(), key=float)


def sec_row(yolo_csv: str) -> dict[str, float]:
    return {r["second"]: float(r["max_conf"])
            for r in csv.DictReader(open(ROOT / yolo_csv)) if r["imgsz"] == "480"}


old_s = sec_row("research/yolo_sandbox_frames.csv")
new_s = sec_row("research/yolo_sandbox_frames_new.csv")
assert set(old_s) == set(new_s), "секундные ряды old/new не совпадают по ключам"
and_s = {k: min(old_s[k], new_s[k]) for k in old_s}


def contour(rows: dict[str, float]) -> list[tuple[float, bool, int]]:
    preds = []
    for t0 in WINS_T:
        sec = str(int(float(t0)))
        raw = rows[sec] if sec in rows else rows[str(int(float(t0)) + 1)]
        pv = raw if raw >= 0.25 else 0.0
        pa = float(ast[t0]["p_drone"])
        preds.append((float(t0), 0.5 * pv + 0.5 * pa >= 0.5, int(ast[t0]["airborne_gt"])))
    return preds


def metrics(preds):
    tp = sum(1 for _, pt, t in preds if pt and t)
    fp_ = sum(1 for _, pt, t in preds if pt and not t)
    fn = sum(1 for _, pt, t in preds if not pt and t)
    P = tp / (tp + fp_) if tp + fp_ else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else float("nan"), fp=fp_, fn=fn)


m_old = metrics(contour(old_s))
m_new = metrics(contour(new_s))
m_and = metrics(contour(and_s))
p("\nG4 — контур (гейт 0,25, D0):")
for tag, m in (("old (самопроверка 0,961/0,991/FP8)", m_old),
               ("new (самопроверка 0,946/0,955/FP7)", m_new),
               ("AND-ряд", m_and)):
    p(f"  {tag:>34}: P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} FP={m['fp']} FN={m['fn']}")
    rows_csv.append(dict(domain="contour", rule=tag.split()[0], tau="",
                         **{k: round(v, 3) for k, v in m.items()}))
ok(f"канон old: F1={m_old['F1']:.3f} FP={m_old['fp']}", abs(m_old["F1"] - 0.961) < 0.0015 and m_old["fp"] == 8)
ok(f"канон new: F1={m_new['F1']:.3f} FP={m_new['fp']}", abs(m_new["F1"] - 0.946) < 0.0015 and m_new["fp"] == 7)

# ---------- Вердикт основной ячейки AND@0,40 ----------
g1 = rec[("AND", 0.40)][0] >= 0.895
g2 = fp[("AND", 0.40)] <= 0.128
g3 = rec[("AND", 0.40)][1] >= 0.804
g4 = m_and["R"] >= 0.98 and m_and["F1"] >= 0.95 and m_and["fp"] <= 8
p("\nВЕРДИКТ AND@0,40:")
p(f"  G1 полёт {rec[('AND', 0.40)][0]:.1%} ≥ 89,5 % → {'ЗЕЛЁНЫЙ' if g1 else 'красный'}")
p(f"  G2 FP    {fp[('AND', 0.40)]:.1%} ≤ 12,8 %  → {'ЗЕЛЁНЫЙ' if g2 else 'красный'}")
p(f"  G3 дальн {rec[('AND', 0.40)][1]:.1%} ≥ 80,4 % (отчётный) → {'зелёный' if g3 else 'красный'}")
p(f"  G4 контур R={m_and['R']:.3f} F1={m_and['F1']:.3f} FP={m_and['fp']} → {'ЗЕЛЁНЫЙ' if g4 else 'красный'}")
p(f"  ИТОГ: {'кандидат офлайн-линейки (G1∧G2)' if g1 and g2 else 'ГОЛОСОВАНИЕ ЗАКРЫТО'}; G4 {'даёт' if g4 else 'НЕ даёт'} контурную заявку")

OUT_TXT = ROOT / "research/it74_vote_analysis.txt"
OUT_CSV = ROOT / "research/it74_vote_grid.csv"
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("it-74 vote analysis — срез stdout (протокол: iterations/it-74-two-model-vote-PLANNED.md)\n\n")
    f.write("\n".join(LOG) + "\n")
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["domain", "rule", "tau", "flight", "farnear", "fp", "n", "P", "R", "F1", "fn"])
    w.writeheader()
    w.writerows(rows_csv)
p(f"\nCSV: {OUT_CSV} ({len(rows_csv)} строк)")
