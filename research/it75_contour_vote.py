#!/usr/bin/env python3
"""it-75: контур на pv-рядах ансамбля (AND=min / OR=max) — ноль инференса.

Каркас it-71/it-74 (144 AST-окна, imgsz480 секундные ряды, гейт 0,25, D0).
Рядов ровно два; критерии K1–K3 и правило вердикта предрегистрированы в
iterations/it-75-contour-vote-rows-PLANNED.md (коммит 2429db9) — до запуска менять нельзя.

Запуск: research/.venv/bin/python research/it75_contour_vote.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = []


def p(line: str = "") -> None:
    print(line)
    LOG.append(line)


def ok(name: str, cond: bool) -> None:
    p(f"  [{'OK' if cond else 'ПРОВАЛ'}] {name}")
    assert cond, name


ast = {r["t0"]: r for r in csv.DictReader(open(ROOT / "research/ast_windows.csv"))}
WINS_T = sorted(ast.keys(), key=float)
GT = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(ROOT / "research/gt_sandbox_video.csv"))}
events, cur = [], None
for sec in sorted(GT):
    if GT[sec][1]:
        cur = [sec, sec] if cur is None else [cur[0], sec]
    elif cur is not None:
        events.append(tuple(cur)); cur = None
if cur is not None:
    events.append(tuple(cur))


def sec_row(yolo_csv: str) -> dict[str, float]:
    return {r["second"]: float(r["max_conf"])
            for r in csv.DictReader(open(ROOT / yolo_csv)) if r["imgsz"] == "480"}


old_s = sec_row("research/yolo_sandbox_frames.csv")
new_s = sec_row("research/yolo_sandbox_frames_new.csv")
assert set(old_s) == set(new_s)
rows = {"old": old_s, "new": new_s,
        "AND(min)": {k: min(old_s[k], new_s[k]) for k in old_s},
        "OR(max)": {k: max(old_s[k], new_s[k]) for k in old_s}}


def contour(rw: dict[str, float]):
    preds = []
    for t0 in WINS_T:
        sec = str(int(float(t0)))
        raw = rw[sec] if sec in rw else rw[str(int(float(t0)) + 1)]
        pv = raw if raw >= 0.25 else 0.0
        pa = float(ast[t0]["p_drone"])
        preds.append((float(t0), 0.5 * pv + 0.5 * pa >= 0.5, int(ast[t0]["airborne_gt"])))
    tp = sum(1 for _, pt, t in preds if pt and t)
    fp = sum(1 for _, pt, t in preds if pt and not t)
    fn = sum(1 for _, pt, t in preds if not pt and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    delays = []
    for es, ee in events:
        hits = [t for t, pt, _ in preds if pt and es <= t <= ee + 1.0]
        if hits:
            delays.append(min(hits) - es)
    m = dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else float("nan"), fp=fp, fn=fn,
             delay_med=sorted(delays)[len(delays) // 2] if delays else -1, n_delayed=len(delays))
    return preds, m


p(f"{'ряд':>9} {'P':>6} {'R':>6} {'F1':>6} {'FP':>3} {'FN':>3} {'medD':>5} {'nD':>3}")
res, mets = {}, {}
for tag, rw in rows.items():
    preds, m = contour(rw)
    res[tag], mets[tag] = preds, m
    p(f"{tag:>9} {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} {m['fp']:>3} {m['fn']:>3} {m['delay_med']:>5.1f} {m['n_delayed']:>3}")

p("\nсами-проверки (сверка с it74_vote_analysis.txt / it71):")
ok(f"old: F1={mets['old']['F1']:.3f} R={mets['old']['R']:.3f} FP={mets['old']['fp']} (0,961/0,991/8)",
   abs(mets["old"]["F1"] - 0.961) < 0.0015 and mets["old"]["fp"] == 8)
ok(f"new: F1={mets['new']['F1']:.3f} R={mets['new']['R']:.3f} FP={mets['new']['fp']} (0,946/0,955/7)",
   abs(mets["new"]["F1"] - 0.946) < 0.0015 and mets["new"]["fp"] == 7)
ok(f"AND: F1={mets['AND(min)']['F1']:.3f} R={mets['AND(min)']['R']:.3f} FP={mets['AND(min)']['fp']} (0,942/0,946/7)",
   abs(mets["AND(min)"]["F1"] - 0.942) < 0.0015 and abs(mets["AND(min)"]["R"] - 0.946) < 0.0015 and mets["AND(min)"]["fp"] == 7)

m = mets["OR(max)"]
k1, k2, k3 = m["R"] >= 0.98, m["F1"] >= 0.95, m["fp"] <= 8
p("\nВЕРДИКТ OR-ряд (K1–K3):")
p(f"  K1 R={m['R']:.3f} ≥ 0,98 → {'ЗЕЛЁНЫЙ' if k1 else 'красный'}")
p(f"  K2 F1={m['F1']:.3f} ≥ 0,95 → {'ЗЕЛЁНЫЙ' if k2 else 'красный'}")
p(f"  K3 FP={m['fp']} ≤ 8 → {'ЗЕЛЁНЫЙ' if k3 else 'красный'}")
fp_or = sorted(t for t, pt, gt in res["OR(max)"] if pt and not gt)
canon_fp = sorted(t for t, pt, gt in res["old"] if pt and not gt)
p(f"  FP-окна OR: {fp_or}")
p(f"  FP-окна канона old: {canon_fp}")
p(f"  прирост-FP относительно канона: {sorted(set(fp_or) - set(canon_fp))}")
p(f"  ИТОГ: {'ряд с max-conf держит тройку — контурная заявка пересматривается отдельно (риск насыщения it-73 помнить)' if k1 and k2 and k3 else 'АНСАМБЛЬ ЗАКРЫТ ДЛЯ КОНТУРА: контур остаётся на каноне old, ансамбль = только image-линейка (цена ×2 — решение автора)'}")

OUT = ROOT / "research/it75_contour_vote.txt"
with open(OUT, "w", encoding="utf-8") as f:
    f.write("it-75 контур на рядах ансамбля — срез stdout (протокол: iterations/it-75-contour-vote-rows-PLANNED.md)\n\n")
    f.write("\n".join(LOG) + "\n")
p(f"\nTXT: {OUT}")
