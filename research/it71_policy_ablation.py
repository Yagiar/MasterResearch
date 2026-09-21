#!/usr/bin/env python3
"""it-71: аблация решёточной политики fusion под новые веса — НОЛЬ инференса.

Каркас окон как в fusion_sim_full.py (it-36): 144 окна AST, pv = YOLO max_conf секунды
(с детекторным гейтом g: max_conf < g → 0), pa = AST p_drone, GT airborne majority.
Сетка и правило вердикта предрегистрированы в
research/iterations/it-71-fusion-decision-policy-ablation.md — до запуска менять нельзя.

Запуск: research/.venv/bin/python research/it71_policy_ablation.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ast = {r["t0"]: r for r in csv.DictReader(open(ROOT / "research/ast_windows.csv"))}
WINS_T = sorted(ast.keys(), key=float)

GT = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(ROOT / "research/gt_sandbox_video.csv"))}
# GT-события airborne: непрерывные участки секунд
events, cur = [], None
for sec in sorted(GT):
    if GT[sec][1]:
        cur = [sec, sec] if cur is None else [cur[0], sec]
    elif cur is not None:
        events.append(tuple(cur)); cur = None
if cur is not None:
    events.append(tuple(cur))


def load_windows(yolo_csv: str):
    yolo = {r["second"]: r for r in csv.DictReader(open(ROOT / yolo_csv)) if r["imgsz"] == "480"}
    wins = []
    for t0 in WINS_T:
        sec = int(float(t0))
        raw = float(yolo[str(sec)]["max_conf"]) if str(sec) in yolo else float(yolo[str(sec + 1)]["max_conf"])
        wins.append((float(t0), raw, float(ast[t0]["p_drone"]), int(ast[t0]["airborne_gt"])))
    return wins


def causal_median(vals, k):
    out = []
    for i in range(len(vals)):
        seg = sorted(vals[max(0, i - k + 1): i + 1])
        out.append(seg[len(seg) // 2])
    return out


def decide(pv, pa, D):
    if D == "D0 late τ0.5":
        return 0.5 * pv + 0.5 * pa >= 0.5
    if D == "D1 late τ0.45":
        return 0.5 * pv + 0.5 * pa >= 0.45
    if D == "D2 late τ0.40":
        return 0.5 * pv + 0.5 * pa >= 0.40
    if D == "D3 max(pv,pa)≥0.5":
        return max(pv, pa) >= 0.5
    if D == "D4 pv≥0.4 | pa≥0.6":
        return pv >= 0.4 or pa >= 0.6
    raise ValueError(D)


POLICIES = ["D0 late τ0.5", "D1 late τ0.45", "D2 late τ0.40",
            "D3 max(pv,pa)≥0.5", "D4 pv≥0.4 | pa≥0.6", "D5 med3+D0"]

print(f"GT-события airborne: {events}")
print(f"{'веса':>5} {'g':>5} {'политика':<18} {'P':>6} {'R':>6} {'F1':>6} {'FP':>3} {'FN':>3}  задержка, с")
rows = []
for wtag, csv_path in (("old", "research/yolo_sandbox_frames.csv"),
                       ("new", "research/yolo_sandbox_frames_new.csv")):
    for g in (0.25, 0.4):
        wins = load_windows(csv_path)
        pv_g = [w[1] if w[1] >= g else 0.0 for w in wins]
        pv_med3 = causal_median(pv_g, 3)
        for D in POLICIES:
            preds = []
            for i, (t0, _, pa, gt) in enumerate(wins):
                pv = pv_med3[i] if D == "D5 med3+D0" else pv_g[i]
                p = (0.5 * pv + 0.5 * pa >= 0.5) if D == "D5 med3+D0" else decide(pv, pa, D)
                preds.append((t0, p, gt))
            tp = sum(1 for _, p, t in preds if p and t)
            fp = sum(1 for _, p, t in preds if p and not t)
            fn = sum(1 for _, p, t in preds if not p and t)
            P = tp / (tp + fp) if tp + fp else float("nan")
            R = tp / (tp + fn) if tp + fn else float("nan")
            F = 2 * P * R / (P + R) if P + R else float("nan")
            delays = []
            for es, ee in events:
                hits = [t for t, p, _ in preds if p and es <= t <= ee + 1.0]
                if hits:
                    delays.append(min(hits) - es)
            d_str = f"{min(delays):.1f}–{max(delays):.1f}/med {sorted(delays)[len(delays) // 2]:.1f}/{len(delays)}" if delays else "—/0"
            print(f"{wtag:>5} {g:>5} {D:<18} {P:>6.3f} {R:>6.3f} {F:>6.3f} {fp:>3} {fn:>3}  {d_str}")
            rows.append(dict(w=wtag, gate=g, policy=D, P=round(P, 3), R=round(R, 3),
                             F1=round(F, 3), fp=fp, fn=fn, n_delayed=len(delays),
                             delay_med=round(sorted(delays)[len(delays) // 2], 1) if delays else -1,
                             delay_max=round(max(delays), 1) if delays else -1))

OUT = ROOT / "research/it71_policy_ablation.csv"
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\nCSV: {OUT} ({len(rows)} строк)")

canon = next(r for r in rows if r["w"] == "old" and r["gate"] == 0.25 and r["policy"] == "D0 late τ0.5")
print(f"канон (old,0.25,D0): F1={canon['F1']} R={canon['R']} FP={canon['fp']} delay_med={canon['delay_med']}")
cands = [r for r in rows if r["w"] == "new" and r["R"] >= 0.98 and r["F1"] >= 0.95 and r["fp"] <= canon["fp"]]
print("КАНДИДАТЫ (new; R≥0,98, F1≥0,95, FP≤канона):",
      [(r["gate"], r["policy"], r["F1"], r["R"], r["fp"]) for r in cands] or "НЕТ")
for r in cands:
    ok8 = r["delay_med"] <= canon["delay_med"] + 1.0
    print(f"  E8-справка (медиана) {r['gate']}/{r['policy']}: med задержка {r['delay_med']} с "
          f"(порог канон+1,0 = {canon['delay_med'] + 1.0:.1f}) → {'OK' if ok8 else 'КРАСНЫЙ'}")
