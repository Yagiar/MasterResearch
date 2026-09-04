#!/usr/bin/env python3
"""Симуляция политик fusion на ПОЛНОСТЬЮ смешанных окнах (YOLO×AST), it-06.

В отличие от jsonl-прогонов (где аудио в fusion почти не доходило), здесь обе
модальности доступны в каждом из 144 окон: p_v — YOLO max_conf по секундам
(it-04, imgsz 480), p_a — AST p(drone) по окнам 1с/0.5с (it-05).
GT: airborne (majority секунды окна).
Запуск: research/.venv/bin/python research/fusion_sim_full.py
"""
import csv
import math

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"

ast = {r["t0"]: r for r in csv.DictReader(open(f"{ROOT}/research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(f"{ROOT}/research/yolo_sandbox_frames.csv"))
        if r["imgsz"] == "480"}

WINS = sorted(ast.keys(), key=float)

def H(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def prf(preds):
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * R / (P + R) if P + R else float("nan")
    return P, R, F

# окна: (t0, p_v, p_a, gt_airborne)
wins = []
for t0 in WINS:
    sec = int(float(t0))
    pv = float(yolo[str(sec)]["max_conf"]) if str(sec) in yolo else float(yolo[str(sec + 1)]["max_conf"])
    pa = float(ast[t0]["p_drone"])
    wins.append((float(t0), pv, pa, int(ast[t0]["airborne_gt"])))

def fuse(pv, pa, policy):
    if policy == "video-only":
        return pv
    if policy == "audio-only":
        return pa
    if policy.startswith("late"):
        wv = float(policy.split("w_v=")[1]) if "w_v=" in policy else 0.5
        base = wv * pv + (1 - wv) * pa
        if "+Δ" in policy:
            if pv >= 0.5 and pa >= 0.5:
                base += 0.1          # δ_conf: согласие «дрон»
            elif (pv >= 0.5) != (pa >= 0.5):
                base -= 0.1          # δ_unconf: противоречие
        return min(max(base, 0.0), 1.0)
    if policy.startswith("entropy"):
        lam = float(policy.split("λ=")[1])
        e_v, e_a = math.exp(-lam * H(pv)), math.exp(-lam * H(pa))
        return (e_v * pv + e_a * pa) / (e_v + e_a)
    if policy == "consensus":
        if (pv >= 0.5) == (pa >= 0.5):
            return 0.5 * pv + 0.5 * pa
        # противоречие: вес менее уверенного канала → 0
        return pv if pv >= pa else pa
    raise ValueError(policy)

POLICIES = ["video-only", "audio-only", "late 0.5/0.5", "late 0.5/0.5+Δ",
            "late w_v=0.7", "late w_v=0.9", "entropy λ=1", "entropy λ=2", "consensus"]

print(f"окон: {len(wins)} (все смешанные: YOLO×AST) | airborne окон: {sum(w[3] for w in wins)}\n")
print(f"{'политика':<18} {'P':>6} {'R':>6} {'F1':>6}   FP   FN")
for pol in POLICIES:
    preds = [(int(fuse(pv, pa, pol) >= 0.5), gt) for _, pv, pa, gt in wins]
    P, R, F = prf(preds)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    print(f"{pol:<18} {P:6.3f} {R:6.3f} {F:6.3f}  {fp:4d} {fn:4d}")

# --- временнАя консистентность аудио: медианный фильтр p_a (окна перекрываются) ---
def median_smooth(vals: list[float], k: int) -> list[float]:
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - k // 2), min(len(vals), i + k // 2 + 1)
        seg = sorted(vals[lo:hi])
        out.append(seg[len(seg) // 2])
    return out

print("\n--- с медианным фильтром p_a (временная консистентность в fusion-слое) ---")
results_csv = []
for k in (3, 5, 7):
    pas = median_smooth([w[2] for w in wins], k)
    for name, mk in (("audio-only", lambda pv, pa, w: pa),
                     ("late 0.5/0.5", lambda pv, pa, w: 0.5 * pv + 0.5 * pa)):
        preds = [(int(mk(w[1], pa, w) >= 0.5), w[3]) for w, pa in zip(wins, pas)]
        P, R, F = prf(preds)
        fp = sum(1 for p, t in preds if p and not t)
        fn = sum(1 for p, t in preds if not p and t)
        print(f"{name} + median{k:<2}      {P:6.3f} {R:6.3f} {F:6.3f}  {fp:4d} {fn:4d}")
        results_csv.append(dict(policy=f"{name}+median{k}", P=round(P,3), R=round(R,3), F1=round(F,3), FP=fp, FN=fn))

with open(f"{ROOT}/research/fusion_sim_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["policy", "P", "R", "F1", "FP", "FN"])
    w.writeheader()
    for pol in POLICIES:
        preds = [(int(fuse(pv, pa, pol) >= 0.5), gt) for _, pv, pa, gt in wins]
        P, R, F = prf(preds)
        fp = sum(1 for p, t in preds if p and not t)
        fn = sum(1 for p, t in preds if not p and t)
        w.writerow(dict(policy=pol, P=round(P,3), R=round(R,3), F1=round(F,3), FP=fp, FN=fn))
    w.writerows(results_csv)

# профиль одной лучшей политики: где late ошибается
print("\nlate 0.5/0.5+Δ: расхождения с GT (t0, p_v, p_a, pred, gt):")
for t0, pv, pa, gt in wins:
    pred = int(fuse(pv, pa, "late 0.5/0.5+Δ") >= 0.5)
    if pred != gt:
        print(f"  t={t0:5.1f}  p_v={pv:.2f}  p_a={pa:.2f}  pred={pred}  gt={gt}")
