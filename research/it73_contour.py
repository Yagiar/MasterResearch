#!/usr/bin/env python3
"""it-73 контур: SAHI-pv против full-frame pv на канкасе окон it-71 (ноль инференса).

Плечи: old/new × full480/sahi640 × гейт {0,25; 0,4} × политика D0 (late τ=0,5).
Канон самепроверки: old-full g0,25 D0 → F1=0,961 R=0,991 FP=8 (числа it-71).
W1: возврат ≥4/6 потерянных it-66 airborne-окон и R(new-sahi) ≥ 0,98.
Критерии предрегистрированы: iterations/it-73-sahi-pv-contour-PLANNED.md.
Запуск: research/.venv/bin/python research/it73_contour.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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


def load_windows(yolo_csv: str, imgsz: str):
    yolo = {r["second"]: r for r in csv.DictReader(open(ROOT / yolo_csv)) if r["imgsz"] == imgsz}
    wins = []
    for t0 in WINS_T:
        sec = int(float(t0))
        raw = float(yolo[str(sec)]["max_conf"]) if str(sec) in yolo else float(yolo[str(sec + 1)]["max_conf"])
        wins.append((float(t0), raw, float(ast[t0]["p_drone"]), int(ast[t0]["airborne_gt"])))
    return wins


def run(wins, g):
    preds = []
    for t0, raw, pa, gt in wins:
        pv = raw if raw >= g else 0.0
        preds.append((t0, 0.5 * pv + 0.5 * pa >= 0.5, gt))
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
    dmed = sorted(delays)[len(delays) // 2] if delays else -1
    return preds, dict(P=P, R=R, F1=F, fp=fp, fn=fn, n_delayed=len(delays), delay_med=dmed)


SH = {"old-full": ("research/yolo_sandbox_frames.csv", "480"),
      "new-full": ("research/yolo_sandbox_frames_new.csv", "480"),
      "old-sahi": ("research/yolo_sandbox_frames_old_sahi640.csv", "sahi640"),
      "new-sahi": ("research/yolo_sandbox_frames_new_sahi640.csv", "sahi640")}
res = {}
print(f"{'плечо':>9} {'g':>5} {'P':>6} {'R':>6} {'F1':>6} {'FP':>3} {'FN':>3} {'medD':>5} {'nD':>3}")
for tag, (path, imgsz) in SH.items():
    wins = load_windows(path, imgsz)
    for g in (0.25, 0.4):
        preds, m = run(wins, g)
        res[(tag, g)] = preds
        print(f"{tag:>9} {g:>5} {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} {m['fp']:>3} {m['fn']:>3} {m['delay_med']:>5.1f} {m['n_delayed']:>3}")

canon = res[("old-full", 0.25)]
newf = res[("new-full", 0.25)]
news = res[("new-sahi", 0.25)]
lost = [c[0] for c, nf in zip(canon, newf) if c[2] and not nf[1]]
print(f"\nсами-проверка канона old-full g0,25: R={sum(1 for _, p, t in canon if p and t) / max(1, sum(1 for _, _, t in canon if t)):.3f}")
print(f"потерянные new-full окна (GT airborne, old positive): {len(lost)} шт: {lost}")
restore = sum(1 for ns in news if ns[0] in lost and ns[1])
print(f"возвращено new-sahi g0,25 из них: {restore}/{len(lost)}")
fp_sahi = sorted(t for t, p, gt in news if p and not gt)
print(f"FP-окна new-sahi g0,25: {fp_sahi}")
fp_canon = sorted(t for t, p, gt in canon if p and not gt)
print(f"FP-окна канона (old-full): {fp_canon}")
