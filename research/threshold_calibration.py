#!/usr/bin/env python3
"""Калибровка порога AST по PR-кривой (it-12): F1(порог) для raw p_a и каузальной медианы-5,
соло и в late с p_v (YOLO). GT: airborne (majority секунды окна).
Запуск: research/.venv/bin/python research/threshold_calibration.py
"""
import csv

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
ast = {r["t0"]: r for r in csv.DictReader(open(f"{ROOT}/research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(f"{ROOT}/research/yolo_sandbox_frames.csv"))
        if r["imgsz"] == "480"}

wins = []
for t0 in sorted(ast.keys(), key=float):
    sec = int(float(t0))
    wins.append((float(yolo[str(sec)]["max_conf"]), float(ast[t0]["p_drone"]),
                 int(ast[t0]["airborne_gt"])))

def causal_med(vals, k=5):
    out = []
    for i in range(len(vals)):
        seg = sorted(vals[max(0, i - k + 1): i + 1])
        out.append(seg[len(seg) // 2])
    return out

pa_med = causal_med([w[1] for w in wins])

def prf(scored, thr):
    preds = [(int(s >= thr), g) for s, (_, _, g) in scored]
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

CHANNELS = {
    "audio raw": [w[1] for w in wins],
    "audio median5": pa_med,
    "late raw (0.5pv+0.5pa)": [0.5 * w[0] + 0.5 * w[1] for w in wins],
    "late median5": [0.5 * w[0] + 0.5 * m for w, m in zip(wins, pa_med)],
}

print(f"окон: {len(wins)} | airborne: {sum(w[2] for w in wins)}")
print(f"{'порог':>5} | " + " | ".join(f"{n:<24}" for n in CHANNELS))
best = {n: (0.0, None) for n in CHANNELS}
for i in range(1, 20):
    thr = i * 0.05
    cells = []
    for name, scores in CHANNELS.items():
        P, R, F = prf(list(zip(scores, wins)), thr)
        cells.append(f"P={P:.2f} R={R:.2f} F1={F:.3f}")
        if F > best[name][0]:
            best[name] = (F, thr, P, R)
    print(f"{thr:5.2f} | " + " | ".join(f"{c:<24}" for c in cells))

print("\nЛучший порог по F1 (и вариант с ограничением P>=0.95):")
for name, scores in CHANNELS.items():
    F, thr, P, R = best[name]
    best95 = None
    for i in range(1, 20):
        t = i * 0.05
        P, R, F = prf(list(zip(scores, wins)), t)
        if P >= 0.95 and (best95 is None or F > best95[0]):
            best95 = (F, t, P, R)
    b95 = f" | при P>=0.95: thr={best95[1]:.2f} F1={best95[0]:.3f} (P={best95[2]:.2f} R={best95[3]:.2f})" if best95 else ""
    print(f"  {name:<24}: thr={thr:.2f} F1={F:.3f} (P={P:.2f} R={R:.2f}){b95}")
