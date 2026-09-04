#!/usr/bin/env python3
"""Стресс-симуляция политик fusion на смешанных окнах (it-07).

Стрессы (аналоги «канала деградации» пайплайна, офлайн):
  - WindowDrop аудио: окно теряет аудио с вероятностью p_drop → решение по видео (перенормировка);
  - NoiseDegradation аудио: p_a' = clip(p_a + N(0, sigma), 0, 1) — уверенность канала становится шумной;
  - Outage: аудио полностью пропадает на последних X% клипа (отказ модальности).
Каждая точка усреднена по N_SEEDS сидам (mean ± std). GT: airborne (majority секунды).
Запуск: research/.venv/bin/python research/stress_sim.py
"""
import csv
import math
import random
from statistics import mean, stdev

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
N_SEEDS = 25

ast = {r["t0"]: r for r in csv.DictReader(open(f"{ROOT}/research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(f"{ROOT}/research/yolo_sandbox_frames.csv"))
        if r["imgsz"] == "480"}

wins = []
for t0 in sorted(ast.keys(), key=float):
    sec = int(float(t0))
    pv = float(yolo[str(sec)]["max_conf"])
    wins.append((float(t0), pv, float(ast[t0]["p_drone"]), int(ast[t0]["airborne_gt"])))
N = len(wins)


def H(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def median_smooth(vals, k=5):
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - k // 2), min(len(vals), i + k // 2 + 1)
        seg = sorted(vals[lo:hi])
        out.append(seg[len(seg) // 2])
    return out


def decide(pol, pv, pa):
    """pa=None → моно-видео (перенормировка, как в пайплайне)."""
    if pol == "video-only":
        return int(pv >= 0.5)
    if pa is None:
        return int(pv >= 0.5)
    if pol == "audio-only":
        return int(pa >= 0.5)
    if pol.startswith("late"):
        wv = float(pol.split("w_v=")[1]) if "w_v=" in pol else 0.5
        s = wv * pv + (1 - wv) * pa
        if "+Δ" in pol:
            if pv >= 0.5 and pa >= 0.5:
                s += 0.1
            elif (pv >= 0.5) != (pa >= 0.5):
                s -= 0.1
        return int(s >= 0.5)
    if pol.startswith("entropy"):
        lam = float(pol.split("λ=")[1])
        e_v, e_a = math.exp(-lam * H(pv)), math.exp(-lam * H(pa))
        return int((e_v * pv + e_a * pa) / (e_v + e_a) >= 0.5)
    if pol == "consensus":
        if (pv >= 0.5) == (pa >= 0.5):
            return int(0.5 * pv + 0.5 * pa >= 0.5)
        return int((pv if pv >= pa else pa) >= 0.5)
    if pol == "late+median5":
        return None  # обрабатывается отдельно (нужен весь ряд)
    raise ValueError(pol)


def run_trial(pol, drop_p=0.0, sigma=0.0, outage_frac=0.0, rng=None):
    rng = rng or random.Random(0)
    pas = []
    for i, (_, pv, pa, _) in enumerate(wins):
        if rng.random() < drop_p or (outage_frac and i >= N * (1 - outage_frac)):
            pas.append(None)
        else:
            pas.append(min(max(pa + rng.gauss(0, sigma), 0.0), 1.0))
    if pol == "late+median5":
        # медиана только по ВАЛИДНЫМ окнам в радиусе k//2 (пропуски не кодируем нулями)
        preds = []
        for i, (_, pv, _, gt) in enumerate(wins):
            if pas[i] is None:
                preds.append((int(pv >= 0.5), gt))
                continue
            lo, hi = max(0, i - 2), min(N, i + 3)
            valid = [pas[j] for j in range(lo, hi) if pas[j] is not None]
            sm = sorted(valid)[len(valid) // 2] if valid else pas[i]
            preds.append((int(0.5 * pv + 0.5 * sm >= 0.5), gt))
    else:
        preds = [(decide(pol, pv, pa), gt) for (_, pv, _, gt), pa in zip(wins, pas)]
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return 2 * P * R / (P + R) if P + R else 0.0


POLICIES = ["video-only", "audio-only", "late 0.5/0.5", "late 0.5/0.5+Δ",
            "late w_v=0.7", "late w_v=0.9", "entropy λ=1", "consensus", "late+median5"]

rows = []
print("=== 1. WindowDrop аудио (F1 mean±std, 25 сидов) ===")
print(f"{'p_drop':>7} | " + " | ".join(f"{p.replace('late ', 'l').replace('0.5/0.5+Δ','Δ').replace('0.5/0.5','05')[:9]:>9}" for p in POLICIES))
csv_out = []
for drop_p in (0.0, 0.2, 0.4, 0.6, 0.8):
    cells = []
    for pol in POLICIES:
        vals = [run_trial(pol, drop_p=drop_p, rng=random.Random(s)) for s in range(N_SEEDS)]
        m, sd = mean(vals), stdev(vals)
        cells.append(f"{m:.3f}±{sd:.3f}")
        csv_out.append(dict(stress=f"drop{drop_p}", policy=pol, F1_mean=round(m, 4), F1_std=round(sd, 4)))
    print(f"{drop_p:>7} | " + " | ".join(f"{c:>9}" for c in cells))

print("\n=== 2. Шумовая деградация аудио sigma (F1 mean±std) ===")
print(f"{'sigma':>7} | " + " | ".join(f"{p.replace('late ', 'l').replace('0.5/0.5+Δ','Δ').replace('0.5/0.5','05')[:9]:>9}" for p in POLICIES))
for sigma in (0.0, 0.2, 0.4, 0.6):
    cells = []
    for pol in POLICIES:
        vals = [run_trial(pol, sigma=sigma, rng=random.Random(s)) for s in range(N_SEEDS)]
        m, sd = mean(vals), stdev(vals)
        cells.append(f"{m:.3f}±{sd:.3f}")
        csv_out.append(dict(stress=f"noise{sigma}", policy=pol, F1_mean=round(m, 4), F1_std=round(sd, 4)))
    print(f"{sigma:>7} | " + " | ".join(f"{c:>9}" for c in cells))

print("\n=== 3. Отказ аудио на последних X% клипа (F1 mean±std) ===")
print(f"{'outage':>7} | " + " | ".join(f"{p.replace('late ', 'l').replace('0.5/0.5+Δ','Δ').replace('0.5/0.5','05')[:9]:>9}" for p in POLICIES))
for of in (0.1, 0.25, 0.5):
    cells = []
    for pol in POLICIES:
        vals = [run_trial(pol, outage_frac=of, rng=random.Random(s)) for s in range(N_SEEDS)]
        m, sd = mean(vals), stdev(vals)
        cells.append(f"{m:.3f}±{sd:.3f}")
        csv_out.append(dict(stress=f"outage{of}", policy=pol, F1_mean=round(m, 4), F1_std=round(sd, 4)))
    print(f"{of:>7} | " + " | ".join(f"{c:>9}" for c in cells))

with open(f"{ROOT}/research/stress_sim_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["stress", "policy", "F1_mean", "F1_std"])
    w.writeheader()
    w.writerows(csv_out)
print(f"\nCSV: research/stress_sim_results.csv ({len(csv_out)} строк)")
