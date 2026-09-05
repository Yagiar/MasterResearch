#!/usr/bin/env python3
"""Скоринг решений fusion из decisions.jsonl против GT-разметки sandbox-клипа
(research/gt_sandbox_video.csv) + симуляция политик весов на реальных окнах.

Требует: research/gt_sandbox_video.csv (second,drone_visible,airborne).
Запуск:  python3 research/score_fusion_vs_gt.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from statistics import mean

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)  # корень workspace (it-39: без абсолютных путей)
JSONL = f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"
GT_CSV = f"{ROOT}/research/gt_sandbox_video.csv"
CLIP = 72.609
GAP_S = 120.0


def H(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def prf(preds: list[tuple[int, int]]) -> tuple[float, float, float, int]:
    tp = sum(1 for p, t in preds if p and t)
    fp = sum(1 for p, t in preds if p and not t)
    fn = sum(1 for p, t in preds if not p and t)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return prec, rec, f1, tp + fp + fn


gt = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(GT_CSV))}

rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: r["ts"])
runs: list[list[dict]] = []
for r in rows:
    if runs and r["ts"] - runs[-1][-1]["ts"] <= GAP_S:
        runs[-1].append(r)
    else:
        runs.append([r])


def sec(ts: float, t0: float) -> int:
    return int((ts - t0) % CLIP)


print("=== 1. Выравнивание фазы клипа (по audio-only vs airborne) ===")
t0_fit: dict[int, float] = {}
for i, run in enumerate(runs, 1):
    audio = [r for r in run if r["mode"] == "audio-only"]
    if not audio:
        continue
    best = (float("nan"), None)
    t_first = run[0]["ts"]
    for off in [x * 0.25 for x in range(-12, int(CLIP / 0.25))]:
        t0 = t_first - off
        preds = [(int(r["decision"]), gt[sec(r["ts"], t0)][1]) for r in audio]
        f1 = prf(preds)[2]
        if math.isnan(best[0]) or (not math.isnan(f1) and f1 > best[0]):
            best = (f1, t0)
    t0_fit[i] = best[1]
    print(f"  сегмент #{i}: audio-only n={len(audio)}, лучший F1 vs airborne = {best[0]:.3f} (t0={best[1]:.1f})")

print("\n=== 2. Скоринг режимов против GT (P / R / F1) ===")
for i, run in enumerate(runs, 1):
    if i not in t0_fit:
        continue
    t0 = t0_fit[i]
    print(f"  --- сегмент #{i} ---")
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in run:
        by_mode[r["mode"]].append(r)
    for mode, rs in sorted(by_mode.items()):
        for gt_name, gix in (("visible", 0), ("airborne", 1)):
            preds = [(int(r["decision"]), gt[sec(r["ts"], t0)][gix]) for r in rs]
            p, r_, f1, n = prf(preds)
            print(f"    {mode:<12} vs {gt_name:<8} P={p:.3f} R={r_:.3f} F1={f1:.3f} (n={n})")

print("\n=== 3. Симуляция политик весов (только смешанные окна; моно — без изменений) ===")
for i, run in enumerate(runs, 1):
    if i not in t0_fit:
        continue
    t0 = t0_fit[i]
    mono = [r for r in run if r["contributions"]["p_v"] is None or r["contributions"]["p_a"] is None]
    mixed = [r for r in run if r["contributions"]["p_v"] is not None and r["contributions"]["p_a"] is not None]
    if not mixed:
        print(f"  сегмент #{i}: смешанных окон нет — симуляция не применима")
        continue
    print(f"  --- сегмент #{i}: mono={len(mono)}, mixed={len(mixed)} ---")
    policies = {
        "late 0.5/0.5": (0.5, 0.5),
        "w_v=0.7": (0.7, 0.3),
        "w_v=0.9": (0.9, 0.1),
    }
    for lam in (1.0, 2.0):
        policies[f"entropy λ={lam}"] = None
    for name, wv in policies.items():
        preds = []
        for r in mono:
            preds.append((int(r["decision"]), gt[sec(r["ts"], t0)][1]))
        for r in mixed:
            pv, pa = r["contributions"]["p_v"], r["contributions"]["p_a"]
            if wv is None:
                lam = float(name.split("λ=")[1])
                e_v, e_a = math.exp(-lam * H(pv)), math.exp(-lam * H(pa))
                w1, w2 = e_v / (e_v + e_a), e_a / (e_v + e_a)
            else:
                w1, w2 = wv
            preds.append((int((w1 * pv + w2 * pa) >= 0.5), gt[sec(r["ts"], t0)][1]))
        p, r_, f1, n = prf(preds)
        print(f"    {name:<14} vs airborne: P={p:.3f} R={r_:.3f} F1={f1:.3f} (n={n})")
