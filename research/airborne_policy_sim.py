#!/usr/bin/env python3
"""Офлайн-симуляция политик «airborne» на записанных решениях (it-54).

Данные (без новых прогонов):
  * ПОЗИТИВ: sandbox floor0.3 (watermark k5 Δ0), media-мэппинг корректен → recall vs airborne;
  * НЕГАТИВ: negative-session floor0.3 (все секунды airborne=0) → FP. media_ts негативного
    прогона искажён fps-метаданными concat-файла — для FP-симуляции не важно (все окна негативны).

Политики (target=airborne; видео = присутствие, аудио = активность/полёт):
  P0 (записанная):      fused = 0.5·p_v + 0.5·p_a ≥ 0.5                      — it-43
  P1 audio-подтвержд.:  fused ≥ 0.5 И p_a ≥ τ_a (аудио обязано подтвердить)
  P2 аудио-первично:    p_a ≥ τ_a (видео игнорируется)
  P3 видео-вето:        fused ≥ 0.5 И НЕ(p_v ≥ 0.5 И p_a < τ_low) — видео противоречит только
                        при «виден И аудио молчит»
Свип τ_a ∈ [0.1..0.9]. Итог: recall (sandbox) × FP (негатив) — обоснование выбора политики.
Запуск: research/.venv/bin/python research/airborne_policy_sim.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in open(ROOT / "MasterDiploma/data/decisions/decisions.jsonl", encoding="utf-8") if l.strip()]

neg = rows[129844:130285]            # негативная сессия, target=airborne floor=0.3 (все окна airborne=0)
sand = rows[129385:129840]           # sandbox, target=airborne floor=0.3 (11–65 с airborne=1)
CLIP = 72.609
gt = {int(r["second"]): int(r["airborne"]) for r in csv.DictReader(open(ROOT / "research/gt_sandbox_video.csv"))}

def fused(d):
    c = d["contributions"]
    pv = c.get("p_v") or 0.0
    pa = c.get("p_a") or 0.0
    return 0.5 * pv + 0.5 * pa

def sim(policy, tau):
    """Возврат: (recall_sandbox, fp_neg) — доли положительных решений."""
    # sandbox: recall + FP на наземных секундах
    tp = fp_s = fn = 0
    for d in sand:
        c = d["contributions"]
        pv = c.get("p_v") or 0.0
        pa = c.get("p_a") or 0.0
        gt_sec = gt[int(d["media_ts"] % CLIP)] if d.get("media_ts") is not None else None
        if policy == "P0":
            dec = fused(d) >= 0.5
        elif policy == "P1":
            dec = fused(d) >= 0.5 and pa >= tau
        elif policy == "P2":
            dec = pa >= tau
        elif policy == "P3":
            dec = fused(d) >= 0.5 and not (pv >= 0.5 and pa < 0.15)
        if gt_sec is None:
            continue
        if dec and gt_sec:
            tp += 1
        elif dec and not gt_sec:
            fp_s += 1
        elif not dec and gt_sec:
            fn += 1
    R = tp / (tp + fn) if tp + fn else 0.0
    # негатив: доля FP (все окна негативны)
    fp_n = 0
    for d in neg:
        c = d["contributions"]
        pv = c.get("p_v") or 0.0
        pa = c.get("p_a") or 0.0
        if policy == "P0":
            dec = fused(d) >= 0.5
        elif policy == "P1":
            dec = fused(d) >= 0.5 and pa >= tau
        elif policy == "P2":
            dec = pa >= tau
        elif policy == "P3":
            dec = fused(d) >= 0.5 and not (pv >= 0.5 and pa < 0.15)
        fp_n += bool(dec)
    return R, fp_s, fn, fp_n / max(1, len(neg))

print(f"позитив (sandbox floor0.3): {len(sand)} реш. | негатив (negative-session floor0.3): {len(neg)} реш.")
print(f"\n{'политика':<28} {'recall(sand)':>12} {'FP(ground)':>10} {'FP(neg-сессия)':>15}")
out = []
for policy in ("P0", "P1", "P2", "P3"):
    if policy == "P0":
        R, fp_s, fn, fp_n = sim("P0", 0)
        print(f"{policy:<28} {R:>12.3f} {fp_s:>10} {fp_n:>15.1%}")
        out.append(dict(policy=policy, tau=0.0, recall=round(R, 3), fp_ground=fp_s, fp_neg=round(fp_n, 3)))
        continue
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        R, fp_s, fn, fp_n = sim(policy, tau)
        print(f"{policy} τ_a={tau:.1f}{'':<14}"[:28].ljust(28) + f"{R:>12.3f} {fp_s:>10} {fp_n:>15.1%}")
        out.append(dict(policy=f"{policy} τ_a={tau:.1f}", recall=round(R, 3), fp_ground=fp_s, fp_neg=round(fp_n, 3)))

with open(ROOT / "research/airborne_policy_sim.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["policy", "tau", "recall", "fp_ground", "fp_neg"])
    w.writeheader()
    w.writerows(out)
print("\nCSV: research/airborne_policy_sim.csv")
