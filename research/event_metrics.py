#!/usr/bin/env python3
"""Событийные метрики вместо оконного F1 (it-43; ревью GPT-6-Astra §10).

Ревью: «Для системы предупреждения основными показателями должны быть доля обнаруженных
событий, число ложных тревог в час и задержка первого обнаружения. F1 по окнам —
дополнительный». Событие = непрерывный участок GT «airborne» (≥1 с). Окно считается
обнаружившим событие, если решение положительно в момент, попадающий в событие
(задержка = t_первого_положительного_окна − начало события).
Ложная тревога = положительное решение вне событий; нормировка на час наблюдения.
Один клип 72.6 с ⇒ FP/час экстраполируется и потому груба — это диагностика, не оценка.
Запуск: research/.venv/bin/python research/event_metrics.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import argparse
_YA = argparse.ArgumentParser(); _YA.add_argument("--yolo-csv", default="research/yolo_sandbox_frames.csv"); _YSRC = str(ROOT / _YA.parse_known_args()[0].yolo_csv)  # it-66: пересчёт новыми весами
CLIP = 72.609

gt = {int(r["second"]): (int(r["drone_visible"]), int(r["airborne"]))
      for r in csv.DictReader(open(ROOT / "research/gt_sandbox_video.csv"))}
ast = {r["t0"]: r for r in csv.DictReader(open(ROOT / "research/ast_windows.csv"))}
yolo = {r["second"]: r for r in csv.DictReader(open(_YSRC))
        if r["imgsz"] == "480"}

# события GT «airborne»: непрерывные участки
events = []
cur = None
for sec in sorted(gt):
    if gt[sec][1]:
        if cur is None:
            cur = [sec, sec]
        else:
            cur[1] = sec
    elif cur is not None:
        events.append(tuple(cur)); cur = None
if cur is not None:
    events.append(tuple(cur))
print(f"GT-события «airborne»: {events}")

wins = []
for t0 in sorted(ast.keys(), key=float):
    sec = int(float(t0))
    pv = float(yolo[str(sec)]["max_conf"]) if str(sec) in yolo else float(yolo[str(sec + 1)]["max_conf"])
    wins.append((float(t0), pv, float(ast[t0]["p_drone"]), int(ast[t0]["airborne_gt"])))

def causal_median(vals, k=5):
    out = []
    for i in range(len(vals)):
        seg = sorted(vals[max(0, i - k + 1): i + 1])
        out.append(seg[len(seg) // 2])
    return out

pas_med5 = causal_median([w[2] for w in wins], 5)
window_s = 0.5  # шаг сетки окон

CONFIGS = {
    "audio-only τ=0.25":       lambda i, pv, pa: pa >= 0.25,
    "audio-only τ=0.5":        lambda i, pv, pa: pa >= 0.5,
    "late τ=0.5":              lambda i, pv, pa: 0.5 * pv + 0.5 * pa >= 0.5,
    "late+med5 τ=0.55":        lambda i, pv, pa: 0.5 * pv + 0.5 * pas_med5[i] >= 0.55,
    "video-only τ=0.5":        lambda i, pv, pa: pv >= 0.5,
}

rows_out = []
print(f"\n{'конфиг':<22} {'событ.обнаруж.':>14} {'задержка, с':>12} {'FP':>4} {'FP/час(экстр.)':>15}")
for name, fn in CONFIGS.items():
    preds = [(w[0], fn(i, w[1], w[2])) for i, w in enumerate(wins)]
    # события: доля обнаруженных + задержка первого положительного от старта события
    detected, delays = 0, []
    for (es, ee) in events:
        hits = [t for t, p in preds if p and es <= t <= ee + 1.0]
        if hits:
            detected += 1
            delays.append(min(hits) - es)
    fp = sum(1 for t, p in preds if p and not any(es <= t <= ee for es, ee in events))
    fp_hour = fp / (CLIP / 3600.0)
    d_str = f"{min(delays):.1f}–{max(delays):.1f}" if delays else "—"
    n_str = f"{detected}/{len(events)}"
    print(f"{name:<22} {n_str:>14} {d_str:>12} {fp:>4} {fp_hour:>15.0f}")
    rows_out.append(dict(config=name, events_detected=n_str, delay_s=d_str, fp=fp, fp_per_hour=round(fp_hour)))

with open(ROOT / "research/event_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "events_detected", "delay_s", "fp", "fp_per_hour"])
    w.writeheader(); w.writerows(rows_out)
print("\nCSV: research/event_metrics.csv")
