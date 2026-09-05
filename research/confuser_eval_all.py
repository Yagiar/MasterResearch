#!/usr/bin/env python3
"""Конфьюзерный скоринг AST на ВСЁМ корпусе ESC-50 — 2000 клипов, 50 классов (it-50).

Расширение it-49 (там 120 клипов airplane/helicopter/engine): полная карта ложных тревог
AST на независимых негативах. Тот же рантайм (samid-drone-detector), окна 1с/шаг 0.5с;
инференс на GPU при доступности (CPU-прогон 2.5 с/окно → ~13 ч, GPU ~десятки мс).
Докачиваемость: уже посчитанные filename из confuser_eval_all.csv пропускаются,
результаты сбрасываются на диск каждые 25 клипов.
Запуск: research/.venv/bin/python research/confuser_eval_all.py
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
ESC = MD / "train/data/esc-50/ESC-50-master"
OUT = ROOT / "research/confuser_eval_all.csv"

sys.path.insert(0, str(MD / "services/acoustic-detector/src"))
sys.path.insert(0, str(MD / "libs/common/src"))

import base64

import numpy as np
import soundfile as sf
import torch

from acoustic_detector.ast_classifier import AstAudioClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

det = AstAudioClassifier(weights_path=str(MD / "models/acoustic/samid-drone-detector"),
                         device=DEVICE, target_sample_rate=16000)

meta = list(csv.DictReader(open(ESC / "meta/esc50.csv", encoding="utf-8")))
print(f"клипов в корпусе: {len(meta)}")

# докачка
done: set[str] = set()
if OUT.exists():
    with open(OUT, encoding="utf-8") as fh:
        done = {r["filename"] for r in csv.DictReader(fh)}
    print(f"уже посчитано: {len(done)} (докачиваем остальные)")
fieldnames = ["filename", "target", "category", "fold", "n_windows", "max_p_drone", "med_p_drone", "alarm_frac_025", "alarm_frac_050"]

fh = open(OUT, "a", newline="", encoding="utf-8")
w = csv.DictWriter(fh, fieldnames=fieldnames)
if not done:
    w.writeheader()

win, hop = 16000, 8000
t_start = time.time()
n_new = 0
import librosa  # noqa: PLC0415

for i, r in enumerate(meta):
    if r["filename"] in done:
        continue
    pcm, sr = sf.read(ESC / "audio" / r["filename"], dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if sr != 16000:
        pcm = (librosa.resample(pcm.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000) * 32767).astype("<i2")
    max_p, ps = 0.0, []
    for k in range(0, max(1, (len(pcm) - win) // hop + 1)):
        chunk = pcm[k * hop:k * hop + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        d = det.detect(base64.b64encode(chunk.tobytes()).decode(), src_sample_rate=16000, channels=1)
        p = float(d.p_drone) if d.p_drone is not None else (float(d.confidence) if d.label == "drone" else 0.0)
        max_p = max(max_p, p)
        ps.append(p)
    ps.sort()
    med = ps[len(ps) // 2] if ps else 0.0
    w.writerow(dict(filename=r["filename"], target=r["target"], category=r["category"], fold=r["fold"],
                    n_windows=len(ps), max_p_drone=round(max_p, 4), med_p_drone=round(med, 4),
                    alarm_frac_025=round(sum(1 for p in ps if p >= 0.25) / len(ps), 4) if ps else 0.0,
                    alarm_frac_050=round(sum(1 for p in ps if p >= 0.5) / len(ps), 4) if ps else 0.0))
    n_new += 1
    if n_new % 25 == 0:
        fh.flush()
        el = time.time() - t_start
        print(f"  {n_new} новых ({i + 1}/{len(meta)}), {el / n_new:.2f} с/клип", flush=True)
fh.flush()
fh.close()
print(f"готово: {n_new} новых клипов за {(time.time() - t_start) / 60:.1f} мин → {OUT}")

# --- сводка ---
rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
print(f"\n=== сводка по {len(rows)} клипам, {len({r['category'] for r in rows})} классам ===")
by_cat: dict[str, list[float]] = {}
for r in rows:
    by_cat.setdefault(r["category"], []).append(float(r["max_p_drone"]))
top = sorted(by_cat.items(), key=lambda kv: -sorted(kv[1])[len(kv[1]) // 2])
print("топ-12 классов-конфьюзеров (медиана max p_drone):")
for cat, vals in top[:12]:
    v = sorted(vals)
    print(f"  {cat:<22} n={len(v):>3}  медиана={v[len(v) // 2]:.3f}  p90={v[int(0.9 * len(v))]:.3f}  max={v[-1]:.3f}")
allmax = sorted(float(r["max_p_drone"]) for r in rows)
print(f"\nFP (max p_drone ≥ τ) на всём негативном корпусе ({len(rows)} клипов × 5 с = {len(rows) * 5 / 3600:.2f} ч аудио):")
for tau in (0.25, 0.35, 0.5, 0.7):
    fp = sum(1 for v in allmax if v >= tau)
    fp_hour = fp * 5 / 3600 * (3600 / (len(rows) * 5 / 3600))  # клипов/час × 5с тревоги — грубая экстраполяция
    print(f"  τ={tau:.2f}: {fp}/{len(rows)} = {fp / len(rows):.1%} клипов; "
          f"экстраполяция ~{fp * 5:.0f} с тревоги на {len(rows) * 5 / 3600:.2f} ч аудио")
