#!/usr/bin/env python3
"""Независимые конфьюзеры: p(drone) AST на ESC-50 airplane/helicopter/engine (it-49).

Контекст: порог аудио-ветки калибровался на sandbox-клипе (it-40, τ=0.25 → R=1.000 при 5 FP).
Ревью §3/§10 требует независимого материала: MMAUD недоступен headless (it-49, блокер), поэтому
первый шаг — независимые НЕГАТИВЫ: ESC-50 (Freesound, независимые записи, не sandbox) классы
airplane/helicopter/engine — акустические конфьюзеры, близкие к «выбегу винтов/технике».
Замер: тот же рантайм AST (samid-drone-detector), окна 1с/шаг 0.5с; на каждом клипе берём
максимум p_drone по окнам (худший случай для FP) и медиану. Итог — доля конфьюзеров,
прорывающихся при τ∈{0.25, 0.5} и рекомендация рабочего порога.
Запуск: research/.venv/bin/python research/confuser_eval.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
ESC = MD / "train/data/esc-50/ESC-50-master"
CONFUSER_CLASSES = {"helicopter", "engine", "airplane"}

sys.path.insert(0, str(MD / "services/acoustic-detector/src"))
sys.path.insert(0, str(MD / "libs/common/src"))

import base64

import numpy as np
import soundfile as sf

from acoustic_detector.ast_classifier import AstAudioClassifier

det = AstAudioClassifier(weights_path=str(MD / "models/acoustic/samid-drone-detector"),
                         device="cpu", target_sample_rate=16000)

meta = list(csv.DictReader(open(ESC / "meta/esc50.csv", encoding="utf-8")))
clips = [r for r in meta if r["category"] in CONFUSER_CLASSES]
print(f"конфьюзеров: {len(clips)} (классы: {sorted(CONFUSER_CLASSES)})")

rows = []
for i, r in enumerate(clips):
    wav = ESC / "audio" / r["filename"]
    pcm, sr = sf.read(wav, dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if sr != 16000:
        import librosa

        pcm = (librosa.resample(pcm.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000) * 32767).astype("<i2")
    win, hop = 16000, 8000
    max_p, med_p = 0.0, []
    for k in range(0, max(1, (len(pcm) - win) // hop + 1)):
        chunk = pcm[k * hop:k * hop + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        d = det.detect(base64.b64encode(chunk.tobytes()).decode(), src_sample_rate=16000, channels=1)
        p = d.p_drone if d.p_drone is not None else (float(d.confidence) if d.label == "drone" else 0.0)
        max_p = max(max_p, float(p))
        med_p.append(float(p))
    med_p.sort()
    med = med_p[len(med_p) // 2] if med_p else 0.0
    rows.append(dict(filename=r["filename"], category=r["category"], fold=r["fold"],
                     max_p_drone=round(max_p, 4), med_p_drone=round(med, 4)))
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(clips)} обработано…")

with open(ROOT / "research/confuser_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename", "category", "fold", "max_p_drone", "med_p_drone"])
    w.writeheader()
    w.writerows(rows)

# сводка по порогам (худший случай: максимум p_drone по окнам клипа)
import statistics

print("\n=== p(drone) на конфьюзерах (120 клипов; максимум по окнам клипа) ===")
for cat in sorted(CONFUSER_CLASSES):
    vals = sorted(r["max_p_drone"] for r in rows if r["category"] == cat)
    n = len(vals)
    print(f"  {cat:<11} медиана={vals[n // 2]:.3f}  p90={vals[int(0.9 * n)]:.3f}  max={vals[-1]:.3f}")
allmax = sorted(r["max_p_drone"] for r in rows)
print(f"  ВСЕ        медиана={allmax[len(allmax) // 2]:.3f}  p90={allmax[int(0.9 * len(allmax))]:.3f}  max={allmax[-1]:.3f}")
print("\nдоля конфьюзеров с max_p_drone ≥ τ (ложные тревоги на независимых негативах):")
for tau in (0.25, 0.35, 0.5, 0.7):
    fp = sum(1 for r in rows if r["max_p_drone"] >= tau)
    print(f"  τ={tau:.2f}: {fp}/{len(rows)} = {fp / len(rows):.1%}")
