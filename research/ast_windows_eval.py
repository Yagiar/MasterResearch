#!/usr/bin/env python3
"""AST офлайн по окнам sandbox-аудио: p(drone) на сетке 1с/0.5с, скоринг vs GT, дамп CSV.

Переиспользует рантайм acoustic-detector (тот же код, что в пайплайне).
Запуск: research/.venv/bin/python research/ast_windows_eval.py
"""
import csv
import sys

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)  # корень workspace (it-39: без абсолютных путей)
WAV = f"{ROOT}/MasterDiploma/sandboxDataForSimulator/sandbox-audio-for-simulator.wav"
AST_DIR = f"{ROOT}/MasterDiploma/models/acoustic/samid-drone-detector"
GT = {int(r["second"]): int(r["airborne"])
      for r in csv.DictReader(open(f"{ROOT}/research/gt_sandbox_video.csv"))}

sys.path.insert(0, f"{ROOT}/MasterDiploma/services/acoustic-detector/src")
sys.path.insert(0, f"{ROOT}/MasterDiploma/libs/common/src")

import numpy as np
import soundfile as sf

from acoustic_detector.ast_classifier import AstAudioClassifier

det = AstAudioClassifier(weights_path=AST_DIR, device="cpu", target_sample_rate=16000)

pcm, sr = sf.read(WAV, dtype="int16")
if pcm.ndim > 1:
    pcm = pcm[:, 0]
assert sr == 48000, sr
# ресемплинг 48k→16k (как в пайплайне)
import librosa
pcm16k = librosa.resample(pcm.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000)
pcm16k = (pcm16k * 32767).astype("<i2")

WIN, HOP = 16000, 8000  # 1.0 c / 0.5 c
rows = []
import base64

for k in range(0, (len(pcm16k) - WIN) // HOP + 1):
    t0 = k * 0.5
    chunk = pcm16k[k * HOP : k * HOP + WIN]
    d = det.detect(base64.b64encode(chunk.tobytes()).decode(), src_sample_rate=16000, channels=1)
    # it-36 (ревью §4): честная вероятность положительного класса, БЕЗ обнуления по label.
    # Раньше: p_drone = confidence if label=='drone' else 0.0 — множество значений {0}∪[0.5;1],
    # что делало калибровку порога ниже 0.5 невозможной и порождало ложный вывод о «бимодальности».
    p_drone = d.p_drone if d.p_drone is not None else (
        float(d.confidence) if d.label == "drone" else 0.0)
    # GT окна: доля airborne-секунд в [t0, t0+1) >= 0.5
    secs = [int(np.floor(t0 + x)) for x in np.arange(0, 1.0, 0.1)]
    frac_air = sum(GT.get(s, 0) for s in secs) / len(secs)
    rows.append(dict(t0=round(t0, 1), p_drone=round(p_drone, 4), label=d.label,
                     airborne_gt=int(frac_air >= 0.5)))

with open(f"{ROOT}/research/ast_windows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

tp = sum(1 for r in rows if r["p_drone"] >= 0.5 and r["airborne_gt"])
fp = sum(1 for r in rows if r["p_drone"] >= 0.5 and not r["airborne_gt"])
fn = sum(1 for r in rows if r["p_drone"] < 0.5 and r["airborne_gt"])
P = tp / (tp + fp) if tp + fp else float("nan")
R = tp / (tp + fn) if tp + fn else float("nan")
F = 2 * P * R / (P + R) if P + R else float("nan")
print(f"окон: {len(rows)} | дрона (p>=0.5): {tp+fp} | airborne окон: {sum(r['airborne_gt'] for r in rows)}")
print(f"AST vs airborne GT: P={P:.3f} R={R:.3f} F1={F:.3f}")
print("\nпрофиль p(drone) по t0 (шаг 0.5с, . = <0.5, # = >=0.5):")
line = "".join("#" if r["p_drone"] >= 0.5 else "." for r in rows)
for i in range(0, len(line), 72):
    print(f"  t={i*0.5:5.1f}s  {line[i:i+72]}")
