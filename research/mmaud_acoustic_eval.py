#!/usr/bin/env python3
"""Независимая акустическая калибровка на MMAUD Mavic3 (it-57).

Аудио из ROSBag (/audio1/audio, 6 кГц → ресемпл 16 кГц), 207.3 с; GT — высота дрона
из lidar-GT (unix-время совпадает с bag): z<0.5 м — стоит, z>1 м — летит.
Метрики: p_drone распределения по состояниям; recall/FP по порогам (независимая
калибровка порога — главный незакрытый тезис ревью §10).
Запуск: research/.venv/bin/python research/mmaud_acoustic_eval.py
"""
import csv
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"
WAV = ROOT / "research/mmaud_audio1_16k.wav"
BAG_T0 = 1692846875.090  # unix-время первого аудиосообщения в bag

import glob as _glob

GTS = sorted(_glob.glob(str(MD / "train/data/mmaud/Mavic3/ground_truth/*.npy")))
gt_ts = [float(Path(g).stem) for g in GTS]
gt_z = [float(np.load(g)[2]) for g in GTS]

def z_at(unix_ts: float) -> float:
    j = min(len(gt_z) - 1, bisect_left(gt_ts, unix_ts))
    return gt_z[j]

sys.path.insert(0, str(MD / "services/acoustic-detector/src"))
sys.path.insert(0, str(MD / "libs/common/src"))
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

from acoustic_detector.ast_classifier import AstAudioClassifier  # noqa: E402

det = AstAudioClassifier(weights_path=str(MD / "models/acoustic/samid-drone-detector"),
                         device="cpu", target_sample_rate=16000)
x, sr = librosa.load(str(WAV), sr=16000, mono=True)
pcm = (x * 32767).astype("<i2")
win, hop = 16000, 8000
rows = []
for k in range(0, (len(pcm) - win) // hop + 1):
    import base64  # noqa: PLC0415

    chunk = pcm[k * hop:k * hop + win]
    t_unix = BAG_T0 + k * 0.5 + 0.5  # центр окна
    d = det.detect(base64.b64encode(chunk.tobytes()).decode(), src_sample_rate=16000, channels=1)
    p = float(d.p_drone) if d.p_drone is not None else (float(d.confidence) if d.label == "drone" else 0.0)
    z = z_at(t_unix)
    rows.append(dict(t_unix=round(t_unix, 2), z=round(z, 3), p_drone=round(p, 4)))
    if (k + 1) % 100 == 0:
        print(f"  {k + 1} окон…", flush=True)

with open(ROOT / "research/mmaud_acoustic_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["t_unix", "z", "p_drone"])
    w.writeheader()
    w.writerows(rows)

static = [r["p_drone"] for r in rows if r["z"] < 0.5]
flight = [r["p_drone"] for r in rows if r["z"] > 1.0]
transit = [r for r in rows if 0.5 <= r["z"] <= 1.0]
def med(v):
    v = sorted(v); return v[len(v) // 2] if v else float("nan")
def p90(v):
    v = sorted(v); return v[int(0.9 * (len(v) - 1))] if v else float("nan")

print(f"\nокон: {len(rows)} (стоит: {len(static)}, летит: {len(flight)}, переход: {len(transit)})")
for name, seg in (("СТОИТ", static), ("ЛЕТИТ", flight)):
    if seg:
        print(f"  {name}: p_drone медиана={med(seg):.3f} p90={p90(seg):.3f} доля ≥0.5: {sum(1 for p in seg if p >= 0.5)/len(seg):.1%}")
print("\nнезависимая калибровка порога (recall по полёту / FP по статике):")
for tau in (0.25, 0.3, 0.5, 0.7):
    r_flight = sum(1 for p in flight if p >= tau) / max(1, len(flight))
    fp_static = sum(1 for p in static if p >= tau) / max(1, len(static))
    print(f"  τ={tau:.2f}: recall(полёт)={r_flight:.1%}  FP(стоит)={fp_static:.1%}")
print(f"\nCSV: {ROOT / 'research/mmaud_acoustic_eval.csv'}")
