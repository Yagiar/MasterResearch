#!/usr/bin/env python3
"""Извлечение всех 4 каналов MMAUD-аудио из ROSBag + проверка независимости (it-62).

Гипотеза: каналы решётки — независимые записи; усреднение 4 каналов подавляет
некоррелированный шум крыши (~6 дБ), сохраняя когерентный сигнал дрона.
Запись: research/mmaud_audio{1..4}_6k.wav + сводка корреляций.
Запуск: research/.venv/bin/python research/mmaud_multichannel.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
BAG = Path("/home/otrix/Загрузки/Mavic3.bag")
OUTDIR = ROOT / "research"

from rosbags.rosbag1 import Reader  # noqa: E402
from rosbags.typesys import Stores, get_typestore, get_types_from_msg  # noqa: E402

ts = get_typestore(Stores.ROS1_NOETIC)
ts.register(get_types_from_msg("uint8[] data", "audio_common_msgs/msg/AudioData"))

TOPICS = [f"/audio{i}/audio" for i in range(1, 5)]
chan = {t: [] for t in TOPICS}

with Reader(str(BAG)) as r:
    conns = [c for c in r.connections if c.topic in TOPICS]
    for conn, _, raw in r.messages(connections=conns):
        msg = ts.deserialize_ros1(raw, conn.msgtype)
        chan[conn.topic].append(np.frombuffer(bytes(msg.data), dtype="<i2").astype(np.float32) / 32768.0)
        done = all(len(v) and sum(len(x) for x in chan[t]) > 6000 * 198 for t, v in ((t, chan[t]) for t in TOPICS))
        if done:
            break

for t in TOPICS:
    x = np.concatenate(chan[t])
    out = OUTDIR / f"mmaud_{t.strip('/').replace('/', '_')}_6k.wav".replace("audio1_audio", "audio1").replace("audio_audio", "audio")
    # имя: mmaud_audio1_6k.wav … mmaud_audio4_6k.wav
    ch = t.split("/")[1]  # "audio1".."audio4" (t = /audio1/audio)
    out = OUTDIR / f"mmaud_{ch}_6k.wav"
    sf.write(str(out), x, 6000, subtype="PCM_16")
    print(f"{t}: {len(x)} сэмплов ({len(x)/6000:.1f} с) → {out.name}")

# --- корреляции попарно (на общем 60-с сегменте) ---
def load(ch):
    x, _ = sf.read(str(OUTDIR / f"mmaud_{ch}_6k.wav"))
    return x[:360000]  # 60 с

a1, a2, a3, a4 = load("audio1"), load("audio2"), load("audio3"), load("audio4")
n = min(len(a1), len(a2), len(a3), len(a4))
print("\nпопарная корреляция каналов (60 с):")
pairs = [("1", "2"), ("1", "3"), ("1", "4"), ("2", "3"), ("2", "4"), ("3", "4")]
for i, j in pairs:
    x_, y_ = globals()[f"a{i}"][:n], globals()[f"a{j}"][:n]
    c = float(np.corrcoef(x_, y_)[0, 1])
    print(f"  audio{i} vs audio{j}: {c:.3f}")
avg = (a1[:n] + a2[:n] + a3[:n] + a4[:n]) / 4.0
print(f"\nусреднение 4 каналов: RMS {np.sqrt(np.mean(a1[:n]**2)):.3f} → {np.sqrt(np.mean(avg**2)):.3f}")
sf.write(str(OUTDIR / "mmaud_audio_avg_6k.wav"), avg, 6000, subtype="PCM_16")
print("усреднённый канал → research/mmaud_audio_avg_6k.wav")
