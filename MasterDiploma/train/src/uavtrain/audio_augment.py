"""Аугментации аудио для обучения акустического классификатора — закрытие domain gap.

Идея (как в MERIDIAN, arXiv:2506.11049): модель, обученная на «чистом» домене (DADS — записи
вблизи микрофона), не обобщается на «грязный» (звук дрона из видео: кодек/ветер/фон, низкий SNR).
Лечится не размером модели, а аугментациями, имитирующими целевой домен — применяются on-the-fly
ТОЛЬКО к train-сплиту:
  - подмешивание фонового шума (ESC-50 / DADS#non-drone) на случайном SNR;
  - random gain (изменение громкости);
  - pitch / time shift (librosa) — небольшой;
  - codec_sim: ресемпл 16k→8k→16k + лёгкий клиппинг (имитация сжатия аудиодорожки видео);
  - SpecAugment на мел-спектрограмме: маскирование полос по времени и частоте.

Аугментации сигнала применяются ДО извлечения признаков; SpecAugment — ПОСЛЕ (на спектрограмме).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EPS = 1e-8


@dataclass
class AugmentConfig:
    enabled: bool = False
    p_background: float = 0.6          # вероятность подмешать фон
    snr_db_range: tuple[float, float] = (0.0, 20.0)   # диапазон SNR для микса (низкий → «грязнее»)
    p_gain: float = 0.5
    gain_db_range: tuple[float, float] = (-10.0, 6.0)
    p_pitch: float = 0.3
    pitch_steps_range: tuple[float, float] = (-2.0, 2.0)   # полутона
    p_time_shift: float = 0.3
    time_shift_frac: float = 0.2       # сдвиг до ±20% длины окна
    p_codec: float = 0.4               # имитация сжатия (16k→8k→16k + клиппинг)
    p_specaug: float = 0.6
    spec_time_masks: int = 2
    spec_freq_masks: int = 2
    spec_time_mask_frac: float = 0.15  # ширина маски по времени до 15% T
    spec_freq_mask_frac: float = 0.15  # ширина маски по частоте до 15% F
    bg_wavs: list[Path] = field(default_factory=list)  # пул фоновых wav для миксов (заполняется в train)


def collect_background_wavs(dirs_or_specs: list[str], *, dataset_dir_fn, limit: int = 4000) -> list[Path]:
    """Собрать пул фоновых wav из каталогов датасетов / 'имя#подпуть' (как в prepare_audio). Подвыборка до limit."""
    out: list[Path] = []
    for spec in dirs_or_specs:
        spec = spec.strip()
        if not spec:
            continue
        if "#" in spec:
            name, sub = spec.split("#", 1)
            root = dataset_dir_fn(name.strip()) / sub.strip()
        elif "/" in spec or Path(spec).exists():
            root = Path(spec)
        else:
            root = dataset_dir_fn(spec)
        if root.exists():
            out.extend(sorted(root.rglob("*.wav")))
    if len(out) > limit:
        rng = random.Random(1337)
        out = rng.sample(out, limit)
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + _EPS))


def _mix_background(sig: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    """Подмешать фон к сигналу так, чтобы SNR(сигнал/фон) ≈ snr_db. Фон тайлится/обрезается под длину sig."""
    if bg.size == 0:
        return sig
    if bg.size < sig.size:
        reps = int(np.ceil(sig.size / bg.size))
        bg = np.tile(bg, reps)
    start = random.randint(0, max(0, bg.size - sig.size))
    bg = bg[start : start + sig.size]
    s_rms, n_rms = _rms(sig), _rms(bg)
    if n_rms < _EPS:
        return sig
    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    return (sig + bg * (target_n_rms / n_rms)).astype(np.float32)


def augment_signal(sig: np.ndarray, cfg: AugmentConfig, sr: int, librosa_mod) -> np.ndarray:
    """Аугментации на уровне сигнала (вызывать ДО извлечения признаков). sig — float [-1,1]."""
    x = sig.astype(np.float32)
    if cfg.p_time_shift and random.random() < cfg.p_time_shift and x.size:
        shift = int(random.uniform(-cfg.time_shift_frac, cfg.time_shift_frac) * x.size)
        x = np.roll(x, shift)
    if cfg.p_pitch and random.random() < cfg.p_pitch and x.size:
        steps = random.uniform(*cfg.pitch_steps_range)
        try:
            x = librosa_mod.effects.pitch_shift(x, sr=sr, n_steps=steps).astype(np.float32)
        except Exception:  # noqa: BLE001
            pass
    if cfg.bg_wavs and cfg.p_background and random.random() < cfg.p_background:
        bg_path = random.choice(cfg.bg_wavs)
        try:
            bg, _ = librosa_mod.load(str(bg_path), sr=sr, mono=True)
            x = _mix_background(x, bg.astype(np.float32), random.uniform(*cfg.snr_db_range))
        except Exception:  # noqa: BLE001
            pass
    if cfg.p_gain and random.random() < cfg.p_gain:
        g = 10.0 ** (random.uniform(*cfg.gain_db_range) / 20.0)
        x = (x * g).astype(np.float32)
    if cfg.p_codec and random.random() < cfg.p_codec and x.size:
        try:
            half = librosa_mod.resample(x, orig_sr=sr, target_sr=sr // 2)
            x = librosa_mod.resample(half, orig_sr=sr // 2, target_sr=sr).astype(np.float32)
            x = np.clip(x * random.uniform(1.0, 2.0), -1.0, 1.0).astype(np.float32)  # лёгкий клиппинг
        except Exception:  # noqa: BLE001
            pass
    # финальная нормализация пика (как librosa.load: модель училась на сигналах в [-1,1])
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = (x / peak).astype(np.float32)
    return x


def spec_augment(mel: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    """SpecAugment: маскирование полос по времени и частоте на спектрограмме [F, T] (нулями).

    Вызывать ПОСЛЕ z-score? Нет — лучше ДО, чтобы маска = 0 после нормировки соответствовала «тишине».
    Здесь применяется к уже готовым (z-score) признакам — маска = 0 ≈ среднее, нейтрально. Достаточно.
    """
    if not (cfg.p_specaug and random.random() < cfg.p_specaug):
        return mel
    out = mel.copy()
    f_dim, t_dim = out.shape
    for _ in range(cfg.spec_freq_masks):
        w = max(1, int(random.uniform(0, cfg.spec_freq_mask_frac) * f_dim))
        f0 = random.randint(0, max(0, f_dim - w))
        out[f0 : f0 + w, :] = 0.0
    for _ in range(cfg.spec_time_masks):
        w = max(1, int(random.uniform(0, cfg.spec_time_mask_frac) * t_dim))
        t0 = random.randint(0, max(0, t_dim - w))
        out[:, t0 : t0 + w] = 0.0
    return out
