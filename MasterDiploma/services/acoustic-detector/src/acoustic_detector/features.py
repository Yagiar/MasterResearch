"""Извлечение акустических признаков из PCM аудио-окна.

Декодирует PCM (int16 little-endian, моно — при многоканальном берёт первый канал) из
payload `AudioRawMsg`, ресемплит к целевой `sample_rate` и считает признаки:
  - `feature="mfcc"`   -> MFCC `[n_mfcc, T]`;
  - `feature="melspec"` -> лог-мел-спектрограмма `[n_mels, T]`.
Признаки приводятся к фиксированной ширине `n_frames` (паддинг/обрезка по времени) и
нормируются (z-score) — это формат, который ест `LightweightAudioCNN` (вход `[1, F, n_frames]`).

Дополнительно отдаёт оценку SNR окна (грубая: через медианную и пиковую энергию) — это
подсказка качества для adaptive gating (`InferenceMsg.quality.snr_db`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdet_common.serialization import b64decode_str

# параметры по умолчанию (синхронизированы с train/uavtrain/prepare_audio.DEFAULT_FEATURE_PARAMS)
DEFAULT_N_MFCC = 40
DEFAULT_N_MELS = 64
DEFAULT_N_FRAMES = 64
_EPS = 1e-8


@dataclass(frozen=True)
class AudioFeatures:
    """Признаки одного окна: тензор-подобный массив [F, n_frames] + оценка SNR (дБ)."""

    array: np.ndarray          # float32, shape [F, n_frames]
    snr_db: float
    rms: float = 0.0           # RMS окна — для health-гейта fusion (research/it-16, it-19)


def _pcm_int16_to_float(pcm_bytes: bytes, channels: int) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1 and arr.size % channels == 0:
        arr = arr.reshape(-1, channels)[:, 0]  # берём первый канал
    return arr


def _estimate_snr_db(signal: np.ndarray) -> float:
    """Грубая оценка SNR окна: 10·log10(пиковая энергия / медианная энергия)."""
    if signal.size == 0:
        return 0.0
    frame = max(1, signal.size // 32)
    energies = np.array([float(np.mean(signal[i : i + frame] ** 2)) for i in range(0, signal.size, frame)])
    energies = energies[energies > 0]
    if energies.size == 0:
        return 0.0
    peak = float(np.percentile(energies, 95))
    noise = float(np.median(energies))
    if noise <= 0:
        return 40.0
    return float(10.0 * np.log10(max(peak / noise, _EPS)))


def _fix_width(feat: np.ndarray, n_frames: int) -> np.ndarray:
    """Привести спектрограмму [F, T] к [F, n_frames] (паддинг нулями справа / обрезка)."""
    f, t = feat.shape
    if t == n_frames:
        out = feat
    elif t > n_frames:
        out = feat[:, :n_frames]
    else:
        out = np.pad(feat, ((0, 0), (0, n_frames - t)), mode="constant")
    return out.astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = float(x.mean()), float(x.std())
    return (x - mu) / (sd + _EPS)


class FeatureExtractor:
    """Экстрактор признаков аудио-окна (librosa)."""

    def __init__(
        self,
        *,
        feature: str = "mfcc",
        sample_rate: int = 16000,
        n_mfcc: int = DEFAULT_N_MFCC,
        n_mels: int = DEFAULT_N_MELS,
        n_frames: int = DEFAULT_N_FRAMES,
    ) -> None:
        if feature not in ("mfcc", "melspec"):
            raise ValueError(f"неизвестный тип признака: {feature!r} (ожидалось mfcc|melspec)")
        self._feature = feature
        self._sr = int(sample_rate)
        self._n_mfcc = int(n_mfcc)
        self._n_mels = int(n_mels)
        self._n_frames = int(n_frames)

    @property
    def feature_dim(self) -> int:
        return self._n_mfcc if self._feature == "mfcc" else self._n_mels

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def from_audio_raw(self, payload_b64: str, *, src_sample_rate: int, channels: int) -> AudioFeatures:
        """payload base64 PCM -> AudioFeatures."""
        import librosa  # noqa: PLC0415

        signal = _pcm_int16_to_float(b64decode_str(payload_b64), channels)
        if src_sample_rate != self._sr and signal.size > 0:
            signal = librosa.resample(signal, orig_sr=src_sample_rate, target_sr=self._sr)
        snr = _estimate_snr_db(signal)
        rms = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0

        if signal.size == 0:
            feat = np.zeros((self.feature_dim, self._n_frames), dtype=np.float32)
            return AudioFeatures(array=feat, snr_db=snr, rms=rms)

        if self._feature == "mfcc":
            feat = librosa.feature.mfcc(y=signal, sr=self._sr, n_mfcc=self._n_mfcc)
        else:
            mel = librosa.feature.melspectrogram(y=signal, sr=self._sr, n_mels=self._n_mels)
            feat = librosa.power_to_db(mel, ref=np.max)
        feat = _zscore(_fix_width(feat, self._n_frames))
        return AudioFeatures(array=feat.astype(np.float32), snr_db=snr, rms=rms)
