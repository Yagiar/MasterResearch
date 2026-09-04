"""AstAudioClassifier — обёртка над предобученной AST-моделью (`Rashidbm/samid-drone-detector`).

Audio Spectrogram Transformer, дообученный на DADS + DroneAudioSet с агрессивными
аугментациями (codec round-trip, urban-noise overlay, SpecAugment, Mixup). Бэкбон —
`MIT/ast-finetuned-audioset-10-10-0.4593` (предобучен на AudioSet), что даёт устойчивость
к «грязному» звуку из видео (низкий SNR, сжатие) — там, где LightweightCNN с DADS-обучением
проваливалась (см. probe_audio.py: p(drone)≤0.026 на sandbox-аудио для lwcnn).

Совместимый интерфейс с `LightweightCnnDetector`: detect(pcm_int16, src_sr, channels) → AudioDetection;
properties backend/arch/model_name/model_ver. Отличие — `expects_pcm=True`: принимает СЫРОЙ PCM, а не
готовые признаки, потому что у AST свой `ASTFeatureExtractor` (128 мел, max_length=1024). Признаки
нашего `FeatureExtractor` для AST не подходят (другие n_mels, нет нужной нормализации).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uavdet_common.serialization import b64decode_str

from .cnn import DRONE_CLASS_INDEX, AudioDetection


def _pcm_int16_to_float(pcm_bytes: bytes, channels: int) -> np.ndarray:
    """PCM int16 little-endian → float32 [-1,1]; при многоканальном берём первый канал."""
    arr = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1 and arr.size % channels == 0:
        arr = arr.reshape(-1, channels)[:, 0]
    return arr


def _estimate_snr_db(signal: np.ndarray) -> float:
    """Грубая оценка SNR окна (для quality.snr_db) — та же формула, что в features.py."""
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
    return float(10.0 * np.log10(max(peak / noise, 1e-8)))


@dataclass(frozen=True)
class AstFeatures:
    """SNR-оценка окна (AST сам строит свою спектрограмму внутри detect())."""

    snr_db: float
    rms: float = 0.0


class AstAudioClassifier:
    """AST-обёртка: PCM int16 → ASTFeatureExtractor → AutoModelForAudioClassification → AudioDetection.

    `weights_path` — КАТАЛОГ с config.json / preprocessor_config.json / model.safetensors
    (локально склонированный HF-репо или путь, который `from_pretrained` примет как HF id).
    Если каталога нет — `backend="energy-threshold"` (заглушка) для сквозного пути без модели.
    """

    expects_pcm = True  # маркер для AudioConsumer: передавать сырой PCM, не наши features

    def __init__(
        self,
        *,
        weights_path: str,
        device: str = "cpu",
        target_sample_rate: int = 16000,
    ) -> None:
        self._weights_path = weights_path
        self._device = device
        self._target_sr = int(target_sample_rate)
        self._model = None
        self._fe = None
        self._id2label: dict[int, str] = {0: "no_drone", 1: "drone"}
        self._backend = "energy-threshold"

        path = Path(weights_path)
        if path.exists() and (path / "config.json").exists() and (path / "preprocessor_config.json").exists():
            self._load(weights_path)
        else:
            # не локальный каталог — может быть HF id ('Rashidbm/samid-drone-detector'); попробуем загрузить
            try:
                self._load(weights_path)
            except Exception:  # noqa: BLE001 — нет модели или нет интернета → fallback на energy-threshold
                pass

    def _load(self, src: str) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  # noqa: PLC0415

        self._fe = AutoFeatureExtractor.from_pretrained(src)
        model = AutoModelForAudioClassification.from_pretrained(src)
        model.eval()
        model.to(self._device)
        self._model = model
        self._id2label = {int(k): v for k, v in (model.config.id2label or {}).items()}
        self._backend = "ast"
        _ = torch  # подавить noqa

    @property
    def model_name(self) -> str:
        return Path(self._weights_path).name or "ast"

    @property
    def arch(self) -> str:
        return "ast" if self._backend == "ast" else "energy-threshold"

    @property
    def model_ver(self) -> str:
        return "1" if self._backend == "ast" else "fallback"

    @property
    def backend(self) -> str:
        return self._backend

    def features_from_pcm(self, payload_b64: str, *, src_sample_rate: int, channels: int) -> AstFeatures:
        """Грубо посчитать SNR окна (для quality.snr_db); AST сам декодит PCM в detect_pcm."""
        sig = _pcm_int16_to_float(b64decode_str(payload_b64), channels)
        if src_sample_rate != self._target_sr and sig.size > 0:
            try:
                import librosa  # noqa: PLC0415
                sig = librosa.resample(sig, orig_sr=src_sample_rate, target_sr=self._target_sr)
            except Exception:  # noqa: BLE001
                pass
        return AstFeatures(snr_db=_estimate_snr_db(sig), rms=float(np.sqrt(np.mean(sig**2))) if sig.size else 0.0)

    def detect(self, payload_b64: str, *, src_sample_rate: int, channels: int) -> AudioDetection:
        """Сырой PCM → label/confidence. payload_b64 — base64 PCM int16 LE моно/стерео; channels — из meta."""
        if self._backend != "ast":
            return AudioDetection(label="non-drone", confidence=0.5, backend="energy-threshold")
        import torch  # noqa: PLC0415

        sig = _pcm_int16_to_float(b64decode_str(payload_b64), channels)
        if src_sample_rate != self._target_sr and sig.size > 0:
            try:
                import librosa  # noqa: PLC0415
                sig = librosa.resample(sig, orig_sr=src_sample_rate, target_sr=self._target_sr)
            except Exception:  # noqa: BLE001
                pass
        if sig.size == 0:
            return AudioDetection(label="non-drone", confidence=0.5, backend="ast")
        inputs = self._fe(sig, sampling_rate=self._target_sr, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
        # маппинг id2label: 0=no_drone, 1=drone (из config.json модели samid)
        drone_idx = next((i for i, lbl in self._id2label.items() if str(lbl).lower() in ("drone", "yes_drone", "uav")), DRONE_CLASS_INDEX)
        drone_idx = int(min(drone_idx, len(probs) - 1))
        p_drone = float(probs[drone_idx])
        return AudioDetection(
            label="drone" if p_drone >= 0.5 else "non-drone",
            confidence=p_drone if p_drone >= 0.5 else 1.0 - p_drone,
            backend="ast",
            p_drone=p_drone,
        )
