"""Акустический классификатор спектрограмм «дрон / не-дрон» — две архитектуры.

  - `lwcnn`     — LightweightAudioCNN: 3 Conv-BN-ReLU-блока + GAP + Linear, ~24k параметров (edge);
  - `resnet18`  — ResNet-18 (torchvision) с 1-канальным входом, ~11M параметров (качество/обобщение).

Вход — `[B, 1, F, T]` (F = n_mfcc или n_mels, T = n_frames). ВАЖНО: определения ДОЛЖНЫ совпадать с
`train/src/uavtrain/audio_models.py` (веса сохраняются как `state_dict` — несовпадение модулей →
load_state_dict упадёт). Порядок классов = `["non-drone", "drone"]` (см. `uavtrain.config.AUDIO_CLASSES`):
индекс 1 = «дрон».

`LightweightCnnDetector` — обёртка: грузит `state_dict` из `weights_path` под архитектуру `arch`;
если файла нет — переходит в режим энергетического порога, чтобы сквозной путь работал без обучения.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DRONE_CLASS_INDEX = 1  # AUDIO_CLASSES = ["non-drone", "drone"]
ARCHS = ("lwcnn", "resnet18")


def build_audio_model(arch: str, feature_dim: int = 64, n_frames: int = 64, n_classes: int = 2):
    """Создать torch.nn.Module по имени архитектуры. Импорт torch/torchvision — ленивый.

    Должно быть синхронизировано с train/src/uavtrain/audio_models.build_audio_model (pretrained=False:
    при инференсе веса грузятся из state_dict, ImageNet-инициализация не нужна и не должна тянуть скачку)."""
    arch = (arch or "lwcnn").lower()
    if arch == "lwcnn":
        return _build_lwcnn(n_classes)
    if arch == "resnet18":
        return _build_resnet18(n_classes)
    raise ValueError(f"неизвестная архитектура акустического классификатора: {arch!r} (ожидалось {ARCHS})")
    _ = (feature_dim, n_frames)


def _build_lwcnn(n_classes: int):
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class LightweightAudioCNN(nn.Module):
        def __init__(self, n_cls: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, n_cls)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x).flatten(1))

    return LightweightAudioCNN(n_classes)


def _build_resnet18(n_classes: int):
    """ResNet-18 для 1-канальных спектрограмм: conv1 → 1→64, fc → Linear(512, n_classes). Без предобучения."""
    from torch import nn  # noqa: PLC0415
    from torchvision.models import resnet18  # noqa: PLC0415

    net = resnet18(weights=None)
    old = net.conv1
    net.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


def build_audio_cnn(feature_dim: int, n_frames: int, n_classes: int = 2):
    """LightweightAudioCNN — оставлено для обратной совместимости; см. build_audio_model для выбора архитектуры."""
    _ = (feature_dim, n_frames)
    return _build_lwcnn(n_classes)


@dataclass(frozen=True)
class AudioDetection:
    label: str          # "drone" | "non-drone"
    confidence: float
    backend: str        # "cnn" | "energy-threshold"


class LightweightCnnDetector:
    """Акустический детектор: CNN по спектрограмме либо энергетический порог (если нет весов)."""

    def __init__(
        self,
        *,
        weights_path: str,
        feature_dim: int,
        n_frames: int,
        device: str = "cpu",
        energy_threshold: float = 0.01,   # порог средней энергии окна для режима-заглушки (амплитуды нормированы к [-1,1])
        arch: str = "lwcnn",              # lwcnn | resnet18 — должно совпадать с тем, на чём обучались веса
    ) -> None:
        self._weights_path = Path(weights_path)
        self._device = device
        self._energy_threshold = float(energy_threshold)
        self._arch = (arch or "lwcnn").lower()
        self._model = None
        self._backend = "energy-threshold"
        if self._weights_path.exists():
            self._load_cnn(feature_dim, n_frames)

    def _load_cnn(self, feature_dim: int, n_frames: int) -> None:
        import torch  # noqa: PLC0415

        model = build_audio_model(self._arch, feature_dim=feature_dim, n_frames=n_frames, n_classes=2)
        state = torch.load(str(self._weights_path), map_location=self._device)
        model.load_state_dict(state)
        model.eval()
        model.to(self._device)
        self._model = model
        self._backend = "cnn"

    @property
    def model_name(self) -> str:
        return self._weights_path.stem if self._backend == "cnn" else "energy-threshold"

    @property
    def arch(self) -> str:
        return self._arch if self._backend == "cnn" else "energy-threshold"

    @property
    def model_ver(self) -> str:
        return "1" if self._backend == "cnn" else "fallback"

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, features_array: np.ndarray, *, raw_energy: float | None = None) -> AudioDetection:
        """Классифицировать окно.

        `features_array` — спектрограмма [F, T] (для CNN-режима). `raw_energy` — средняя
        энергия исходного сигнала окна (для режима-заглушки); если не передана и CNN нет —
        используется энергия признаков как суррогат.
        """
        if self._backend == "cnn":
            import torch  # noqa: PLC0415

            with torch.no_grad():
                x = torch.from_numpy(features_array).float().unsqueeze(0).unsqueeze(0).to(self._device)  # [1,1,F,T]
                logits = self._model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
            p_drone = float(probs[DRONE_CLASS_INDEX])
            return AudioDetection(
                label="drone" if p_drone >= 0.5 else "non-drone",
                confidence=p_drone if p_drone >= 0.5 else 1.0 - p_drone,
                backend="cnn",
            )
        # энергетический порог (заглушка)
        energy = raw_energy if raw_energy is not None else float(np.mean(features_array.astype(np.float64) ** 2))
        is_drone = energy >= self._energy_threshold
        # «уверенность» — насколько энергия отстоит от порога (сжато в [0,1])
        conf = min(1.0, abs(energy - self._energy_threshold) / max(self._energy_threshold, 1e-6))
        conf = 0.5 + 0.5 * conf  # [0.5, 1.0]
        return AudioDetection(label="drone" if is_drone else "non-drone", confidence=conf, backend="energy-threshold")
