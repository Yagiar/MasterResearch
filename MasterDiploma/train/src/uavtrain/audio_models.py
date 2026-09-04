"""Архитектуры акустического классификатора «дрон / не-дрон».

Две архитектуры (выбор флагом `--arch`):
  - `lwcnn`     — LightweightAudioCNN: 3 Conv-BN-ReLU-блока + GAP + Linear, ~24k параметров (edge);
  - `resnet18`  — ResNet-18 (torchvision) с 1-канальным входом, ~11M параметров (качество/обобщение).

ВАЖНО: эти определения должны совпадать с `services/acoustic-detector/src/acoustic_detector/cnn.py`
(веса сохраняются как `state_dict` — несовпадение модулей → load_state_dict упадёт). Вход — `[B, 1, F, T]`
(F = n_mels или n_mfcc, T = n_frames); порядок классов = `["non-drone", "drone"]` (индекс 1 = дрон).
"""

from __future__ import annotations

ARCHS = ("lwcnn", "resnet18")


def build_audio_model(arch: str, *, feature_dim: int = 64, n_frames: int = 64, n_classes: int = 2, pretrained: bool = True):
    """Создать torch.nn.Module по имени архитектуры. Импорт torch/torchvision — ленивый."""
    arch = (arch or "lwcnn").lower()
    if arch == "lwcnn":
        return _build_lwcnn(n_classes)
    if arch == "resnet18":
        return _build_resnet18(n_classes, pretrained=pretrained)
    raise ValueError(f"неизвестная архитектура аудио-классификатора: {arch!r} (ожидалось {ARCHS})")
    _ = (feature_dim, n_frames)  # для resnet18 вход переменного размера обрабатывается AdaptiveAvgPool в конце


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


def _build_resnet18(n_classes: int, *, pretrained: bool = True):
    """ResNet-18 для 1-канальных спектрограмм: conv1 заменён на 1→64 (веса = среднее RGB-весов ImageNet),
    fc заменён на Linear(512, n_classes). `pretrained=True` берёт ImageNet-веса (лучший старт даже для аудио)."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torchvision.models import resnet18  # noqa: PLC0415

    try:
        from torchvision.models import ResNet18_Weights  # noqa: PLC0415
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
    except Exception:  # noqa: BLE001 — старый torchvision / нет интернета на скачку весов
        net = resnet18(weights=None) if not pretrained else resnet18(pretrained=False)

    old_conv = net.conv1  # Conv2d(3, 64, 7, 2, 3, bias=False)
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        # инициализируем 1-канальный conv1 средним по RGB-каналам предобученного (если есть)
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    net.conv1 = new_conv
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net
