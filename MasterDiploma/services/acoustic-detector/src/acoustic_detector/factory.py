"""DetectorFactory — сборка FeatureExtractor + детектора (LightweightCnnDetector / AstAudioClassifier) по конфигу.

Маршрутизация по `acoustic_detector.arch`:
  - `lwcnn` | `resnet18` — наш FeatureExtractor (melspec/mfcc) → LightweightCnnDetector со state_dict;
  - `ast`                — AstAudioClassifier (HF transformers AST, predобучен на AudioSet, дообучен на DADS).
                            ASTу нужен СЫРОЙ PCM (свой ASTFeatureExtractor внутри), наши features не используются.
"""

from __future__ import annotations

from typing import Any

from .ast_classifier import AstAudioClassifier
from .cnn import LightweightCnnDetector
from .features import FeatureExtractor


def build_extractor(ad_cfg: dict[str, Any]) -> FeatureExtractor:
    return FeatureExtractor(
        feature=str(ad_cfg.get("feature", "mfcc")),
        sample_rate=int(ad_cfg.get("sample_rate", 16000)),
        n_mfcc=int(ad_cfg.get("n_mfcc", 40)),
        n_mels=int(ad_cfg.get("n_mels", 64)),
        n_frames=int(ad_cfg.get("n_frames", 64)),
    )


def build_detector(ad_cfg: dict[str, Any], extractor: FeatureExtractor):
    """Возвращает детектор с интерфейсом detect(...) + backend/arch/model_name/model_ver.
    Для arch=ast — `expects_pcm=True` (AudioConsumer передаёт сырой PCM, минуя `extractor`)."""
    arch = str(ad_cfg.get("arch", "lwcnn")).lower()
    if arch == "ast":
        return AstAudioClassifier(
            weights_path=str(ad_cfg.get("weights_path", "/models/acoustic/samid-drone-detector")),
            device=str(ad_cfg.get("device", "cpu")),
            target_sample_rate=int(ad_cfg.get("sample_rate", 16000)),
        )
    return LightweightCnnDetector(
        weights_path=str(ad_cfg.get("weights_path", "/models/acoustic/lwcnn.pt")),
        feature_dim=extractor.feature_dim,
        n_frames=extractor.n_frames,
        device=str(ad_cfg.get("device", "cpu")),
        energy_threshold=float(ad_cfg.get("energy_threshold", 0.01)),
        arch=arch,
    )
