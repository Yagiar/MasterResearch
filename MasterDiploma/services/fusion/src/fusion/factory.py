"""FusionStrategyFactory + сборка GatingPolicy по конфигу (паттерн Factory)."""

from __future__ import annotations

from typing import Any

from .gating import AdaptiveGating, FixedGating
from .strategies.audio_only import AudioOnly
from .strategies.base import FusionStrategy
from .strategies.hybrid import HybridFusion
from .strategies.late import LateFusion
from .strategies.video_only import VideoOnly


def _label_threshold(fusion_cfg: dict[str, Any], strat_cfg: dict[str, Any]) -> float:
    return float(strat_cfg.get("label_threshold", fusion_cfg.get("decision_threshold", 0.5)))


def build_strategy(fusion_cfg: dict[str, Any]) -> FusionStrategy:
    """Создать стратегию слияния по `fusion.mode`."""
    mode = str(fusion_cfg.get("mode", "video-only"))
    if mode == "video-only":
        return VideoOnly()
    if mode == "audio-only":
        return AudioOnly()
    if mode == "late":
        c = dict(fusion_cfg.get("late", {}))
        return LateFusion(
            delta_conf=float(c.get("delta_conf", 0.1)),
            delta_unconf=float(c.get("delta_unconf", 0.1)),
            label_threshold=_label_threshold(fusion_cfg, c),
        )
    if mode == "hybrid":
        c = dict(fusion_cfg.get("hybrid", {}))
        return HybridFusion(
            delta_conf=float(c.get("delta_conf", 0.1)),
            delta_unconf=float(c.get("delta_unconf", 0.15)),
            label_threshold=_label_threshold(fusion_cfg, c),
        )
    raise ValueError(f"неизвестный режим слияния: {mode!r}")


def build_gating(fusion_cfg: dict[str, Any]):
    """Создать GatingPolicy по `fusion.gating`.

    `fusion.gating.enabled: true` -> AdaptiveGating (по подсказкам качества из InferenceMsg.quality);
    иначе -> FixedGating с весами из `fusion.late.w_v/w_a` (или `fusion.weights.*`).
    """
    gating_cfg = dict(fusion_cfg.get("gating", {}))
    # базовые веса берём из секции late (она есть всегда) или из явной weights
    weights_cfg = dict(fusion_cfg.get("weights", fusion_cfg.get("late", {})))
    base_w_v = float(weights_cfg.get("w_v", 0.5))
    base_w_a = float(weights_cfg.get("w_a", 0.5))

    if gating_cfg.get("enabled", False):
        return AdaptiveGating(
            base_w_v=base_w_v,
            base_w_a=base_w_a,
            q_floor=float(gating_cfg.get("q_floor", 0.2)),
            sharpness_ref=float(gating_cfg.get("sharpness_ref", 150.0)),
            snr_ref_db=float(gating_cfg.get("snr_ref_db", 20.0)),
            snr_floor_db=float(gating_cfg.get("snr_floor_db", 0.0)),
        )
    return FixedGating(w_v=base_w_v, w_a=base_w_a)
