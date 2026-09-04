"""AudioOnly — решение по одному акустическому каналу (baseline; заготовка).

p_fused = p_a (уверенность лучшей аудиодетекции «дрон» в окне). На MVP не задействован
(акустический детектор — заготовка). Симметричен VideoOnly.
"""

from __future__ import annotations

from ..window_buffer import AlignedWindow
from .base import FusionOutcome


class AudioOnly:
    """Стратегия «только аудио» (заготовка)."""

    mode = "audio-only"

    def fuse(self, window: AlignedWindow, *, w_v: float, w_a: float, threshold: float) -> FusionOutcome | None:
        best_a = window.best_audio()
        if best_a is None:
            return None
        p_a = best_a.confidence if best_a.label == "drone" else 0.0
        return FusionOutcome(
            p_fused=p_a,
            decision=p_a >= threshold,
            p_v=None,
            p_a=p_a,
            w_v=0.0,
            w_a=1.0,
            delta=0.0,
            source_msg_ids=(best_a.msg_id,),
        )
