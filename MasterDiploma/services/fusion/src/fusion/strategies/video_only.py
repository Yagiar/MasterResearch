"""VideoOnly — решение по одному видеоканалу (baseline; MVP-режим пайплайна).

p_fused = p_v (уверенность лучшей видеодетекции «дрон» в окне; 0, если дрон не виден).
Аудио игнорируется. Используется как baseline в ablation против late/hybrid.
"""

from __future__ import annotations

from ..window_buffer import AlignedWindow
from .base import FusionOutcome


class VideoOnly:
    """Стратегия «только видео»."""

    mode = "video-only"

    def fuse(self, window: AlignedWindow, *, w_v: float, w_a: float, threshold: float) -> FusionOutcome | None:
        best_v = window.best_video()
        if best_v is None:
            return None  # видеодетекций в окне нет — решение не формируем
        p_v = best_v.confidence if best_v.label == "drone" else 0.0
        return FusionOutcome(
            p_fused=p_v,
            decision=p_v >= threshold,
            p_v=p_v,
            p_a=None,
            w_v=1.0,
            w_a=0.0,
            delta=0.0,
            source_msg_ids=(best_v.msg_id,),
        )
