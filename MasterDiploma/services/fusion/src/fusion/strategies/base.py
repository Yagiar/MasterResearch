"""Базовые типы стратегий слияния."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..window_buffer import AlignedWindow


@dataclass(frozen=True)
class FusionOutcome:
    """Результат слияния для одного окна выравнивания."""

    p_fused: float           # итоговая вероятность «БПЛА»
    decision: bool           # p_fused >= порога
    p_v: float | None        # вклад видео (None — модальность не участвовала)
    p_a: float | None        # вклад аудио
    w_v: float | None        # вес видео
    w_a: float | None        # вес аудио
    delta: float = 0.0       # поправка правила компенсации (late/hybrid); 0 для video-only/audio-only
    source_msg_ids: tuple[str, ...] = ()


@runtime_checkable
class FusionStrategy(Protocol):
    """Контракт стратегии слияния."""

    mode: str

    def fuse(self, window: AlignedWindow, *, w_v: float, w_a: float, threshold: float) -> FusionOutcome | None:
        """Слить окно в решение; None — если данных в окне недостаточно для этой стратегии."""
        ...


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))
