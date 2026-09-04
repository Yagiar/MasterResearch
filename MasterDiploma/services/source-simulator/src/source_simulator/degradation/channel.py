"""DegradationChannel — декоратор адаптера источника: оборачивает итераторы
кадров/аудио-окон, прогоняя каждое событие через цепочку стратегий деградации.
Если стратегия вернула None — событие выпадает из потока.

На MVP канал не используется (configs.pilot.yaml -> source.degradation.enabled: false);
если включён без стратегий — ведёт себя как прозрачный pass-through.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from ..adapters.base import AudioItem, DataSourceAdapter, FrameItem
from .strategies import DegradationStrategy


class DegradationChannel:
    """Декоратор поверх DataSourceAdapter (паттерн Decorator)."""

    def __init__(self, inner: DataSourceAdapter, strategies: Iterable[DegradationStrategy]) -> None:
        self._inner = inner
        self._strategies = list(strategies)
        # имя для логов/метрик
        self.name = f"degraded({inner.name})" if self._strategies else inner.name

    @property
    def source_id(self) -> str:
        return self._inner.source_id

    def _pipe(self, item: FrameItem | AudioItem) -> FrameItem | AudioItem | None:
        current: FrameItem | AudioItem | None = item
        for strategy in self._strategies:
            if current is None:
                return None
            current = strategy.apply(current)
        return current

    def frames(self) -> Iterator[FrameItem]:
        for frame in self._inner.frames():
            out = self._pipe(frame)
            if isinstance(out, FrameItem):
                yield out

    def audio_windows(self) -> Iterator[AudioItem]:
        for window in self._inner.audio_windows():
            out = self._pipe(window)
            if isinstance(out, AudioItem):
                yield out

    def close(self) -> None:
        self._inner.close()
