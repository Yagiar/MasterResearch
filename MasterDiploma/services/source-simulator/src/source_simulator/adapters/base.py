"""Базовые типы адаптеров источника.

Адаптер — итератор «событий потока»: видеокадров (`FrameItem`) и аудио-окон
(`AudioItem`). `SimulatorController` тянет события и шлёт их в gateway по gRPC,
выдерживая реальный темп (FPS / hop аудио-окна).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class FrameItem:
    """Один видеокадр, готовый к отправке в gateway."""

    jpeg_bytes: bytes
    seq: int
    width: int
    height: int
    fps_nominal: float
    # момент захвата; если None — controller проставит time.time() при отправке
    ts: float | None = None
    # it-34: позиция кадра на таймлайне исходного медиа (с от старта источника, монотонная,
    # с учётом loop-проходов); None — адаптер не знает (синтетика). Ревью §6.1: wall-clock
    # ts не описывает содержимое — нужен общий медиатаймлайн для обеих модальностей.
    media_ts: float | None = None
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioItem:
    """Одно аудио-окно (PCM), готовое к отправке в gateway."""

    pcm_bytes: bytes
    seq: int
    sample_rate: int
    channels: int
    len_ms: int
    hop_ms: int
    ts: float | None = None
    # it-34: позиция НАЧАЛА окна на таймлайне исходного медиа (с); конец = media_ts + len_ms/1000
    media_ts: float | None = None
    meta: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Контракт адаптера источника (паттерн Adapter + Strategy).

    Реализации: DatasetReplayAdapter, VideoFileAdapter, SyntheticAdapter,
    (в будущем) SensorAdapter.
    """

    name: str

    @property
    def source_id(self) -> str:
        """Идентификатор источника (камеры/микрофона)."""
        ...

    def frames(self) -> Iterator[FrameItem]:
        """Видеокадры в порядке воспроизведения (может быть бесконечным при loop)."""
        ...

    def audio_windows(self) -> Iterator[AudioItem]:
        """Аудио-окна; пустой итератор, если у источника нет звука."""
        ...

    def close(self) -> None:
        """Освободить ресурсы (файлы, декодеры)."""
        ...


class NullAudioMixin:
    """Подмешивается адаптерам без звука: `audio_windows()` -> пусто."""

    def audio_windows(self) -> Iterator[AudioItem]:  # noqa: D102
        return iter(())


def sleep_to_keep_rate(period_s: float, last_emit_monotonic: float) -> float:
    """Уснуть так, чтобы события шли с заданным периодом; вернуть новый «момент»."""
    if period_s <= 0:
        return time.monotonic()
    now = time.monotonic()
    target = last_emit_monotonic + period_s
    if target > now:
        time.sleep(target - now)
        return target
    return now


def _unused(*_: Any) -> None:
    """Заглушка, чтобы линтер не ругался на параметры в скелетах-наследниках."""
