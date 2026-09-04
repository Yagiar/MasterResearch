"""SyntheticAdapter — генератор синтетического потока (движущаяся «мишень» на
фоне + опц. синтез звука винтов). Заготовка: на MVP не используется.

Назначение (вне MVP): воспроизводимые стресс-сценарии без датасетов —
управляемые SNR, частота кадров, размер цели, искусственные пропуски.
"""

from __future__ import annotations

from typing import Iterator

from .base import AudioItem, FrameItem


class SyntheticAdapter:
    """Заготовка синтетического источника (не реализована на MVP)."""

    name = "synthetic"

    def __init__(self, *, source_id: str, **_: object) -> None:
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def frames(self) -> Iterator[FrameItem]:
        raise NotImplementedError("SyntheticAdapter будет реализован вне MVP")

    def audio_windows(self) -> Iterator[AudioItem]:
        raise NotImplementedError("SyntheticAdapter будет реализован вне MVP")

    def close(self) -> None:
        return None
