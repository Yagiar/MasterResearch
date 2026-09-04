"""VideoFileAdapter — адаптер произвольного видеофайла без датасет-разметки.

Заготовка: на MVP его роль закрывает DatasetReplayAdapter. Здесь предполагается
поддержка нескольких файлов в плейлисте, выбора кодека, ресемплинга FPS и т.п.
"""

from __future__ import annotations

from typing import Iterator

from .base import AudioItem, DataSourceAdapter, FrameItem


class VideoFileAdapter:
    """Заготовка адаптера видеофайла (не реализована на MVP)."""

    name = "video_file"

    def __init__(self, *, source_id: str, **_: object) -> None:
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def frames(self) -> Iterator[FrameItem]:
        raise NotImplementedError("VideoFileAdapter будет реализован вне MVP")

    def audio_windows(self) -> Iterator[AudioItem]:
        return iter(())

    def close(self) -> None:
        return None
