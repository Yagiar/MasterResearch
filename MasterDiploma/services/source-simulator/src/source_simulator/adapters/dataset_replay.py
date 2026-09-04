"""DatasetReplayAdapter — проигрывает видеофайл (replay из публичного датасета)
покадрово, кодируя кадры в JPEG. MVP-источник для сквозного пайплайна.

Расширение (вне MVP): чтение разметки датасета и склейка кадр+аудиоклип
по событию пролёта; пока — только видео.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2  # opencv-python-headless

from .base import DataSourceAdapter, FrameItem, NullAudioMixin

_DEFAULT_FPS = 25.0
_DEFAULT_JPEG_QUALITY = 75


class DatasetReplayAdapter(NullAudioMixin):
    """Адаптер «replay одного видеофайла».

    Параметры (из configs.pilot.yaml -> source.dataset_replay):
      - video_path: путь к видеофайлу;
      - fps: целевой темп воспроизведения (если 0 — берётся FPS из файла);
      - loop: проигрывать по кругу;
      - jpeg_quality: качество JPEG-кодирования кадров (1..100).
    """

    name = "dataset_replay"

    def __init__(
        self,
        *,
        source_id: str,
        video_path: str,
        fps: float = _DEFAULT_FPS,
        loop: bool = True,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
    ) -> None:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"видеофайл источника не найден: {video_path}")
        self._source_id = source_id
        self._video_path = str(path)
        self._loop = loop
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self._requested_fps = float(fps)
        self._cap: cv2.VideoCapture | None = None
        self._fps_nominal = _DEFAULT_FPS

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def target_period_s(self) -> float:
        """Период между кадрами для выдерживания темпа воспроизведения."""
        fps = self._requested_fps if self._requested_fps > 0 else self._fps_nominal
        return 1.0 / fps if fps > 0 else 0.0

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(f"не удалось открыть видеофайл: {self._video_path}")
        file_fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps_nominal = file_fps if file_fps and file_fps > 0 else _DEFAULT_FPS
        return cap

    def frames(self) -> Iterator[FrameItem]:
        self._cap = self._open()
        seq = 0
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    if self._loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                ok_enc, buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok_enc:
                    continue
                height, width = frame.shape[:2]
                seq += 1
                yield FrameItem(
                    jpeg_bytes=buf.tobytes(),
                    seq=seq,
                    width=int(width),
                    height=int(height),
                    fps_nominal=float(self._fps_nominal),
                    meta={"source_kind": "dataset_replay"},
                )
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
