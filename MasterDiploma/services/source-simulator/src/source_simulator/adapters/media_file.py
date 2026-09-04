"""MediaFileAdapter — источник «видеофайл + (опц.) аудиодорожка из wav».

Расширяет идею `DatasetReplayAdapter` второй модальностью: помимо покадрового видео
отдаёт аудио-окна фиксированной длины с заданным шагом (sliding window) из отдельного
wav-файла. Это даёт **синхронные** видео+аудио потоки для проверки мультимодального
fusion без специализированного датасета (видео и аудио стартуют с общего момента;
`ts` каждого события controller проставляет в момент отправки, fusion выравнивает по ±ε).

Если `audio_path` не задан — ведёт себя как видео-только источник (`audio_windows()` пуст).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .base import AudioItem, FrameItem

_DEFAULT_FPS = 25.0
_DEFAULT_JPEG_QUALITY = 75
_DEFAULT_WIN_MS = 1000
_DEFAULT_HOP_MS = 500
_DEFAULT_SR = 16000


class MediaFileAdapter:
    """Адаптер «видеофайл (+ опц. аудио из wav)»."""

    name = "media_file"

    def __init__(
        self,
        *,
        source_id: str,
        video_path: str,
        audio_path: str | None = None,
        fps: float = _DEFAULT_FPS,
        loop: bool = True,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        audio_sample_rate: int = _DEFAULT_SR,
        audio_win_ms: int = _DEFAULT_WIN_MS,
        audio_hop_ms: int = _DEFAULT_HOP_MS,
    ) -> None:
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"видеофайл источника не найден: {video_path}")
        self._source_id = source_id
        self._video_path = str(vp)
        self._audio_path = str(Path(audio_path)) if audio_path else None
        if self._audio_path and not Path(self._audio_path).exists():
            raise FileNotFoundError(f"аудиофайл источника не найден: {audio_path}")
        self._loop = loop
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self._requested_fps = float(fps)
        self._fps_nominal = _DEFAULT_FPS
        self._cap: cv2.VideoCapture | None = None
        self._sr = int(audio_sample_rate)
        self._win_ms = int(audio_win_ms)
        self._hop_ms = int(audio_hop_ms)

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def has_audio(self) -> bool:
        return self._audio_path is not None

    # --- темпы воспроизведения ---
    @property
    def target_period_s(self) -> float:
        """Период между видеокадрами."""
        fps = self._requested_fps if self._requested_fps > 0 else self._fps_nominal
        return 1.0 / fps if fps > 0 else 0.0

    @property
    def audio_period_s(self) -> float:
        """Период между аудио-окнами (= hop)."""
        return self._hop_ms / 1000.0 if self._hop_ms > 0 else 0.0

    # --- видео ---
    def _open_video(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(f"не удалось открыть видеофайл: {self._video_path}")
        file_fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps_nominal = file_fps if file_fps and file_fps > 0 else _DEFAULT_FPS
        return cap

    def frames(self) -> Iterator[FrameItem]:
        self._cap = self._open_video()
        seq = 0
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    if self._loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                ok_enc, buf = cv2.imencode(".jpg", frame, params)
                if not ok_enc:
                    continue
                h, w = frame.shape[:2]
                seq += 1
                yield FrameItem(
                    jpeg_bytes=buf.tobytes(), seq=seq, width=int(w), height=int(h),
                    fps_nominal=float(self._fps_nominal), meta={"source_kind": "media_file"},
                )
        finally:
            self.close()

    # --- аудио ---
    def _load_audio_mono(self) -> np.ndarray:
        """Загрузить wav как float32 моно на целевой sample_rate."""
        try:
            import soundfile as sf  # noqa: PLC0415

            data, sr = sf.read(self._audio_path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]
            if sr != self._sr:
                import librosa  # noqa: PLC0415

                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=self._sr)
            return data.astype(np.float32)
        except ImportError:  # soundfile/librosa недоступны — пробуем librosa напрямую
            import librosa  # noqa: PLC0415

            data, _ = librosa.load(self._audio_path, sr=self._sr, mono=True)
            return data.astype(np.float32)

    @staticmethod
    def _float_to_pcm16(x: np.ndarray) -> bytes:
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def audio_windows(self) -> Iterator[AudioItem]:
        if self._audio_path is None:
            return
        signal = self._load_audio_mono()
        win = max(1, int(self._sr * self._win_ms / 1000.0))
        hop = max(1, int(self._sr * self._hop_ms / 1000.0))
        seq = 0
        while True:
            if signal.size < win:
                # короткий файл — одно окно (паддинг нулями)
                chunk = np.pad(signal, (0, win - signal.size), mode="constant")
                seq += 1
                yield AudioItem(
                    pcm_bytes=self._float_to_pcm16(chunk), seq=seq, sample_rate=self._sr, channels=1,
                    len_ms=self._win_ms, hop_ms=self._hop_ms, meta={"source_kind": "media_file"},
                )
                if not self._loop:
                    return
                continue
            offset = 0
            while offset + win <= signal.size:
                seq += 1
                yield AudioItem(
                    pcm_bytes=self._float_to_pcm16(signal[offset : offset + win]),
                    seq=seq, sample_rate=self._sr, channels=1, len_ms=self._win_ms, hop_ms=self._hop_ms,
                    meta={"source_kind": "media_file"},
                )
                offset += hop
            if not self._loop:
                return

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
