"""MediaFileAdapter — источник «видеофайл + (опц.) аудиодорожка из wav».

Расширяет идею `DatasetReplayAdapter` второй модальностью: помимо покадрового видео
отдаёт аудио-окна фиксированной длины с заданным шагом (sliding window) из отдельного
wav-файла. Это даёт **синхронные** видео+аудио потоки для проверки мультимодального
fusion без специализированного датасета.

Синхронизация (it-34, ревью §6.1): обе модальности несут `media_ts` — позицию на
ОБЩЕМ таймлайне исходного медиа (с от старта источника, монотонно с учётом loop).
Видео отдаётся с шагом-страйдом, чтобы содержание шло со скоростью запрошенного FPS
(иначе файл 25 FPS при отдаче 5 FPS растягивался бы в 5 раз относительно аудио).
`ts` остаётся wall-clock моментом отправки (controller); fusion-выравнивание и GT-скоринг
должны опираться на `media_ts`, а не на wall-clock (it-35).

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
        # it-47: аудио грузится В КОНСТРУКТОРЕ, а не лениво в аудио-потоке — иначе видео
        # стримит, пока librosa ресемплирует wav (~36 с на sandbox), и медиа-часы аудио
        # отстают от видео на время загрузки (постоянный сдвиг модальностей)
        self._signal = None
        if self._audio_path:
            self._signal = self._load_audio_mono()

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

    def _frame_stride(self) -> int:
        """Шаг чтения кадров, чтобы СОДЕРЖИМОЕ шло с запрошенной скоростью (it-34, ревью §6.1).

        Раньше читался каждый кадр, а отправка шла по запрошенному FPS: файл 25 FPS при
        отдаче 5 FPS растягивал видео в 5 раз, а аудио шло своим шагом — модальности
        рассинхронизировались. Теперь отдаём каждый N-й кадр (N = file_fps / requested_fps):
        медиа-время кадра advance = N/file_fps ≈ 1/requested_fps — общий таймлайн с аудио.
        """
        if self._requested_fps <= 0 or self._fps_nominal <= 0:
            return 1
        return max(1, round(self._fps_nominal / self._requested_fps))

    def frames(self) -> Iterator[FrameItem]:
        self._cap = self._open_video()
        seq = 0
        fi = 0                # индекс кадра в текущем loop-проходе
        media_base = 0.0      # накопленное медиа-время завершённых проходов (с)
        stride = self._frame_stride()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    if self._loop:
                        media_base += fi / self._fps_nominal  # длительность прошедшего прохода
                        fi = 0
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                if fi % stride == 0:
                    ok_enc, buf = cv2.imencode(".jpg", frame, params)
                    if ok_enc:
                        h, w = frame.shape[:2]
                        seq += 1
                        yield FrameItem(
                            jpeg_bytes=buf.tobytes(), seq=seq, width=int(w), height=int(h),
                            fps_nominal=float(self._fps_nominal),
                            media_ts=media_base + fi / self._fps_nominal,
                            meta={"source_kind": "media_file"},
                        )
                fi += 1
        finally:
            self.close()

    # --- аудио ---
    def _load_audio_mono(self) -> np.ndarray:
        """Загрузить wav как float32 моно на целевой sample_rate (вызывается из __init__, it-47)."""
        if self._signal is not None:  # уже предзагружено в конструкторе
            return self._signal
        try:  # noqa: SIM105
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
        signal = self._signal  # предзагружено в __init__ (it-47)
        win = max(1, int(self._sr * self._win_ms / 1000.0))
        hop = max(1, int(self._sr * self._hop_ms / 1000.0))
        seq = 0
        media_base = 0.0   # накопленное медиа-время завершённых loop-проходов (с)
        while True:
            if signal.size < win:
                # короткий файл — одно окно (паддинг нулями)
                chunk = np.pad(signal, (0, win - signal.size), mode="constant")
                seq += 1
                yield AudioItem(
                    pcm_bytes=self._float_to_pcm16(chunk), seq=seq, sample_rate=self._sr, channels=1,
                    len_ms=self._win_ms, hop_ms=self._hop_ms,
                    media_ts=media_base, meta={"source_kind": "media_file"},
                )
                if not self._loop:
                    return
                media_base += signal.size / self._sr
                continue
            offset = 0
            while offset + win <= signal.size:
                seq += 1
                yield AudioItem(
                    pcm_bytes=self._float_to_pcm16(signal[offset : offset + win]),
                    seq=seq, sample_rate=self._sr, channels=1, len_ms=self._win_ms, hop_ms=self._hop_ms,
                    media_ts=media_base + offset / self._sr,
                    meta={"source_kind": "media_file"},
                )
                offset += hop
            if not self._loop:
                return
            media_base += signal.size / self._sr

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
