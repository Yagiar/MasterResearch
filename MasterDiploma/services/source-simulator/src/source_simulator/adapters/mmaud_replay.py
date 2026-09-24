"""MmaudReplayAdapter — источник «replay секвенции из датасета MMAUD».

MMAUD (NTU, ICRA 2024) — единственный публичный датасет с РЕАЛЬНО синхронными аудио+видео
записями БПЛА (стерео-fisheye-камеры + 4-канальный аудио-массив + LiDAR/радар/Leica GT,
выровнено по ROS-таймстемпам). Для имитатора нам нужны только видео-секвенция, её аудио и
аннотации «есть дрон / какой / где» — LiDAR/радар не используются.

Раскладка конкретного Kaggle-зеркала может отличаться (mp4 + wav + csv ИЛИ извлечённые кадры
ИЛИ rosbag). Поэтому адаптер построен на двух «провайдерах»:
  - VIDEO: либо mp4-файл (cv2.VideoCapture), либо папка с кадрами `*.jpg/*.png` (по имени отсортированы);
  - AUDIO: wav-файл (один канал, если многоканальный — берётся `audio_channel`), нарезается на окна.
Конкретные пути к видео/аудио/аннотациям секвенции задаются в конфиге (`source.mmaud`),
поэтому адаптер не привязан жёстко к одной раскладке. Парсинг аннотаций MMAUD (для ground-truth)
— TODO: реализуется под фактический формат файла разметки (см. `load_ground_truth`).

Опционально адаптер умеет выгрузить ground-truth секвенции в jsonl (момент → есть дрон / класс /
bbox) — для офлайн-оценки решений fusion (`uavtrain.evaluate.evaluate_fusion_jsonl`, этап 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .base import AudioItem, FrameItem

_DEFAULT_FPS = 30.0           # MMAUD: камеры 30 Hz
_DEFAULT_JPEG_QUALITY = 80
_DEFAULT_SR = 16000           # целевой SR для аудио-окон (MMAUD-массив ~41.8 кГц — ресемплим)
_DEFAULT_WIN_MS = 1000
_DEFAULT_HOP_MS = 500
_FRAME_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class MmaudReplayAdapter:
    """Адаптер replay одной секвенции MMAUD (видео + аудио)."""

    name = "mmaud_replay"

    def __init__(
        self,
        *,
        source_id: str,
        video_path: str,                       # mp4-файл ИЛИ папка с кадрами секвенции
        audio_path: str | None = None,         # wav секвенции (если None — только видео)
        annotations_path: str | None = None,   # файл разметки секвенции (csv/json) — для ground-truth (TODO-парсинг)
        fps: float = _DEFAULT_FPS,
        loop: bool = True,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        audio_sample_rate: int = _DEFAULT_SR,
        audio_win_ms: int = _DEFAULT_WIN_MS,
        audio_hop_ms: int = _DEFAULT_HOP_MS,
        audio_channel: int = 0,                # какой канал брать из многоканального аудио MMAUD
    ) -> None:
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"видео секвенции MMAUD не найдено: {video_path}")
        self._source_id = source_id
        self._video_path = vp
        self._video_is_dir = vp.is_dir()
        self._audio_path = Path(audio_path) if audio_path else None
        if self._audio_path and not self._audio_path.exists():
            raise FileNotFoundError(f"аудио секвенции MMAUD не найдено: {audio_path}")
        self._annotations_path = Path(annotations_path) if annotations_path else None
        self._loop = loop
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self._requested_fps = float(fps)
        self._fps_nominal = _DEFAULT_FPS
        self._cap = None
        self._sr = int(audio_sample_rate)
        self._win_ms = int(audio_win_ms)
        self._hop_ms = int(audio_hop_ms)
        self._audio_channel = int(audio_channel)

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def has_audio(self) -> bool:
        return self._audio_path is not None

    @property
    def target_period_s(self) -> float:
        fps = self._requested_fps if self._requested_fps > 0 else self._fps_nominal
        return 1.0 / fps if fps > 0 else 0.0

    @property
    def audio_period_s(self) -> float:
        return self._hop_ms / 1000.0 if self._hop_ms > 0 else 0.0

    # --- видео ---
    def _frame_files(self) -> list[Path]:
        return sorted(p for p in self._video_path.iterdir() if p.suffix.lower() in _FRAME_EXT)

    def frames(self) -> Iterator[FrameItem]:
        import cv2  # noqa: PLC0415

        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        if self._video_is_dir:
            # it-83: у папки кадров собственной частоты нет — таймбейс медиа = темп отправки
            # (requested_fps); иначе media_ts идёт по номинальным 30 Hz при подаче, например,
            # 2 к/с и сжимает таймлайн ×15 (находка it-82: 240 с wall → 18,3 с media).
            # it-85: делитель — сам _fps_nominal без max(1.0, ·): кламп сжимал медиа-часы ×(1/fps)
            # при fps<1 (A@0.5: лаг +1 с/с, разрыв стыковки с окнами аудио в fusion ±ε).
            self._fps_nominal = self._requested_fps if self._requested_fps > 0 else _DEFAULT_FPS
            files = self._frame_files()
            if not files:
                raise RuntimeError(f"в папке кадров MMAUD нет изображений: {self._video_path}")
            seq = 0
            fi = 0            # индекс кадра в текущем loop-проходе (it-60: медиа-время)
            media_base = 0.0  # накопленное медиа-время завершённых проходов (с)
            while True:
                for f in files:
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    ok, buf = cv2.imencode(".jpg", img, params)
                    if not ok:
                        continue
                    h, w = img.shape[:2]
                    seq += 1
                    yield FrameItem(jpeg_bytes=buf.tobytes(), seq=seq, width=int(w), height=int(h),
                                    fps_nominal=float(self._fps_nominal),
                                    media_ts=media_base + fi / self._fps_nominal,
                                    meta={"source_kind": "mmaud", "src": f.name})
                    fi += 1
                media_base += len(files) / self._fps_nominal
                fi = 0
                if not self._loop:
                    return
        else:
            self._cap = cv2.VideoCapture(str(self._video_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"не удалось открыть видео MMAUD: {self._video_path}")
            ff = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps_nominal = ff if ff and ff > 0 else _DEFAULT_FPS
            seq = 0
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
                    yield FrameItem(jpeg_bytes=buf.tobytes(), seq=seq, width=int(w), height=int(h),
                                    fps_nominal=float(self._fps_nominal), meta={"source_kind": "mmaud"})
            finally:
                self.close()

    # --- аудио ---
    def _load_audio_mono(self) -> np.ndarray:
        import soundfile as sf  # noqa: PLC0415

        data, sr = sf.read(str(self._audio_path), dtype="float32", always_2d=True)  # [N, channels]
        ch = min(self._audio_channel, data.shape[1] - 1)
        sig = data[:, ch]
        if sr != self._sr:
            import librosa  # noqa: PLC0415

            sig = librosa.resample(sig.astype(np.float32), orig_sr=sr, target_sr=self._sr)
        return sig.astype(np.float32)

    @staticmethod
    def _float_to_pcm16(x: np.ndarray) -> bytes:
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def audio_windows(self) -> Iterator[AudioItem]:
        if self._audio_path is None:
            return
        sig = self._load_audio_mono()
        win = max(1, int(self._sr * self._win_ms / 1000.0))
        hop = max(1, int(self._sr * self._hop_ms / 1000.0))
        seq = 0
        media_base = 0.0  # накопленное медиа-время завершённых loop-проходов (it-60)
        while True:
            if sig.size < win:
                chunk = np.pad(sig, (0, win - sig.size), mode="constant")
                seq += 1
                yield AudioItem(pcm_bytes=self._float_to_pcm16(chunk), seq=seq, sample_rate=self._sr, channels=1,
                                len_ms=self._win_ms, hop_ms=self._hop_ms, media_ts=media_base,
                                meta={"source_kind": "mmaud"})
                if not self._loop:
                    return
                continue
            offset = 0
            while offset + win <= sig.size:
                seq += 1
                yield AudioItem(pcm_bytes=self._float_to_pcm16(sig[offset:offset + win]), seq=seq, sample_rate=self._sr,
                                channels=1, len_ms=self._win_ms, hop_ms=self._hop_ms,
                                media_ts=media_base + offset / self._sr,
                                meta={"source_kind": "mmaud"})
                offset += hop
            if not self._loop:
                return
            media_base += sig.size / self._sr

    # --- ground truth (TODO: реализовать под фактический формат разметки MMAUD) ---
    def load_ground_truth(self) -> list[dict]:
        """Прочитать аннотации секвенции и вернуть список записей вида
        {"t": <сек от старта>, "drone_present": bool, "drone_class": str|None, "bbox": [x,y,w,h]|None}.

        Заготовка: формат файла разметки MMAUD на конкретном зеркале нужно посмотреть и допилить
        парсинг (csv с колонками времени/позы/класса, либо json). До реализации — пустой список.
        """
        if self._annotations_path is None or not self._annotations_path.exists():
            return []
        raise NotImplementedError(
            "Парсинг аннотаций MMAUD не реализован — нужен фактический формат файла разметки "
            f"({self._annotations_path}); см. структуру датасета (kaggle datasets files ...)."
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None