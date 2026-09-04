"""Стратегии деградации (паттерн Strategy). Каждая принимает событие потока
(`FrameItem`/`AudioItem`) и возвращает изменённое событие либо `None` («потеряно»).

Назначение: воспроизводимые стресс-сценарии для оценки устойчивости fusion —
рассинхрон модальностей (Δt), пропуски кадров, шум в аудио, отказ канала, дрожание
качества. Композируются `DegradationChannel`'ом (Decorator) в цепочку. На пилоте канал
выключен (`source.degradation.enabled: false`); включается для отдельных прогонов.

Реализованы все стратегии (детерминированы при заданном `seed`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..adapters.base import AudioItem, FrameItem

StreamItem = FrameItem | AudioItem


@runtime_checkable
class DegradationStrategy(Protocol):
    """Контракт стратегии деградации."""

    name: str

    def apply(self, item: StreamItem) -> StreamItem | None:
        """Вернуть изменённое событие или None (событие отброшено)."""
        ...


def _replace_ts(item: StreamItem, new_ts: float | None) -> StreamItem:
    """Вернуть копию события с другим `ts` (dataclass frozen → создаём новый)."""
    if isinstance(item, FrameItem):
        return FrameItem(
            jpeg_bytes=item.jpeg_bytes, seq=item.seq, width=item.width, height=item.height,
            fps_nominal=item.fps_nominal, ts=new_ts, meta=item.meta,
        )
    return AudioItem(
        pcm_bytes=item.pcm_bytes, seq=item.seq, sample_rate=item.sample_rate, channels=item.channels,
        len_ms=item.len_ms, hop_ms=item.hop_ms, ts=new_ts, meta=item.meta,
    )


@dataclass
class PassThrough:
    """No-op: событие проходит без изменений (нейтральный элемент цепочки)."""

    name: str = "pass_through"

    def apply(self, item: StreamItem) -> StreamItem | None:
        return item


@dataclass
class FrameDrop:
    """Случайный дроп видеокадров с вероятностью `drop_prob` (аудио не трогает)."""

    drop_prob: float = 0.0
    seed: int | None = None
    name: str = "frame_drop"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def apply(self, item: StreamItem) -> StreamItem | None:
        if isinstance(item, FrameItem) and self._rng.random() < self.drop_prob:
            return None
        return item


@dataclass
class WindowDrop:
    """Случайный дроп аудио-окон с вероятностью `drop_prob` (видео не трогает)."""

    drop_prob: float = 0.0
    seed: int | None = None
    name: str = "window_drop"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def apply(self, item: StreamItem) -> StreamItem | None:
        if isinstance(item, AudioItem) and self._rng.random() < self.drop_prob:
            return None
        return item


@dataclass
class DtShift:
    """Сдвиг временной метки одной модальности на фиксированную величину (рассинхрон Δt).

    `shift_ms` > 0 — «отстаёт», < 0 — «опережает». Применяется к указанной модальности
    (`target` = "video" | "audio"). Если у события `ts` ещё None (controller проставит позже) —
    сдвиг записывается в `meta["dt_shift_ms"]`, и финальный `ts` controller'а скорректировать
    нельзя задним числом → поэтому DtShift имеет смысл ставить ПОСЛЕ того, как ts уже проставлен
    (на пилоте `DegradationChannel` оборачивает адаптер до пейсинга — тогда работает только запись
    в meta как пометка; полноценный сдвиг ts включается, если канал применяется после controller'а).
    Для простоты на пилоте DtShift и сдвигает ts, если он задан, и всегда пишет пометку в meta.
    """

    shift_ms: float = 0.0
    target: str = "audio"
    name: str = "dt_shift"

    def apply(self, item: StreamItem) -> StreamItem | None:
        modality = "video" if isinstance(item, FrameItem) else "audio"
        if modality != self.target:
            return item
        new_meta = {**item.meta, "dt_shift_ms": str(self.shift_ms)}
        new_ts = (item.ts + self.shift_ms / 1000.0) if item.ts is not None else None
        shifted = _replace_ts(item, new_ts)
        # _replace_ts не копирует meta-изменения — пересоберём с new_meta
        if isinstance(shifted, FrameItem):
            return FrameItem(jpeg_bytes=shifted.jpeg_bytes, seq=shifted.seq, width=shifted.width,
                             height=shifted.height, fps_nominal=shifted.fps_nominal, ts=shifted.ts, meta=new_meta)
        return AudioItem(pcm_bytes=shifted.pcm_bytes, seq=shifted.seq, sample_rate=shifted.sample_rate,
                         channels=shifted.channels, len_ms=shifted.len_ms, hop_ms=shifted.hop_ms,
                         ts=shifted.ts, meta=new_meta)


@dataclass
class AudioNoise:
    """Добавление гауссова шума в PCM аудио-окна до целевого SNR (дБ).

    Уровень шума подбирается из энергии сигнала окна: `noise_std = signal_rms / 10^(snr/20)`.
    Видео не трогает. Метка `meta["snr_target"]` пишется для трассировки.
    """

    target_snr_db: float = 20.0
    seed: int | None = None
    name: str = "audio_noise"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def apply(self, item: StreamItem) -> StreamItem | None:
        if not isinstance(item, AudioItem):
            return item
        x = np.frombuffer(item.pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
        noise_std = (rms / (10.0 ** (self.target_snr_db / 20.0))) if rms > 0 else 1e-3
        noisy = x + self._rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        pcm = (np.clip(noisy, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return AudioItem(
            pcm_bytes=pcm, seq=item.seq, sample_rate=item.sample_rate, channels=item.channels,
            len_ms=item.len_ms, hop_ms=item.hop_ms, ts=item.ts,
            meta={**item.meta, "snr_target": f"{self.target_snr_db:.1f}"},
        )


@dataclass
class ModalityDropout:
    """Имитация отказа канала: на интервалах [start_s, end_s) события указанной модальности
    выпадают полностью. Интервалы задаются в секундах от старта потока (момент старта берётся
    из первого увиденного `ts`; если ts ещё None — отсчёт по числу обработанных событий нельзя
    привязать ко времени, поэтому такие события пропускаются как «вне интервалов»).
    """

    modality: str = "audio"
    intervals_s: list[tuple[float, float]] = field(default_factory=list)
    name: str = "modality_dropout"

    def __post_init__(self) -> None:
        self._t0: float | None = None

    def _in_drop(self, ts: float | None) -> bool:
        if ts is None:
            return False
        if self._t0 is None:
            self._t0 = ts
        rel = ts - self._t0
        return any(a <= rel < b for a, b in self.intervals_s)

    def apply(self, item: StreamItem) -> StreamItem | None:
        modality = "video" if isinstance(item, FrameItem) else "audio"
        if modality == self.modality and self._in_drop(item.ts):
            return None
        return item


@dataclass
class ConfidenceJitter:
    """Дрожание «качества» кадра: вносит лёгкое размытие в JPEG (через перекодирование с
    пониженным quality) с вероятностью `prob` — имитирует кратковременную потерю резкости.
    Влияет на `quality.img_sharpness`, который считает visual-detector. Аудио не трогает.
    """

    prob: float = 0.0
    low_quality: int = 25
    seed: int | None = None
    name: str = "confidence_jitter"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def apply(self, item: StreamItem) -> StreamItem | None:
        if not isinstance(item, FrameItem) or self._rng.random() >= self.prob:
            return item
        try:
            import cv2  # noqa: PLC0415

            buf = np.frombuffer(item.jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return item
            ok, re = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.low_quality)])
            if not ok:
                return item
            return FrameItem(jpeg_bytes=re.tobytes(), seq=item.seq, width=item.width, height=item.height,
                             fps_nominal=item.fps_nominal, ts=item.ts, meta={**item.meta, "jitter": "1"})
        except Exception:  # noqa: BLE001 - деградация не должна ронять поток
            return item


__all__ = [
    "DegradationStrategy",
    "PassThrough",
    "FrameDrop",
    "WindowDrop",
    "DtShift",
    "AudioNoise",
    "ModalityDropout",
    "ConfidenceJitter",
]
