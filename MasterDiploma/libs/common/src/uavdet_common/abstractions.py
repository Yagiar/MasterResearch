"""Ключевые интерфейсы системы (точки вариации спрятаны за узкими протоколами — ISP/DIP).

Соответствуют C4 Level 4 из заметки вольта "C4-диаграммы". Конкретные реализации — в сервисах:
- DataSource         → services/source-simulator/adapters/* (DatasetReplayAdapter, VideoFileAdapter, ...)
- DegradationStrategy → services/source-simulator/degradation/strategies.py
- Detector           → services/visual-detector/detector.py (YoloDetector), services/acoustic-detector/cnn.py
- FusionStrategy     → services/fusion/strategies/* (VideoOnly, AudioOnly, LateFusion, HybridFusion)
- GatingPolicy       → services/fusion/gating.py
- MessageBus         → uavdet_common.bus (KafkaMessageBus, InMemoryBus)
- Serializer         → uavdet_common.serialization (JsonSerializer)
- MetricsSink        → services/sink/metrics_collector.py (+ uavdet_common.metrics как база)
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

# Примечание: массивы кадров/аудио типизированы как `Any` (np.ndarray на практике) —
# uavdet-common не зависит от numpy, чтобы оставаться лёгким.


# --- источник данных ---
@runtime_checkable
class DataSource(Protocol):
    """Сменный источник сырых данных. publish_* отдаёт кадры/аудио-окна потребителю
    (на пилоте — gRPC-клиенту, который стримит их в ingest-gateway)."""

    def publish_video(self, frame: Any, ts: float, source_id: str, seq: int) -> None: ...

    def publish_audio(self, window: Any, ts: float, source_id: str, seq: int) -> None: ...

    def run(self) -> None:
        """Запустить воспроизведение/чтение источника (блокирующий цикл)."""
        ...


@runtime_checkable
class DegradationStrategy(Protocol):
    """Параметризуемое искажение потока (Δt-сдвиг, дроп кадров, шум, dropout модальности, jitter)."""

    name: str

    def apply(self, msg: Any) -> Any | None:
        """Вернуть (возможно изменённое) сообщение или None — если сообщение «дропнуто»."""
        ...


# --- детекторы ---
class Detection(Protocol):
    """Результат детекции (унифицированный для видео/аудио)."""

    label: str           # "drone" | "non-drone"
    confidence: float
    bbox: list[float] | None
    track_id: int | None


@runtime_checkable
class Detector(Protocol):
    """Детектор объекта по входу одной модальности."""

    model_name: str
    model_ver: str

    def detect(self, data: Any) -> Detection: ...


# --- слияние ---
class Decision(Protocol):
    decision: bool
    p_fused: float
    mode: str


@runtime_checkable
class FusionStrategy(Protocol):
    """Стратегия слияния детекций из временного окна (video-only / audio-only / late / hybrid)."""

    mode: str

    def fuse(self, window: Any) -> Decision | None: ...


@runtime_checkable
class GatingPolicy(Protocol):
    """Адаптивное взвешивание модальностей по качеству канала."""

    def weights(self, quality: Any) -> tuple[float, float]:
        """Вернуть (w_v, w_a), w_v + w_a == 1."""
        ...


# --- шина сообщений ---
@runtime_checkable
class MessageBus(Protocol):
    """Абстракция брокера сообщений (Kafka на проде, in-memory в тестах)."""

    def publish(self, topic: str, key: str | None, value: bytes) -> None: ...

    def subscribe(self, topic: str, group: str) -> Iterator[tuple[str | None, bytes]]:
        """Бесконечный итератор (key, value) по сообщениям топика для consumer group."""
        ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


# --- сериализация ---
@runtime_checkable
class Serializer(Protocol):
    def encode(self, obj: Any) -> bytes: ...

    def decode(self, raw: bytes, model: type | None = None) -> Any: ...


# --- сток метрик ---
@runtime_checkable
class MetricsSink(Protocol):
    """Приём событий/решений для агрегации метрик (F1, throughput, latency, lag, Δt)."""

    def record(self, event: Any) -> None: ...


__all__ = [
    "DataSource",
    "DegradationStrategy",
    "Detection",
    "Detector",
    "Decision",
    "FusionStrategy",
    "GatingPolicy",
    "MessageBus",
    "Serializer",
    "MetricsSink",
]
