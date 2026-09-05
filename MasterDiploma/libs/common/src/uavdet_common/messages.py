"""Схемы сообщений Kafka-топиков (pydantic v2) + имена топиков.

Все сообщения версионируются полем `schema_ver` (на пилоте — 1; новые поля только опциональные).
Ключ партиционирования во всех топиках — `source_id` (упорядоченность кадров/детекций одного источника).
Полезные данные кадров/аудио (`payload`) — base64-строка (для `payload_kind="jpeg"|"pcm"`) либо ссылка
на object storage (`payload_kind="ref"`, поле `payload_uri`) — claim-check pattern для тяжёлых данных.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VER = 1


# --- имена топиков (единственный источник правды) ---
class Topics:
    VIDEO_RAW = "video.raw"
    AUDIO_RAW = "audio.raw"
    INFERENCE = "inference"
    DECISIONS = "decisions"
    CONFIG_COMMANDS = "config.commands"


Modality = Literal["video", "audio"]
Label = Literal["drone", "non-drone"]
FusionMode = Literal["video-only", "audio-only", "late", "hybrid"]
PayloadKind = Literal["jpeg", "pcm", "wav", "ref"]


def _new_msg_id() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


class _Base(BaseModel):
    schema_ver: int = SCHEMA_VER
    msg_id: str = Field(default_factory=_new_msg_id)
    source_id: str
    ts: float = Field(default_factory=_now)  # момент события (unix-время, секунды)


# --- video.raw ---
class VideoMeta(BaseModel):
    w: int = 0
    h: int = 0
    fps_nominal: float = 0.0
    degradation: dict[str, float | int] | None = None


class VideoRawMsg(_Base):
    seq: int = 0
    media_ts: float | None = None      # it-34: позиция кадра на таймлайне исходного медиа (с); None — источник не отдал
    payload_kind: PayloadKind = "jpeg"
    payload: str | None = None        # base64 JPEG (если payload_kind в {jpeg})
    payload_uri: str | None = None    # ссылка на object storage (если payload_kind == "ref")
    meta: VideoMeta = Field(default_factory=VideoMeta)


# --- audio.raw ---
class AudioWindowSpec(BaseModel):
    len_ms: int = 0
    hop_ms: int = 0


class AudioMeta(BaseModel):
    sr: int = 0
    channels: int = 1
    snr_target: float | None = None


class AudioRawMsg(_Base):
    seq: int = 0
    media_ts: float | None = None      # it-34: позиция НАЧАЛА окна на медиатаймлайне (с); конец = media_ts + len_ms/1000
    window: AudioWindowSpec = Field(default_factory=AudioWindowSpec)
    payload_kind: PayloadKind = "pcm"
    payload: str | None = None
    payload_uri: str | None = None
    meta: AudioMeta = Field(default_factory=AudioMeta)


# --- inference (один топик с полем modality) ---
class ModelRef(BaseModel):
    name: str = "unknown"
    ver: str = "0"


class QualityHint(BaseModel):
    """Подсказки о качестве канала на момент детекции (best-effort; для adaptive gating).

    Все поля опциональны: детектор заполняет то, что может дёшево посчитать. Если поле
    None — gating игнорирует его и для этой модальности использует базовый вес.
    """

    img_sharpness: float | None = None    # резкость кадра (напр. дисперсия лапласиана), 0..∞ — больше = резче
    img_brightness: float | None = None   # средняя яркость кадра, 0..1
    snr_db: float | None = None            # отношение сигнал/шум аудио-окна, дБ
    audio_rms: float | None = None         # RMS окна (0..1 float32) — признак «тишина vs глухота» (research/it-16, it-19)
    motion_score: float | None = None      # it-52: нормированная скорость трека цели (0..1; 0 = статичен) — признак состояния «стоит vs летит» для fusion.target=airborne


class InferenceMsg(_Base):
    modality: Modality
    label: Label
    confidence: float
    p_drone: float | None = None          # вероятность класса «drone» (softmax), если бэкенд отдаёт (research/it-15, it-18); None — совместимость со старыми сообщениями
    media_ts: float | None = None         # it-34/35: событийное время на медиатаймлайне источника (с); None — источник не отдал (тогда ts = wall-clock доставки)
    bbox: list[float] | None = None       # [x, y, w, h] — только для modality=video
    track_id: int | None = None           # только для modality=video с трекером
    model: ModelRef = Field(default_factory=ModelRef)
    det_latency_ms: float = 0.0
    ingest_ts: float | None = None        # когда сообщение появилось в video.raw/audio.raw — для e2e latency
    detect_start_ts: float | None = None  # it-47: когда детектор начал обрабатывать сообщение (wall-clock) — разложение e2e: очередь до детектора
    detect_done_ts: float | None = None   # it-47: когда детектор опубликовал inference (wall-clock) — разложение e2e: очередь до fusion + ожидание окна/второй модальности
    quality: QualityHint = Field(default_factory=QualityHint)  # подсказки качества для adaptive gating


# --- decisions ---
class Contributions(BaseModel):
    p_v: float | None = None
    p_a: float | None = None
    w_v: float | None = None
    w_a: float | None = None
    delta: float = 0.0


class Gating(BaseModel):
    snr_audio: float | None = None
    img_quality: float | None = None


class DecisionMsg(_Base):
    ts_window: list[float]                # [t0, t1] окна выравнивания (в шкале выравнивания: media_ts, иначе wall-clock)
    media_ts: float | None = None         # it-35: событийное время триггер-сообщения на медиатаймлайне (GT-скоринг — по нему, не по ts)
    mode: FusionMode
    decision: bool                        # обнаружен БПЛА / нет
    p_fused: float
    contributions: Contributions = Field(default_factory=Contributions)
    gating: Gating = Field(default_factory=Gating)
    e2e_latency_ms: float = 0.0
    source_msg_ids: list[str] = Field(default_factory=list)


# --- config.commands (опц., compacted-топик) ---
class ConfigCommandMsg(BaseModel):
    schema_ver: int = SCHEMA_VER
    ts: float = Field(default_factory=_now)
    target: str                            # напр. "fusion"
    set: dict[str, object] = Field(default_factory=dict)


# карта "топик -> pydantic-модель" (используется десериализаторами)
TOPIC_MODELS: dict[str, type[BaseModel]] = {
    Topics.VIDEO_RAW: VideoRawMsg,
    Topics.AUDIO_RAW: AudioRawMsg,
    Topics.INFERENCE: InferenceMsg,
    Topics.DECISIONS: DecisionMsg,
    Topics.CONFIG_COMMANDS: ConfigCommandMsg,
}

__all__ = [
    "SCHEMA_VER",
    "Topics",
    "Modality",
    "Label",
    "FusionMode",
    "PayloadKind",
    "VideoMeta",
    "VideoRawMsg",
    "AudioWindowSpec",
    "AudioMeta",
    "AudioRawMsg",
    "ModelRef",
    "QualityHint",
    "InferenceMsg",
    "Contributions",
    "Gating",
    "DecisionMsg",
    "ConfigCommandMsg",
    "TOPIC_MODELS",
]
