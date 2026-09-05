"""Нормализатор: protobuf-сообщения от источника -> канонические сообщения Kafka.

Ставит `ingest_ts` (момент приёма gateway'ем) в `meta` через `degradation`-словарь?
Нет — `ingest_ts` нужен в `InferenceMsg` для e2e-latency, поэтому прокидываем его
в `meta` сырого сообщения отдельным ключом не получится (схема VideoMeta фиксирована),
а вместо этого detector берёт `ingest_ts` как момент, когда забрал сообщение из топика.
Здесь же — заполняем `schema_ver`/`msg_id`/`ts`/`seq` и упаковываем payload в base64.
"""

from __future__ import annotations

from uavdet_common.messages import (
    AudioMeta,
    AudioRawMsg,
    AudioWindowSpec,
    VideoMeta,
    VideoRawMsg,
)
from uavdet_common.serialization import b64encode_bytes

from uavdet_proto import ingest_pb2


def frame_to_video_raw(frame: ingest_pb2.Frame) -> VideoRawMsg:
    """protobuf Frame -> VideoRawMsg (payload — base64 JPEG)."""
    return VideoRawMsg(
        source_id=frame.source_id or "unknown",
        ts=frame.ts,
        seq=int(frame.seq),
        # it-34: позиция кадра на медиатаймлайне источника (0 = адаптер не отдал)
        media_ts=frame.media_ts if frame.media_ts else None,
        payload_kind="jpeg",
        payload=b64encode_bytes(frame.jpeg_bytes),
        meta=VideoMeta(
            w=int(frame.width),
            h=int(frame.height),
            fps_nominal=float(frame.fps_nominal),
        ),
    )


def audio_to_audio_raw(window: ingest_pb2.AudioWindow) -> AudioRawMsg:
    """protobuf AudioWindow -> AudioRawMsg (payload — base64 PCM)."""
    return AudioRawMsg(
        source_id=window.source_id or "unknown",
        ts=window.ts,
        seq=int(window.seq),
        media_ts=window.media_ts if window.media_ts else None,
        window=AudioWindowSpec(len_ms=int(window.len_ms), hop_ms=int(window.hop_ms)),
        payload_kind="pcm",
        payload=b64encode_bytes(window.pcm_bytes),
        meta=AudioMeta(sr=int(window.sample_rate), channels=int(window.channels)),
    )
