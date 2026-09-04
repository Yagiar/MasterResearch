"""GrpcStreamClient — gRPC-клиент для канала «источник -> ingest-gateway».

Преобразует события адаптера (FrameItem/AudioItem) в protobuf-сообщения
`uavdet.ingest.v1.Frame` / `AudioWindow` и стримит их в gateway методами
`StreamVideo` / `StreamAudio`. По завершении стрима возвращает `Ack`.
"""

from __future__ import annotations

import time
from typing import Iterable, Iterator

import grpc

from uavdet_proto import ingest_pb2, ingest_pb2_grpc

from .adapters.base import AudioItem, FrameItem

_DEFAULT_MAX_MESSAGE_MB = 16


class GrpcStreamClient:
    """Тонкий клиент SourceStream-сервиса ingest-gateway."""

    def __init__(self, host: str, port: int, *, max_message_mb: int = _DEFAULT_MAX_MESSAGE_MB) -> None:
        limit = max_message_mb * 1024 * 1024
        options = [
            ("grpc.max_send_message_length", limit),
            ("grpc.max_receive_message_length", limit),
        ]
        self._target = f"{host}:{port}"
        self._channel = grpc.insecure_channel(self._target, options=options)
        self._stub = ingest_pb2_grpc.SourceStreamStub(self._channel)

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        """Дождаться готовности канала к gateway (на старте — gateway может подниматься)."""
        grpc.channel_ready_future(self._channel).result(timeout=timeout_s)

    @staticmethod
    def _to_frame_pb(item: FrameItem, source_id: str) -> ingest_pb2.Frame:
        return ingest_pb2.Frame(
            jpeg_bytes=item.jpeg_bytes,
            ts=item.ts if item.ts is not None else time.time(),
            seq=item.seq,
            source_id=source_id,
            width=item.width,
            height=item.height,
            fps_nominal=item.fps_nominal,
            meta=dict(item.meta),
        )

    @staticmethod
    def _to_audio_pb(item: AudioItem, source_id: str) -> ingest_pb2.AudioWindow:
        return ingest_pb2.AudioWindow(
            pcm_bytes=item.pcm_bytes,
            ts=item.ts if item.ts is not None else time.time(),
            seq=item.seq,
            source_id=source_id,
            sample_rate=item.sample_rate,
            channels=item.channels,
            len_ms=item.len_ms,
            hop_ms=item.hop_ms,
            meta=dict(item.meta),
        )

    def stream_video(self, frames: Iterable[FrameItem], source_id: str) -> ingest_pb2.Ack:
        """Отправить поток видеокадров; вернуть Ack по завершении."""

        def gen() -> Iterator[ingest_pb2.Frame]:
            for frame in frames:
                yield self._to_frame_pb(frame, source_id)

        return self._stub.StreamVideo(gen())

    def stream_audio(self, windows: Iterable[AudioItem], source_id: str) -> ingest_pb2.Ack:
        """Отправить поток аудио-окон; вернуть Ack по завершении."""

        def gen() -> Iterator[ingest_pb2.AudioWindow]:
            for window in windows:
                yield self._to_audio_pb(window, source_id)

        return self._stub.StreamAudio(gen())

    def close(self) -> None:
        self._channel.close()
