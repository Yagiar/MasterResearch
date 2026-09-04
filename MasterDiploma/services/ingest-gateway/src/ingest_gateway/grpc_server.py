"""gRPC-сервер SourceStream: принимает стримы Frame/AudioWindow от источников,
нормализует и публикует в Kafka. По завершении стрима возвращает Ack.
"""

from __future__ import annotations

from concurrent import futures
from typing import Iterator

import grpc

from uavdet_common.logging import get_logger
from uavdet_common.metrics import DROPPED_TOTAL, MESSAGES_TOTAL

from uavdet_proto import ingest_pb2, ingest_pb2_grpc

from .normalizer import audio_to_audio_raw, frame_to_video_raw
from .publisher import Publisher

log = get_logger("ingest-gateway")

_SERVICE = "ingest-gateway"


class SourceStreamServicer(ingest_pb2_grpc.SourceStreamServicer):
    """Реализация сервиса SourceStream поверх Publisher."""

    def __init__(self, publisher: Publisher) -> None:
        self._publisher = publisher

    def StreamVideo(self, request_iterator: Iterator[ingest_pb2.Frame], context: grpc.ServicerContext) -> ingest_pb2.Ack:  # noqa: N802
        received = 0
        for frame in request_iterator:
            try:
                self._publisher.publish_video(frame_to_video_raw(frame))
                MESSAGES_TOTAL.labels(service=_SERVICE, topic="video.raw").inc()
                received += 1
            except Exception as exc:  # noqa: BLE001
                DROPPED_TOTAL.labels(service=_SERVICE, reason="normalize_or_publish").inc()
                log.error("ingest-gateway: ошибка обработки кадра", error=str(exc))
        self._publisher.flush()
        log.info("ingest-gateway: видеопоток завершён", received=received)
        return ingest_pb2.Ack(accepted=True, received=received, message="ok")

    def StreamAudio(self, request_iterator: Iterator[ingest_pb2.AudioWindow], context: grpc.ServicerContext) -> ingest_pb2.Ack:  # noqa: N802
        received = 0
        for window in request_iterator:
            try:
                self._publisher.publish_audio(audio_to_audio_raw(window))
                MESSAGES_TOTAL.labels(service=_SERVICE, topic="audio.raw").inc()
                received += 1
            except Exception as exc:  # noqa: BLE001
                DROPPED_TOTAL.labels(service=_SERVICE, reason="normalize_or_publish").inc()
                log.error("ingest-gateway: ошибка обработки аудио-окна", error=str(exc))
        self._publisher.flush()
        log.info("ingest-gateway: аудиопоток завершён", received=received)
        return ingest_pb2.Ack(accepted=True, received=received, message="ok")


def serve(
    publisher: Publisher,
    *,
    host: str,
    port: int,
    max_message_mb: int = 16,
    max_workers: int = 8,
) -> grpc.Server:
    """Создать и запустить gRPC-сервер; вернуть объект сервера (caller вызывает wait_for_termination)."""
    limit = max_message_mb * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", limit),
            ("grpc.max_receive_message_length", limit),
        ],
    )
    ingest_pb2_grpc.add_SourceStreamServicer_to_server(SourceStreamServicer(publisher), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    log.info("ingest-gateway: gRPC-сервер слушает", endpoint=f"{host}:{port}")
    return server
