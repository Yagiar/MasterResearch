"""Publisher — публикация нормализованных сообщений в Kafka.

Тонкая обёртка над KafkaMessageBus + JsonSerializer: ключ партиционирования —
`source_id` (упорядоченность кадров/аудио одного источника).
"""

from __future__ import annotations

from uavdet_common.bus import KafkaMessageBus
from uavdet_common.messages import AudioRawMsg, Topics, VideoRawMsg
from uavdet_common.serialization import JsonSerializer


class Publisher:
    """Публикует VideoRawMsg/AudioRawMsg в топики video.raw / audio.raw."""

    def __init__(self, bus: KafkaMessageBus, serializer: JsonSerializer | None = None) -> None:
        self._bus = bus
        self._ser = serializer or JsonSerializer()

    def publish_video(self, msg: VideoRawMsg) -> None:
        self._bus.publish(Topics.VIDEO_RAW, msg.source_id, self._ser.encode(msg))

    def publish_audio(self, msg: AudioRawMsg) -> None:
        self._bus.publish(Topics.AUDIO_RAW, msg.source_id, self._ser.encode(msg))

    def flush(self, timeout: float = 5.0) -> None:
        self._bus.flush(timeout)

    def close(self) -> None:
        self._bus.close()
