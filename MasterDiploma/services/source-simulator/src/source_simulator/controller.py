"""SimulatorController — связывает адаптер источника, (опц.) канал деградации и
gRPC-клиент. Тянет события из адаптера, выдерживает реальный темп (FPS видео /
hop аудио-окна), проставляет `ts` момента отправки и стримит в ingest-gateway.

Видео- и аудиопотоки идут параллельно (аудио — в отдельном потоке), если адаптер
отдаёт обе модальности. Видео и аудио стартуют почти одновременно — fusion-движок
выравнивает по ±ε. Останов — кооперативный по флагу (SIGINT/SIGTERM из __main__).
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

from uavdet_common.logging import get_logger

from .adapters.base import AudioItem, DataSourceAdapter, FrameItem
from .grpc_client import GrpcStreamClient

log = get_logger("source-simulator")


class SimulatorController:
    """Оркестратор имитатора источника (паттерн Template Method: read -> pace -> stream)."""

    def __init__(
        self,
        adapter: DataSourceAdapter,
        client: GrpcStreamClient,
        *,
        video_period_s: float,
        audio_period_s: float = 0.0,
        enable_audio: bool = False,
    ) -> None:
        self._adapter = adapter
        self._client = client
        self._video_period_s = max(0.0, video_period_s)
        self._audio_period_s = max(0.0, audio_period_s)
        self._enable_audio = enable_audio
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    # --- пейсинг ---
    def _paced_frames(self) -> Iterator[FrameItem]:
        next_emit = time.monotonic()
        for frame in self._adapter.frames():
            if self._stop.is_set():
                break
            if self._video_period_s > 0:
                now = time.monotonic()
                if next_emit > now:
                    time.sleep(next_emit - now)
                next_emit = max(next_emit + self._video_period_s, time.monotonic())
            yield FrameItem(
                jpeg_bytes=frame.jpeg_bytes, seq=frame.seq, width=frame.width, height=frame.height,
                fps_nominal=frame.fps_nominal, ts=time.time(), meta=frame.meta,
            )

    def _paced_audio(self) -> Iterator[AudioItem]:
        next_emit = time.monotonic()
        for win in self._adapter.audio_windows():
            if self._stop.is_set():
                break
            if self._audio_period_s > 0:
                now = time.monotonic()
                if next_emit > now:
                    time.sleep(next_emit - now)
                next_emit = max(next_emit + self._audio_period_s, time.monotonic())
            yield AudioItem(
                pcm_bytes=win.pcm_bytes, seq=win.seq, sample_rate=win.sample_rate, channels=win.channels,
                len_ms=win.len_ms, hop_ms=win.hop_ms, ts=time.time(), meta=win.meta,
            )

    # --- запуск ---
    def _run_audio_thread(self) -> None:
        source_id = self._adapter.source_id
        log.info("source-simulator: старт аудиопотока", source_id=source_id, period_s=round(self._audio_period_s, 4))
        try:
            ack = self._client.stream_audio(self._paced_audio(), source_id)
            log.info("source-simulator: аудиопоток завершён", accepted=ack.accepted, received=ack.received, message=ack.message)
        except Exception as exc:  # noqa: BLE001
            log.error("source-simulator: ошибка аудиопотока", error=str(exc))

    def run(self) -> None:
        """Проиграть потоки источника в gateway (блокирующий вызов)."""
        source_id = self._adapter.source_id
        audio_thread: threading.Thread | None = None
        if self._enable_audio:
            audio_thread = threading.Thread(target=self._run_audio_thread, name="sim-audio", daemon=True)
            audio_thread.start()

        log.info(
            "source-simulator: старт видеопотока",
            adapter=self._adapter.name, source_id=source_id, period_s=round(self._video_period_s, 4),
        )
        try:
            ack = self._client.stream_video(self._paced_frames(), source_id)
            log.info("source-simulator: видеопоток завершён", accepted=ack.accepted, received=ack.received, message=ack.message)
        except Exception as exc:  # noqa: BLE001
            log.error("source-simulator: ошибка видеопотока", error=str(exc))
            raise
        finally:
            self._stop.set()
            if audio_thread is not None:
                audio_thread.join(timeout=10.0)
            self._adapter.close()

    # обратная совместимость: старое имя
    def run_video(self) -> None:
        self.run()

    def close(self) -> None:
        self._adapter.close()
        self._client.close()
