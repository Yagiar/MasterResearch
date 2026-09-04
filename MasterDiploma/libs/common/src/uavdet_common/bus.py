"""Шина сообщений: KafkaMessageBus (confluent-kafka) и InMemoryBus (для тестов).

Обе реализуют протокол MessageBus из abstractions. KafkaMessageBus — at-least-once
(commit оффсета после успешной обработки сообщения вызывающим кодом — см. consumer_service).
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from typing import Iterator


class KafkaMessageBus:
    """Обёртка над confluent-kafka Producer/Consumer.

    Импорт confluent_kafka — ленивый (внутри методов), чтобы пакет ставился без брокера в тестах.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str | None = None,
        client_id: str = "uavdet",
        auto_offset_reset: str = "latest",
    ) -> None:
        self._bootstrap = bootstrap_servers
        self._group_id = group_id
        self._client_id = client_id
        self._auto_offset_reset = auto_offset_reset
        self._producer = None
        self._consumer = None

    # --- producer ---
    def _ensure_producer(self):
        if self._producer is None:
            from confluent_kafka import Producer

            self._producer = Producer(
                {
                    "bootstrap.servers": self._bootstrap,
                    "client.id": self._client_id,
                    "enable.idempotence": True,
                    "acks": "all",
                }
            )
        return self._producer

    def publish(self, topic: str, key: str | None, value: bytes) -> None:
        p = self._ensure_producer()
        p.produce(topic, key=key.encode("utf-8") if key else None, value=value)
        p.poll(0)

    def flush(self, timeout: float = 5.0) -> None:
        if self._producer is not None:
            self._producer.flush(timeout)

    # --- consumer ---
    def subscribe(self, topic: str, group: str) -> Iterator[tuple[str | None, bytes]]:
        from confluent_kafka import Consumer

        self._consumer = Consumer(
            {
                "bootstrap.servers": self._bootstrap,
                "group.id": group,
                "client.id": self._client_id,
                "auto.offset.reset": self._auto_offset_reset,
                "enable.auto.commit": False,  # коммитим вручную после обработки
            }
        )
        self._consumer.subscribe([topic])
        while True:
            msg = self._consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                # пропускаем ошибочные сообщения (логирование — на стороне consumer_service)
                continue
            key = msg.key().decode("utf-8") if msg.key() else None
            yield key, msg.value()

    def commit(self) -> None:
        if self._consumer is not None:
            self._consumer.commit(asynchronous=False)

    def close(self) -> None:
        self.flush()
        if self._consumer is not None:
            self._consumer.close()


class InMemoryBus:
    """In-memory шина для тестов и самого раннего пилота (один процесс).

    Топики — очереди; subscribe возвращает блокирующий итератор. Партиционирование/группы
    не моделируются (один потребитель на топик).
    """

    _topics: dict[str, "queue.Queue[tuple[str | None, bytes]]"]

    def __init__(self) -> None:
        self._topics = defaultdict(queue.Queue)
        self._lock = threading.Lock()

    def publish(self, topic: str, key: str | None, value: bytes) -> None:
        with self._lock:
            self._topics[topic].put((key, value))

    def subscribe(self, topic: str, group: str) -> Iterator[tuple[str | None, bytes]]:
        with self._lock:
            q = self._topics[topic]
        while True:
            yield q.get()

    def commit(self) -> None:  # noqa: D401 - no-op
        return None

    def close(self) -> None:
        return None

    # тестовый хелпер
    def drain(self, topic: str) -> list[tuple[str | None, bytes]]:
        out: list[tuple[str | None, bytes]] = []
        q = self._topics.get(topic)
        if q is None:
            return out
        while not q.empty():
            out.append(q.get_nowait())
        return out


__all__ = ["KafkaMessageBus", "InMemoryBus"]
