"""KafkaConsumerService — базовый каркас consumer-сервиса (паттерн Template Method).

run() выполняет фиксированный цикл: poll → deserialize → process(abstract) → produce → commit,
с graceful shutdown (SIGINT/SIGTERM), идемпотентностью по msg_id (LRU виденных id) и обработкой ошибок.
Конкретные сервисы (visual-detector, fusion, sink, ...) переопределяют process().
"""

from __future__ import annotations

import signal
from collections import OrderedDict
from typing import Any

from .abstractions import MessageBus
from .logging import get_logger
from .serialization import JsonSerializer


class _LruSet:
    """Маленькое множество последних N виденных msg_id (для дедупликации at-least-once)."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._cap = capacity
        self._d: "OrderedDict[str, None]" = OrderedDict()

    def add(self, key: str) -> bool:
        """Вернуть True, если key новый (добавлен); False — если уже виден."""
        if key in self._d:
            self._d.move_to_end(key)
            return False
        self._d[key] = None
        if len(self._d) > self._cap:
            self._d.popitem(last=False)
        return True


class KafkaConsumerService:
    """Базовый consumer-сервис. Подклассы задают: in_topic, group_id, in_model, реализуют process()."""

    in_topic: str = ""
    group_id: str = ""
    in_model: type | None = None  # pydantic-модель входного сообщения (для десериализации)

    def __init__(self, bus: MessageBus, serializer: JsonSerializer | None = None) -> None:
        self._bus = bus
        self._ser = serializer or JsonSerializer()
        self._log = get_logger(self.__class__.__name__)
        self._stop = False
        self._seen = _LruSet()

    # --- управление жизненным циклом ---
    def _install_signals(self) -> None:
        def _handler(signum, _frame):  # noqa: ANN001
            self._log.info("получен сигнал, останавливаюсь", signum=signum)
            self._stop = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except ValueError:
                pass  # не главный поток — пропускаем

    def run(self) -> None:
        """Template Method: основной цикл потребления."""
        self._install_signals()
        self._log.info("старт сервиса", topic=self.in_topic, group=self.group_id)
        self.on_start()
        try:
            for key, raw in self._bus.subscribe(self.in_topic, self.group_id):
                if self._stop:
                    break
                try:
                    msg = self._deserialize(raw)
                    msg_id = getattr(msg, "msg_id", None)
                    if msg_id is not None and not self._seen.add(msg_id):
                        # дубликат — пропускаем (идемпотентность)
                        self._bus.commit()
                        continue
                    self.process(key, msg)
                    self._bus.commit()
                except Exception:  # noqa: BLE001
                    self._log.exception("ошибка обработки сообщения; коммит, чтобы не зациклиться")
                    self._bus.commit()
        finally:
            self.on_stop()
            self._bus.close()
            self._log.info("сервис остановлен")

    # --- хуки для подклассов ---
    def on_start(self) -> None:  # noqa: D401 - default no-op
        return None

    def on_stop(self) -> None:
        return None

    def process(self, key: str | None, msg: Any) -> None:
        """Обработать одно входное сообщение. Реализуется подклассом.

        Для produce — использовать self.publish(topic, key, model)."""
        raise NotImplementedError

    # --- утилиты ---
    def _deserialize(self, raw: bytes) -> Any:
        return self._ser.decode(raw, self.in_model) if self.in_model else self._ser.decode(raw)

    def publish(self, topic: str, key: str | None, model: Any) -> None:
        self._bus.publish(topic, key, self._ser.encode(model))


__all__ = ["KafkaConsumerService"]
