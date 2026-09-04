"""Фоновый Kafka-потребитель топика `decisions` для дашборда.

Использует `KafkaConsumerService` из uavdet-common (Template Method): подписывается на
`decisions`, кладёт каждое решение в `DecisionStore` и пушит подписчикам WebSocket'а
(через callback). Запускается в отдельном потоке вместе с FastAPI-приложением.
"""

from __future__ import annotations

from typing import Any, Callable

from uavdet_common.consumer_service import KafkaConsumerService
from uavdet_common.messages import DecisionMsg, Topics

from .state import DecisionStore


class DecisionsFeedConsumer(KafkaConsumerService):
    """decisions -> DecisionStore + callback (для WS-broadcast)."""

    in_topic = Topics.DECISIONS
    in_model = DecisionMsg

    def __init__(
        self,
        bus,
        *,
        group_id: str,
        store: DecisionStore,
        on_decision: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(bus)
        self.group_id = group_id
        self._store = store
        self._on_decision = on_decision

    def process(self, key: str | None, msg: DecisionMsg) -> None:  # type: ignore[override]
        payload = msg.model_dump()
        self._store.add(payload)
        if self._on_decision is not None:
            try:
                self._on_decision(payload)
            except Exception:  # noqa: BLE001 - broadcast не должен ронять consumer
                self._log.warning("gateway: ошибка broadcast решения подписчикам WS")
