"""Consumer-сервисы sink.

`DecisionsConsumer` — топик `decisions` -> структурный лог + JsonlStore (офлайн-анализ)
+ MetricsCollector (Prometheus) + (опц.) PgResultStore (таблица uavdet.decisions).
`InferenceRecorder` — топик `inference` -> (опц.) PgResultStore (таблица uavdet.inference)
+ счётчик. На MVP оба запускаются в отдельных потоках (см. __main__).
"""

from __future__ import annotations

from uavdet_common.consumer_service import KafkaConsumerService
from uavdet_common.messages import DecisionMsg, InferenceMsg, Topics
from uavdet_common.metrics import MESSAGES_TOTAL

from .metrics_collector import MetricsCollector
from .store import JsonlStore

_SERVICE = "sink"


class DecisionsConsumer(KafkaConsumerService):
    """decisions -> лог + jsonl + метрики + (опц.) PostgreSQL."""

    in_topic = Topics.DECISIONS
    in_model = DecisionMsg

    def __init__(
        self,
        bus,
        *,
        group_id: str,
        store: JsonlStore | None,
        pg_store=None,                       # uavdet_common.db.PgResultStore | None
        metrics: MetricsCollector | None = None,
    ) -> None:
        super().__init__(bus)
        self.group_id = group_id
        self._store = store
        self._pg = pg_store
        self._metrics = metrics or MetricsCollector()

    def on_start(self) -> None:
        self._log.info("sink/decisions: старт", topic=self.in_topic, jsonl=bool(self._store), postgres=bool(self._pg))

    def on_stop(self) -> None:
        if self._store is not None:
            self._store.close()

    def process(self, key: str | None, msg: DecisionMsg) -> None:  # type: ignore[override]
        self._metrics.observe_decision(msg)
        if self._store is not None:
            self._store.write(msg)
        if self._pg is not None:
            self._pg.write_decision(msg)
        self._log.debug(
            "sink/decisions: решение",
            source_id=msg.source_id,
            decision=msg.decision,
            p_fused=round(msg.p_fused, 3),
            mode=msg.mode,
            e2e_ms=round(msg.e2e_latency_ms, 1),
        )


class InferenceRecorder(KafkaConsumerService):
    """inference -> (опц.) PostgreSQL (таблица uavdet.inference) + счётчик."""

    in_topic = Topics.INFERENCE
    in_model = InferenceMsg

    def __init__(self, bus, *, group_id: str, pg_store=None) -> None:
        super().__init__(bus)
        self.group_id = group_id
        self._pg = pg_store
        self._n = 0

    def on_start(self) -> None:
        self._log.info("sink/inference: старт", topic=self.in_topic, postgres=bool(self._pg))

    def process(self, key: str | None, msg: InferenceMsg) -> None:  # type: ignore[override]
        if self._pg is not None:
            self._pg.write_inference(msg)
        MESSAGES_TOTAL.labels(service=_SERVICE, topic=Topics.INFERENCE).inc()
        self._n += 1
        if self._n % 200 == 1:
            self._log.info("sink/inference: записано", total=self._n, last_modality=msg.modality, last_label=msg.label)
