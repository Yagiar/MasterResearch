"""MetricsCollector — обновление Prometheus-метрик по входящим решениям.

На пилоте считает: число решений по классам, e2e-latency (если проставлен в DecisionMsg),
скользящую долю положительных решений. Метрики качества (precision/recall/F1) считаются
офлайн по jsonl + ground truth датасета — здесь экспонируется только то, что доступно онлайн.
"""

from __future__ import annotations

from uavdet_common.logging import get_logger
from uavdet_common.messages import DecisionMsg
from uavdet_common.metrics import DECISIONS_TOTAL, E2E_LATENCY, MESSAGES_TOTAL

_SERVICE = "sink"
log = get_logger("sink")


class MetricsCollector:
    """Тонкий апдейтер метрик + краткий лог по решению."""

    def __init__(self) -> None:
        self._n = 0
        self._n_drone = 0

    def observe_decision(self, decision: DecisionMsg) -> None:
        MESSAGES_TOTAL.labels(service=_SERVICE, topic="decisions").inc()
        cls = "drone" if decision.decision else "non-drone"
        DECISIONS_TOTAL.labels(service=_SERVICE, decision=cls).inc()
        if decision.e2e_latency_ms > 0:
            E2E_LATENCY.labels(service=_SERVICE).observe(decision.e2e_latency_ms / 1000.0)
        self._n += 1
        if decision.decision:
            self._n_drone += 1
        # негромкий лог раз в N решений, чтобы не засорять stdout
        if self._n % 50 == 1:
            log.info(
                "sink: решения",
                total=self._n,
                drone_ratio=round(self._n_drone / max(1, self._n), 3),
                last_p_fused=round(decision.p_fused, 3),
                last_mode=decision.mode,
            )
