"""Prometheus-метрики + HTTP /metrics эндпоинт (общая база для сервисов).

Использование (например в sink):
    from uavdet_common.metrics import start_metrics_server, MESSAGES_TOTAL, E2E_LATENCY
    start_metrics_server(9108)
    MESSAGES_TOTAL.labels(service="sink", topic="decisions").inc()
    E2E_LATENCY.labels(service="sink").observe(latency_ms / 1000.0)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# --- общие метрики (лейблы заполняет конкретный сервис) ---
MESSAGES_TOTAL = Counter(
    "uavdet_messages_total", "Обработано сообщений", ["service", "topic"]
)
PROCESS_ERRORS_TOTAL = Counter(
    "uavdet_process_errors_total", "Ошибок обработки сообщений", ["service"]
)
DROPPED_TOTAL = Counter(
    "uavdet_dropped_total", "Сообщений отброшено (бэкпрешер/деградация)", ["service", "reason"]
)
DETECT_LATENCY = Histogram(
    "uavdet_detect_latency_seconds", "Время инференса детектора", ["service"]
)
E2E_LATENCY = Histogram(
    "uavdet_e2e_latency_seconds", "End-to-end latency (ingest → decision)", ["service"]
)
CONSUMER_LAG = Gauge("uavdet_consumer_lag", "Отставание консьюмера (сообщений)", ["service", "topic"])
DELTA_T_MS = Histogram("uavdet_delta_t_ms", "Межмодальная задержка Δt (мс)", ["service"])

# счётчики качества (накапливаются sink-сервисом по ground truth, если он есть)
DECISIONS_TOTAL = Counter("uavdet_decisions_total", "Решений по классам", ["service", "decision"])

# it-33 (ревью §6.2): наблюдаемость политики опоздания/совместности окон
LATE_MESSAGES_TOTAL = Counter(
    "uavdet_late_messages_total", "Сообщений за горизонтом опоздания (порядок доставки)", ["service"]
)
JOINT_WINDOWS_TOTAL = Counter(
    "uavdet_joint_windows_total", "Окон выравнивания с обеими модальностями", ["service"]
)
# it-44 (ревью §6.2): почему окно выпущено — watermark (обе модальности закрыли интервал) или max_wait (mono-fallback)
WINDOW_RELEASES_TOTAL = Counter(
    "uavdet_window_releases_total", "Выпусков окон (watermark | max_wait | per-message)", ["service", "reason"]
)


def start_metrics_server(port: int) -> None:
    """Запустить HTTP-сервер Prometheus на /metrics (порт — из конфига сервиса)."""
    start_http_server(port)


__all__ = [
    "MESSAGES_TOTAL",
    "PROCESS_ERRORS_TOTAL",
    "DROPPED_TOTAL",
    "DETECT_LATENCY",
    "E2E_LATENCY",
    "CONSUMER_LAG",
    "DELTA_T_MS",
    "DECISIONS_TOTAL",
    "LATE_MESSAGES_TOTAL",
    "JOINT_WINDOWS_TOTAL",
    "WINDOW_RELEASES_TOTAL",
    "start_metrics_server",
]
