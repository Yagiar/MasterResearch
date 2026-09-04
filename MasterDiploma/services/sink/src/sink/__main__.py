"""Точка входа сервиса sink.

Потребляет ДВА топика: `decisions` (лог + jsonl + метрики + PG) и `inference` (PG).
Каждый consumer крутится в своём потоке (один `KafkaConsumerService` = один топик);
PostgreSQL-хранилище (`PgResultStore`) общее. Graceful shutdown — по SIGINT/SIGTERM
из главного потока (выставляет флаг остановки обоим consumer'ам).

Запуск:
    python -m sink
    python -m sink --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from uavdet_common.bus import KafkaMessageBus
from uavdet_common.config import load_config
from uavdet_common.db import PgResultStore, make_dsn
from uavdet_common.logging import configure_logging, get_logger
from uavdet_common.metrics import start_metrics_server

from .consumer import DecisionsConsumer, InferenceRecorder
from .metrics_collector import MetricsCollector
from .store import JsonlStore


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sink")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    return parser.parse_args(argv)


def _build_pg_store(pg_cfg: dict) -> PgResultStore | None:
    if not pg_cfg or not pg_cfg.get("enabled", False):
        return None
    dsn = pg_cfg.get("dsn") or make_dsn(
        host=str(pg_cfg.get("host", "postgres")),
        port=int(pg_cfg.get("port", 5432)),
        dbname=str(pg_cfg.get("dbname", "uavdet")),
        user=str(pg_cfg.get("user", "uavdet")),
        password=str(pg_cfg.get("password", "uavdet")),
    )
    return PgResultStore(dsn)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("sink")

    kafka_cfg = dict(cfg.get("kafka", {}))
    sink_cfg = dict(cfg.get("sink", {}))
    pg_cfg = dict(cfg.get("postgres", {}))
    bootstrap = str(kafka_cfg.get("bootstrap_servers", "kafka:9092"))
    auto_offset_reset = str(sink_cfg.get("auto_offset_reset", "latest"))

    metrics_port = int(sink_cfg.get("metrics_port", 9108))
    if metrics_port > 0:
        start_metrics_server(metrics_port)
        log.info("sink: метрики на /metrics", port=metrics_port)

    pg_store = _build_pg_store(pg_cfg)
    if pg_store is not None:
        log.info("sink: запись результатов в PostgreSQL включена")

    decisions_log = sink_cfg.get("decisions_log")
    jsonl_store = JsonlStore(decisions_log) if decisions_log else None

    # отдельная шина на каждый consumer (у KafkaMessageBus один consumer внутри)
    decisions_bus = KafkaMessageBus(bootstrap_servers=bootstrap, client_id="sink-decisions", auto_offset_reset=auto_offset_reset)
    decisions_consumer = DecisionsConsumer(
        decisions_bus,
        group_id=str(sink_cfg.get("group_id", "sink")),
        store=jsonl_store,
        pg_store=pg_store,
        metrics=MetricsCollector(),
    )

    threads: list[threading.Thread] = []
    consumers = [decisions_consumer]

    if bool(sink_cfg.get("also_consume_inference", True)):
        inference_bus = KafkaMessageBus(bootstrap_servers=bootstrap, client_id="sink-inference", auto_offset_reset=auto_offset_reset)
        inference_recorder = InferenceRecorder(
            inference_bus,
            group_id=str(sink_cfg.get("inference_group_id", "sink-inference")),
            pg_store=pg_store,
        )
        consumers.append(inference_recorder)

    for c in consumers:
        t = threading.Thread(target=c.run, name=f"sink-{c.in_topic}", daemon=True)
        t.start()
        threads.append(t)

    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("sink: получен сигнал, останавливаюсь", signal=signum)
        for c in consumers:
            c._stop = True  # noqa: SLF001 - кооперативная остановка consumer'ов из главного потока
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        # ждём, пока не придёт сигнал или все consumer-потоки не завершатся сами
        while not stop_event.is_set() and any(t.is_alive() for t in threads):
            stop_event.wait(1.0)
        return 0
    finally:
        for t in threads:
            t.join(timeout=10.0)
        if pg_store is not None:
            pg_store.close()
        log.info("sink: остановлен")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
