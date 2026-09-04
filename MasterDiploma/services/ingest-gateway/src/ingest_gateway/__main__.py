"""Точка входа сервиса ingest-gateway.

Запуск:
    python -m ingest_gateway
    python -m ingest_gateway --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from uavdet_common.bus import KafkaMessageBus
from uavdet_common.config import load_config
from uavdet_common.logging import configure_logging, get_logger
from uavdet_common.metrics import start_metrics_server

from .grpc_server import serve
from .publisher import Publisher


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ingest-gateway")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("ingest-gateway")

    kafka_cfg = dict(cfg.get("kafka", {}))
    grpc_cfg = dict(cfg.get("ingest_grpc", {}))

    bus = KafkaMessageBus(
        bootstrap_servers=str(kafka_cfg.get("bootstrap_servers", "kafka:9092")),
        client_id="ingest-gateway",
    )
    publisher = Publisher(bus)

    metrics_port = int(grpc_cfg.get("metrics_port", 0))
    if metrics_port > 0:
        start_metrics_server(metrics_port)
        log.info("ingest-gateway: метрики на /metrics", port=metrics_port)

    server = serve(
        publisher,
        host=str(grpc_cfg.get("bind_host", "0.0.0.0")),
        port=int(grpc_cfg.get("port", 50051)),
        max_message_mb=int(grpc_cfg.get("max_message_mb", 16)),
        max_workers=int(grpc_cfg.get("max_workers", 8)),
    )

    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("ingest-gateway: получен сигнал, останавливаюсь", signal=signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        stop_event.wait()
        return 0
    finally:
        server.stop(grace=5.0)
        publisher.close()
        log.info("ingest-gateway: остановлен")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
