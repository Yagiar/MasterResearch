"""Точка входа сервиса visual-detector.

Запуск:
    python -m visual_detector
    python -m visual_detector --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys

from uavdet_common.bus import KafkaMessageBus
from uavdet_common.config import load_config
from uavdet_common.logging import configure_logging, get_logger
from uavdet_common.metrics import start_metrics_server

from .consumer import VideoConsumer
from .factory import build_detector, build_tracker


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="visual-detector")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("visual-detector")

    kafka_cfg = dict(cfg.get("kafka", {}))
    vd_cfg = dict(cfg.get("visual_detector", {}))

    bus = KafkaMessageBus(
        bootstrap_servers=str(kafka_cfg.get("bootstrap_servers", "kafka:9092")),
        client_id="visual-detector",
        auto_offset_reset=str(vd_cfg.get("auto_offset_reset", "latest")),
    )

    metrics_port = int(vd_cfg.get("metrics_port", 0))
    if metrics_port > 0:
        start_metrics_server(metrics_port)
        log.info("visual-detector: метрики на /metrics", port=metrics_port)

    detector = build_detector(vd_cfg)
    tracker = build_tracker(vd_cfg)

    service = VideoConsumer(
        bus,
        group_id=str(vd_cfg.get("group_id", "visual-detector")),
        detector=detector,
        tracker=tracker,
    )
    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
