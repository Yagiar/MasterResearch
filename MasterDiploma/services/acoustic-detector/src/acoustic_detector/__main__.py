"""Точка входа сервиса acoustic-detector.

Запуск:
    python -m acoustic_detector
    python -m acoustic_detector --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys

from uavdet_common.bus import KafkaMessageBus
from uavdet_common.config import load_config
from uavdet_common.logging import configure_logging, get_logger
from uavdet_common.metrics import start_metrics_server

from .consumer import AudioConsumer
from .factory import build_detector, build_extractor


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="acoustic-detector")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("acoustic-detector")

    kafka_cfg = dict(cfg.get("kafka", {}))
    ad_cfg = dict(cfg.get("acoustic_detector", {}))

    bus = KafkaMessageBus(
        bootstrap_servers=str(kafka_cfg.get("bootstrap_servers", "kafka:9092")),
        client_id="acoustic-detector",
        auto_offset_reset=str(ad_cfg.get("auto_offset_reset", "latest")),
    )

    metrics_port = int(ad_cfg.get("metrics_port", 0))
    if metrics_port > 0:
        start_metrics_server(metrics_port)
        log.info("acoustic-detector: метрики на /metrics", port=metrics_port)

    extractor = build_extractor(ad_cfg)
    detector = build_detector(ad_cfg, extractor)

    service = AudioConsumer(
        bus,
        group_id=str(ad_cfg.get("group_id", "acoustic-detector")),
        extractor=extractor,
        detector=detector,
    )
    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
