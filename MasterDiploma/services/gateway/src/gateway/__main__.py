"""Точка входа сервиса gateway (dashboard-bff).

Запуск:
    python -m gateway --config configs/pilot.yaml

Поднимает FastAPI через uvicorn. Эндпоинты и фоновый Kafka-потребитель `decisions` — см. app.py.
"""

from __future__ import annotations

import argparse
import sys

from uavdet_common.config import load_config
from uavdet_common.logging import configure_logging, get_logger


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gateway")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    parser.add_argument("--host", default=None, help="bind host (по умолчанию из конфига / 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (по умолчанию из конфига / 8080)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("gateway")

    gw_cfg = dict(cfg.get("gateway", {}))
    host = args.host or str(gw_cfg.get("bind_host", "0.0.0.0"))
    port = int(args.port or gw_cfg.get("port", 8080))

    import uvicorn  # noqa: PLC0415

    log.info("gateway: запуск dashboard-bff", host=host, port=port)
    uvicorn.run("gateway.app:app", host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
