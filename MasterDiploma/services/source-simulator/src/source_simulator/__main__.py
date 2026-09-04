"""Точка входа сервиса source-simulator.

Запуск:
    python -m source_simulator                       # конфиг из $UAVDET_CONFIG или configs/pilot.yaml
    python -m source_simulator --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys

from uavdet_common.config import load_config
from uavdet_common.logging import configure_logging, get_logger

from .controller import SimulatorController
from .factory import build_adapter, maybe_wrap_degradation
from .grpc_client import GrpcStreamClient


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="source-simulator")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу (по умолчанию: $UAVDET_CONFIG / configs/pilot.yaml)")
    return parser.parse_args(argv)


def _video_period_s(adapter, source_cfg: dict) -> float:
    p = getattr(adapter, "target_period_s", None)
    if p is not None:
        return float(p)
    fps = float(source_cfg.get("fps", 25.0))
    return 1.0 / fps if fps > 0 else 0.0


def _audio_period_s(adapter, source_cfg: dict) -> float:
    p = getattr(adapter, "audio_period_s", None)
    if p is not None and float(p) > 0:
        return float(p)
    hop = float(dict(source_cfg.get("audio", {})).get("hop_ms", 500))
    return hop / 1000.0 if hop > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("source-simulator")

    source_cfg = dict(cfg.get("source", {}))
    grpc_cfg = dict(cfg.get("ingest_grpc", {}))

    raw_adapter = build_adapter(source_cfg)
    # темпы и наличие аудио берём с «сырого» адаптера (DegradationChannel их не проксирует)
    video_period = _video_period_s(raw_adapter, source_cfg)
    audio_period = _audio_period_s(raw_adapter, source_cfg)
    enable_audio = bool(getattr(raw_adapter, "has_audio", False)) and bool(source_cfg.get("enable_audio", True))

    adapter = maybe_wrap_degradation(raw_adapter, source_cfg)

    client = GrpcStreamClient(
        host=str(grpc_cfg.get("host", "ingest-gateway")),
        port=int(grpc_cfg.get("port", 50051)),
        max_message_mb=int(grpc_cfg.get("max_message_mb", 16)),
    )

    controller = SimulatorController(
        adapter, client,
        video_period_s=video_period, audio_period_s=audio_period, enable_audio=enable_audio,
    )

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("source-simulator: получен сигнал, останавливаюсь", signal=signum)
        controller.request_stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info(
        "source-simulator: подключение к ingest-gateway",
        target=f"{grpc_cfg.get('host')}:{grpc_cfg.get('port')}",
        adapter=raw_adapter.name, audio="on" if enable_audio else "off",
    )
    try:
        client.wait_ready(timeout_s=float(grpc_cfg.get("connect_timeout_s", 30.0)))
        controller.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("source-simulator: завершение с ошибкой", error=str(exc))
        return 1
    finally:
        controller.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
