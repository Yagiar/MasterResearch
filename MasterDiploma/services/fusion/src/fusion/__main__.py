"""Точка входа сервиса fusion.

Запуск:
    python -m fusion
    python -m fusion --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys

from uavdet_common.bus import KafkaMessageBus
from uavdet_common.config import load_config
from uavdet_common.logging import configure_logging, get_logger
from uavdet_common.metrics import start_metrics_server

from .consumer import InferenceConsumer
from .factory import build_gating, build_strategy
from .temporal import ChannelHealthGate, MedianSmoother
from .window_buffer import TimeWindowBuffer


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fusion")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    configure_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format", "console"))
    log = get_logger("fusion")

    kafka_cfg = dict(cfg.get("kafka", {}))
    fusion_cfg = dict(cfg.get("fusion", {}))

    bus = KafkaMessageBus(
        bootstrap_servers=str(kafka_cfg.get("bootstrap_servers", "kafka:9092")),
        client_id="fusion",
        auto_offset_reset=str(fusion_cfg.get("auto_offset_reset", "latest")),
    )

    metrics_port = int(fusion_cfg.get("metrics_port", 0))
    if metrics_port > 0:
        start_metrics_server(metrics_port)
        log.info("fusion: метрики на /metrics", port=metrics_port)

    strategy = build_strategy(fusion_cfg)
    gating = build_gating(fusion_cfg)
    buffer = TimeWindowBuffer(
        epsilon_ms=float(fusion_cfg.get("window_epsilon_ms", 80.0)),
        # горизонт опоздания: сколько истории удерживается для запаздывающей модальности
        # (it-33, ревью §6.2: порядок доставки не должен менять факт совместного окна)
        lateness_ms=float(fusion_cfg.get("window_lateness_ms", 2000.0)),
    )

    # каузальная медиана p_a по последним k аудио-окнам (0 = выключено; рекомендация it-08: 5)
    temporal_k = int(fusion_cfg.get("audio_temporal_k", 0))
    audio_smoother = MedianSmoother(k=temporal_k) if temporal_k > 0 else None

    # гейт здоровья аудиоканала: «тишина ≠ глухота» по RMS+std p_a (it-16/it-19; выключено по умолчанию)
    health_gate = None
    if bool(fusion_cfg.get("audio_health_gate", False)):
        health_gate = ChannelHealthGate(w=int(fusion_cfg.get("audio_health_w", 12)))

    service = InferenceConsumer(
        bus,
        group_id=str(fusion_cfg.get("group_id", "fusion")),
        strategy=strategy,
        gating=gating,
        window_buffer=buffer,
        decision_threshold=float(fusion_cfg.get("decision_threshold", 0.5)),
        audio_smoother=audio_smoother,
        health_gate=health_gate,
        # it-44 (ревью §6.2): per-message — прежнее поведение; watermark — окно выпускается,
        # когда обе модальности закрыли медиа-интервал, либо по max_wait (явный mono-fallback)
        window_release=str(fusion_cfg.get("window_release", "per-message")),
        window_max_wait_ms=float(fusion_cfg.get("window_max_wait_ms", 2000.0)),
        target=str(fusion_cfg.get("target", "presence")),
        motion_floor=float(fusion_cfg.get("motion_floor", 0.15)),
        audio_confirm_floor=float(fusion_cfg.get("audio_confirm_floor", 0.3)),
    )
    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
