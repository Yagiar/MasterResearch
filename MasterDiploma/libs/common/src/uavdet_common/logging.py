"""Настройка логирования (structlog) — общая для всех сервисов."""

from __future__ import annotations

import logging
import os
import sys

import structlog

_CONFIGURED = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Инициализировать structlog один раз. level: DEBUG/INFO/...; fmt: 'console' | 'json'."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("UAVDET_LOGGING__LEVEL") or "INFO").upper()
    fmt = (fmt or os.environ.get("UAVDET_LOGGING__FORMAT") or "console").lower()

    logging.basicConfig(stream=sys.stdout, level=getattr(logging, level, logging.INFO), format="%(message)s")

    renderer = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None):
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()


__all__ = ["configure_logging", "get_logger"]
