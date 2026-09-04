"""Конфигурация: загрузка YAML-профиля (configs/pilot.yaml) + переопределение через окружение.

Сервисы получают конфиг так:
    cfg = load_config()                # читает UAVDET_CONFIG или configs/pilot.yaml + .env
    cfg["kafka"]["bootstrap_servers"]  # доступ как к dict

Дополнительно: Settings (pydantic-settings) — для типизированных «горячих» параметров,
которые удобно задавать переменными окружения с префиксом UAVDET_ и вложенностью через "__".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CONFIG_PATH = "configs/pilot.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Загрузить YAML-профиль. Путь: аргумент → env UAVDET_CONFIG → configs/pilot.yaml."""
    cfg_path = Path(path or os.environ.get("UAVDET_CONFIG", _DEFAULT_CONFIG_PATH))
    if not cfg_path.exists():
        # допускаем запуск без файла (например, в юнит-тестах) — возвращаем пустой конфиг
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _apply_env_overrides(data)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Переопределить значения из переменных окружения вида UAVDET_<path__через__split>."""
    prefix = "UAVDET_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix) or env_key == "UAVDET_CONFIG":
            continue
        path = env_key[len(prefix) :].lower().split("__")
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[path[-1]] = _coerce(env_val)
    return data


def _coerce(s: str) -> Any:
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


class Settings(BaseSettings):
    """Типизированные настройки окружения (опционально; основной конфиг — YAML)."""

    model_config = SettingsConfigDict(
        env_prefix="UAVDET_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    config: str = _DEFAULT_CONFIG_PATH  # путь к YAML-профилю


__all__ = ["load_config", "Settings"]
