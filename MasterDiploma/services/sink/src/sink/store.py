"""JsonlStore — построчная запись решений в JSON Lines для последующего анализа.

Каждое решение — одна строка JSON (`DecisionMsg.model_dump`). Файл открывается в
режиме дозаписи; директория создаётся при необходимости. На пилоте этого формата
достаточно для расчёта метрик качества офлайн (по ground truth датасета).
"""

from __future__ import annotations

import json
from pathlib import Path

from uavdet_common.messages import DecisionMsg


class JsonlStore:
    """Append-only хранилище решений в .jsonl."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def write(self, decision: DecisionMsg) -> None:
        self._fh.write(json.dumps(decision.model_dump(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
