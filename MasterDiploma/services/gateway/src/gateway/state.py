"""DecisionStore — потокобезопасный кольцевой буфер последних решений + агрегаты.

Заполняется фоновым Kafka-потребителем (`consumer.py`), читается REST/WS-эндпоинтами.
Хранит последние `capacity` решений (dict-представления `DecisionMsg`) и накапливает
сводную статистику (счётчики, скользящие средние).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Stats:
    total: int = 0
    drone: int = 0
    sum_p_fused: float = 0.0
    sum_e2e_ms: float = 0.0
    by_mode: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        n = max(1, self.total)
        return {
            "total": self.total,
            "drone": self.drone,
            "drone_ratio": round(self.drone / n, 4),
            "avg_p_fused": round(self.sum_p_fused / n, 4),
            "avg_e2e_latency_ms": round(self.sum_e2e_ms / n, 2),
            "by_mode": dict(self.by_mode),
        }


class DecisionStore:
    """Кольцевой буфер решений + агрегаты (thread-safe)."""

    def __init__(self, capacity: int = 500) -> None:
        self._capacity = capacity
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._stats = _Stats()
        self._lock = threading.Lock()

    def add(self, decision: dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(decision)
            self._stats.total += 1
            if decision.get("decision"):
                self._stats.drone += 1
            self._stats.sum_p_fused += float(decision.get("p_fused", 0.0) or 0.0)
            self._stats.sum_e2e_ms += float(decision.get("e2e_latency_ms", 0.0) or 0.0)
            mode = str(decision.get("mode", "?"))
            self._stats.by_mode[mode] = self._stats.by_mode.get(mode, 0) + 1

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        return items[-limit:][::-1] if limit > 0 else items[::-1]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return self._stats.to_dict()

    @property
    def capacity(self) -> int:
        return self._capacity
