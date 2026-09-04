"""BatchCollector — микро-батчинг кадров для GPU-эффективного инференса (заготовка).

На MVP не используется (batch_size=1). Назначение (вне MVP): копить до N кадров
или до таймаута, прогонять детектор одним вызовом, разворачивать результаты обратно
по сообщениям — повышает throughput на GPU за счёт небольшой добавки к latency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatchCollector:
    """Заготовка коллектора микро-батчей (не реализована на MVP)."""

    max_batch: int = 1
    max_wait_ms: int = 0

    def __post_init__(self) -> None:
        if self.max_batch > 1:
            raise NotImplementedError("микро-батчинг будет реализован вне MVP (max_batch>1)")
