"""PreprocessChain — цепочка предобработки кадра перед инференсом (паттерн
Chain of Responsibility). На MVP — минимум (no-op): Ultralytics YOLO сам делает
letterbox/resize/нормализацию внутри. Здесь зарезервировано место под кадр-специфичную
предобработку (денойз, гамма-коррекция, ROI-кроп, и т.п.).
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

PreprocessStep = Callable[[np.ndarray], np.ndarray]


class PreprocessChain:
    """Последовательное применение шагов предобработки к кадру."""

    def __init__(self, steps: Iterable[PreprocessStep] | None = None) -> None:
        self._steps: list[PreprocessStep] = list(steps or [])

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        for step in self._steps:
            out = step(out)
        return out

    @classmethod
    def identity(cls) -> "PreprocessChain":
        """Пустая цепочка (MVP-вариант)."""
        return cls([])
