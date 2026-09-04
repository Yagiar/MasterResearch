# Эскиз патча: каузальная медиана p_a в fusion-сервисе

- **Дата:** 2026-09-04 • **Обоснование:** it-06/it-07/it-08 (F1 0.859 → 0.913 против `airborne`-GT, без добавленной задержки)
- **Статус:** ЭСКИЗ — не применён. Применять только вместе с повторным ablation (см. критерий приёмки в it-08).

## 1. Новый модуль `services/fusion/src/fusion/temporal.py`

```python
"""Каузальное временнóе сглаживание аудио-канала fusion (it-08).

Медиана по последним k ВАЛИДНЫМ аудио-решениям (урок it-07: пропуск ≠ ноль;
окно без аудио не участвует в фильтрации и решается как моно-видео).
"""
from __future__ import annotations
from collections import defaultdict, deque


class MedianSmoother:
    def __init__(self, k: int = 5) -> None:
        self._k = k
        self._hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=k))

    def filter(self, source_id: str, p_a: float | None) -> float | None:
        """p_a текущего окна (None = аудио в окне нет) → сглаженное значение|None."""
        if p_a is None:
            return None                      # моно-окно фильтр не трогаем
        self._hist[source_id].append(p_a)
        vals = sorted(self._hist[source_id])
        return vals[len(vals) // 2]
```

## 2. Точка подключения `services/fusion/src/fusion/consumer.py`

В обработке окна, после выбора `best_audio` и ДО вызова стратегии:

```python
p_a_raw = best_a.confidence if (best_a and best_a.label == "drone") else 0.0
p_a_smooth = self._smoother.filter(msg.source_id, p_a_raw if best_a else None)
```

Стратегии принимают уже сглаженное значение (минимальный дифф: передавать
`p_a_smooth` в `strategy.decide(win, p_a=p_a_smooth)`; `late.py`/`hybrid.py`
используют его вместо `best_a.confidence`).

## 3. Конфиг `configs/pilot.yaml`

```yaml
fusion:
  audio_temporal_k: 5   # 0 = выключено (текущее поведение); рекомендация it-08: 5
```

Прокинуть в fusion через env-override (как `UAVDET_FUSION__*`), добавить в
`run()` из `scripts/ablation.sh` варьируемый параметр при повторной ablation.

## 4. Обязательные спутники патча

1. `run_id` (uuid4) в каждом DecisionMsg + ротация `data/decisions/run_<id>.jsonl` — иначе после патча ablation снова несопоставим (it-03, п.4: 0.35% смешанных окон в №4/№5).
2. Разведение `mode` = `late` vs `late+gating` (и теперь `late+temporal`) в поле `mode`.
3. Счётчик «окон с аудио / всего» в метриках fusion (Prometheus) — прямая видимость аномалии доставки аудио (7% в №4 против 75% в №3).

## 5. Риски

- Смена поведения fusion ломает сравнимость со старыми прогонами → все новые прогоны помечать `audio_temporal_k`.
- На другом материале (MMAUD) оптимум k может отличаться — k оставить конфигурируемым.
- `hybrid` получает то же сглаживание автоматически (общий consumer), Δ-правило не трогаем в этой итерации (его шумочувствительность — it-07, п.3 — отдельный вопрос).
