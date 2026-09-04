"""Каузальное временнóе сглаживание аудио-канала fusion (`fusion.audio_temporal_k`).

Зачем: аудио-детекции AST имеют короткие провалы уверенности в полёте (it-05: R=0.766
при P=0.977 на sandbox с GT); каузальная медиана p(drone) по последним k валидным
аудио-окнам поднимает F1 решений против GT «БПЛА активен» с 0.859 до 0.913 без
добавленной задержки (it-08; журналы — research/iterations/).

Правило валидности (it-07): окно без аудио НЕ участвует в фильтрации (пропуск ≠ ноль)
и остаётся моно-видео — стратегия получает его без изменений.
"""

from __future__ import annotations

from collections import defaultdict, deque

from uavdet_common.messages import InferenceMsg

from .window_buffer import AlignedWindow

_LABEL_DRONE = "drone"


class MedianSmoother:
    """Каузальная медиана p(drone) по последним k аудио-решениям (per source_id).

    В историю попадают только окна, где аудио присутствовало; медиана считается
    по накопленным значениям (текущее включается). Отдельная сущность на сервис,
    состояние — deque(maxlen=k) на source_id.
    """

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k должен быть >= 1")
        self._k = k
        self._hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=k))

    def smoothed(self, source_id: str, p_drone: float) -> float:
        """Добавить p(drone) текущего окна (валидного) → медиана последних k значений."""
        self._hist[source_id].append(float(p_drone))
        vals = sorted(self._hist[source_id])
        return vals[len(vals) // 2]


def apply_audio_smoothing(window: AlignedWindow, smoother: MedianSmoother | None) -> AlignedWindow:
    """Заменить в окне best_audio на сглаженное значение p(drone) (каузально).

    Окно без аудио возвращается как есть. Лучшее аудио-сообщение копируется с
    label='drone' и confidence=медиана p(drone): стратегии (late/hybrid/audio-only)
    вычисляют p_a = confidence при label='drone', т.е. получают сглаженное значение,
    а Δ-правило и порог 0.5 работают по сглаженной величине. Остальные поля
    (msg_id, ts, quality) сохраняются — трассировка source_msg_ids и метрики Δt не меняются.
    """
    if smoother is None:
        return window
    best_a = window.best_audio()
    if best_a is None:
        return window
    p_drone = best_a.confidence if best_a.label == _LABEL_DRONE else 0.0
    smoothed = smoother.smoothed(window.source_id, p_drone)
    new_a: InferenceMsg = best_a.model_copy(update={"label": _LABEL_DRONE, "confidence": smoothed})
    audio = [new_a if m.msg_id == best_a.msg_id else m for m in window.audio]
    return AlignedWindow(source_id=window.source_id, t0=window.t0, t1=window.t1,
                         video=list(window.video), audio=audio)
