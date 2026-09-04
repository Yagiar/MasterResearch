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
from statistics import median, pstdev

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
    # p_drone из сообщения (it-18) точнее деградированной пары (label, confidence);
    # fallback — старый маппинг для сообщений без поля
    p_drone = best_a.p_drone if best_a.p_drone is not None else (
        best_a.confidence if best_a.label == _LABEL_DRONE else 0.0)
    smoothed = smoother.smoothed(window.source_id, p_drone)
    new_a: InferenceMsg = best_a.model_copy(update={"label": _LABEL_DRONE, "confidence": smoothed})
    audio = [new_a if m.msg_id == best_a.msg_id else m for m in window.audio]
    return AlignedWindow(source_id=window.source_id, t0=window.t0, t1=window.t1,
                         video=list(window.video), audio=audio)


class ChannelHealthGate:
    """Гейт здоровья аудиоканала (research/it-16, it-19): «тишина» ≠ «глухота».

    Признак глухоты: звука РАЗУМНО МНОГО (скользящая RMS > rms_abs_floor и > доли
    долгосрочной медианы), а выход прижат к нулю (std p_a < std_floor И средний
    p_a < mean_pa_floor) — несколько окон подряд. Тишина (RMS < rms_abs_floor)
    подозрением не считается — каналу нечего ловить; константно-уверенный «drone»
    (средний выход высок) — тоже не глухота. Закрытый гейт обнуляет добавку w_a
    (стратегия решает по видео); возврат — при оживлении выхода или уходе звука.
    rms_abs_floor — абсолютный пол «звука есть» (подбирается под материал; sandbox
    из research/it-16: тишина ≈0.005, полёт ≈0.07).
    """

    def __init__(
        self,
        *,
        w: int = 12,
        std_floor: float = 0.05,
        mean_pa_floor: float = 0.2,
        rms_abs_floor: float = 0.01,
        rms_rel_high: float = 0.8,
        rms_rel_low: float = 0.5,
        suspect_needed: int = 6,
        hist_long: int = 600,
    ) -> None:
        if w < 2:
            raise ValueError("w должен быть >= 2")
        self._w = w
        self._std_floor = std_floor
        self._mean_pa_floor = mean_pa_floor
        self._rms_abs_floor = rms_abs_floor
        self._rms_rel_high = rms_rel_high
        self._rms_rel_low = rms_rel_low
        self._suspect_needed = suspect_needed
        self._hist: dict[str, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=w))
        self._rms_long: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=hist_long))
        self._gate_open: dict[str, bool] = defaultdict(lambda: True)
        self._suspect: dict[str, int] = defaultdict(int)

    def scale(self, source_id: str, rms: float | None, p_drone: float | None) -> float:
        """Множитель w_a (1.0 = аудио допущено; 0.0 = гейт закрыл).

        Окно без аудио (rms/p_drone = None) историю не пополняет и состояние не меняет.
        """
        if rms is None or p_drone is None:
            return 1.0 if self._gate_open[source_id] else 0.0
        self._hist[source_id].append((float(rms), float(p_drone)))
        self._rms_long[source_id].append(float(rms))
        h = self._hist[source_id]
        if len(h) < 2:
            return 1.0
        baseline = max(median(self._rms_long[source_id]), 1e-6)
        rms_vals = [r for r, _ in h]
        rms_rel = sum(rms_vals) / len(rms_vals) / baseline
        p_vals = [p for _, p in h]
        std_pa = pstdev(p_vals)
        mean_pa = sum(p_vals) / len(p_vals)
        loud = sum(rms_vals) / len(rms_vals) > self._rms_abs_floor and rms_rel > self._rms_rel_high
        if self._gate_open[source_id]:
            if loud and std_pa < self._std_floor and mean_pa < self._mean_pa_floor:
                self._suspect[source_id] += 1
            else:
                self._suspect[source_id] = 0
            if self._suspect[source_id] >= self._suspect_needed:
                self._gate_open[source_id] = False
                self._suspect[source_id] = 0
        else:
            # оживление канала: выход отлип от нуля либо звук ушёл
            if mean_pa >= self._mean_pa_floor or std_pa >= 0.15 or rms_rel < self._rms_rel_low:
                self._gate_open[source_id] = True
        return 1.0 if self._gate_open[source_id] else 0.0
