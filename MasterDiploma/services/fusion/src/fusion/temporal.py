"""Каузальное временнóе сглаживание аудио-канала fusion (`fusion.audio_temporal_k`).

Зачем: аудио-детекции AST имеют короткие провалы уверенности в полёте (it-05: R=0.766
при P=0.977 на sandbox с GT); каузальная медиана p(drone) по последним k валидным
аудио-окнам поднимает F1 решений против GT «БПЛА активен» с 0.859 до 0.913 (it-08;
журналы — research/iterations/). Медиана переключается не мгновенно: после смены
сигнала нужно ~k/2 новых окон (при k=5 и шаге 0.5 с — около 1 с задержки обнаружения),
это честная цена сглаживания, а не ноль (ревью 2026-09-05, §5.1).

Правило валидности (it-07): окно без аудио НЕ участвует в фильтрации (пропуск ≠ ноль)
и остаётся моно-видео — стратегия получает его без изменений.
Состояние фильтра пополняется РОВНО ОДИН РАЗ на уникальное аудио-сообщение (msg_id):
fusion-окна триггерятся и видеосообщениями, и одно и то же best_audio попадает в
несколько окон подряд — повторные попадания читают кэшированное значение (ревью §6.3).
"""

from __future__ import annotations

from collections import defaultdict, deque
from statistics import median, pstdev

from uavdet_common.messages import InferenceMsg

from .window_buffer import AlignedWindow

_LABEL_DRONE = "drone"


class _MsgOnceCache:
    """Кэш «msg_id уже учтён» (per source_id, bounded): первое вхождение — False, повторы — True.

    Один аудио-коннект (один InferenceMsg) попадает в несколько fusion-окон подряд
    (окна триггерятся и видеосообщениями) — временнóе состояние (медиана, гейт) должно
    учитывать его один раз. Кэш ограничен по размеру (FIFO-вытеснение).
    """

    def __init__(self, cap: int = 256) -> None:
        self._cap = max(16, cap)
        self._seen: dict[str, dict[str, None]] = defaultdict(dict)
        self._order: dict[str, deque[str]] = defaultdict(deque)

    def is_repeat(self, source_id: str, msg_id: str | None) -> bool:
        """True — msg_id уже встречался у этого source_id (или None → не отслеживаем)."""
        if msg_id is None:
            return False
        seen, order = self._seen[source_id], self._order[source_id]
        if msg_id in seen:
            return True
        seen[msg_id] = None
        order.append(msg_id)
        while len(order) > self._cap:
            seen.pop(order.popleft(), None)
        return False


class MedianSmoother:
    """Каузальная медиана p(drone) по последним k УНИКАЛЬНЫМ аудио-сообщениям (per source_id).

    В историю попадают только окна, где аудио присутствовало; медиана считается
    по накопленным значениям (текущее включается). Одно аудио-сообщение (msg_id)
    учитывается ровно один раз: повторный вызов с тем же msg_id возвращает кэшированное
    сглаженное значение и состояние не меняет — иначе медиана «съедала» бы одно аудио
    k раз через видео-триггеры и её эффект зависел бы от частоты видеопотока (ревью §6.3).
    Отдельная сущность на сервис, состояние — deque(maxlen=k) + кэш msg_id на source_id.
    """

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k должен быть >= 1")
        self._k = k
        self._hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=k))
        # кэш «уже учтённых» msg_id (bounded): msg_id -> сглаженное значение на момент учёта
        self._cache_cap = max(64, 4 * k)
        self._seen: dict[str, dict[str, float]] = defaultdict(dict)
        self._seen_order: dict[str, deque[str]] = defaultdict(deque)

    def smoothed(self, source_id: str, p_drone: float, msg_id: str | None = None) -> float:
        """Учесть p(drone) аудио-сообщения → медиана последних k значений.

        msg_id передаётся всегда, когда известен (из InferenceMsg.msg_id): повторный
        вызов с тем же msg_id отдаёт кэшированный результат без пополнения истории.
        msg_id=None — легаси-поведение (пополнить всегда; для тестов/офлайна).
        """
        if msg_id is not None:
            cached = self._seen[source_id].get(msg_id)
            if cached is not None:
                return cached
        self._hist[source_id].append(float(p_drone))
        vals = sorted(self._hist[source_id])
        out = vals[len(vals) // 2]
        if msg_id is not None:
            seen, order = self._seen[source_id], self._seen_order[source_id]
            seen[msg_id] = out
            order.append(msg_id)
            while len(order) > self._cache_cap:
                seen.pop(order.popleft(), None)
        return out


def apply_audio_smoothing(window: AlignedWindow, smoother: MedianSmoother | None) -> AlignedWindow:
    """Заменить в окне best_audio на сглаженное значение p(drone) (каузально).

    Окно без аудио возвращается как есть. Лучшее аудио-сообщение копируется с
    label='drone' и confidence=медиана p(drone): стратегии (late/hybrid/audio-only)
    вычисляют p_a = confidence при label='drone', т.е. получают сглаженное значение,
    а Δ-правило и порог 0.5 работают по сглаженной величине. Остальные поля
    (msg_id, ts, quality) сохраняются — трассировка source_msg_ids и метрики Δt не меняются.

    Состояние фильтра пополняется только если этот msg_id ещё не учтён (ревью §6.3):
    окно, триггеренное видеосообщением, обычно несёт то же best_audio, что и предыдущее.
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
    smoothed = smoother.smoothed(window.source_id, p_drone, msg_id=best_a.msg_id)
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
        self._once = _MsgOnceCache()

    def scale(self, source_id: str, rms: float | None, p_drone: float | None,
              msg_id: str | None = None) -> float:
        """Множитель w_a (1.0 = аудио допущено; 0.0 = гейт закрыл).

        Окно без аудио (rms/p_drone = None) историю не пополняет и состояние не меняет.
        Повторное вхождение того же аудио-msg_id (через видео-триггеры) историю не пополняет —
        иначе w-окно гейта было бы заполнено дублями одного наблюдения и зависел бы от
        частоты видеопотока (тот же класс бага, что дубли в медиане, ревью §6.3).
        """
        if rms is None or p_drone is None:
            return 1.0 if self._gate_open[source_id] else 0.0
        if self._once.is_repeat(source_id, msg_id):
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
