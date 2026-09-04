"""TimeWindowBuffer — буфер временного выравнивания детекций (per source_id).

Stateful-компонент: накапливает последние детекции каждой модальности и по новой
детекции возвращает «окно» — набор детекций обеих модальностей, попавших в интервал
[t-ε, t+ε] вокруг момента события. На MVP (video-only) аудио-сторона пуста, окно
содержит только видеодетекцию — этого достаточно для решения.

Простая реализация: для каждой модальности храним deque последних детекций;
при добавлении видеодетекции с меткой времени t — ищем аудиодетекции в [t-ε, t+ε].
Старые записи (старше t - ε) удаляем.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from uavdet_common.messages import InferenceMsg

_LABEL_DRONE = "drone"


def _best_of(msgs: list[InferenceMsg]) -> InferenceMsg | None:
    """Лучшая детекция модальности в окне: приоритет — самая уверенная с label='drone';
    если drone-детекций в окне нет — самая уверенная вообще (обычно label='non-drone', conf=0).
    Так non-drone-детекция (conf=0) не «затмевает» drone-детекцию в смешанном окне.
    """
    if not msgs:
        return None
    drone = [m for m in msgs if m.label == _LABEL_DRONE]
    pool = drone if drone else msgs
    return max(pool, key=lambda m: m.confidence)


@dataclass
class AlignedWindow:
    """Окно выравнивания вокруг момента `t0..t1` для одного source_id."""

    source_id: str
    t0: float
    t1: float
    video: list[InferenceMsg] = field(default_factory=list)
    audio: list[InferenceMsg] = field(default_factory=list)

    @property
    def ts_window(self) -> list[float]:
        return [self.t0, self.t1]

    def best_video(self) -> InferenceMsg | None:
        return _best_of(self.video)

    def best_audio(self) -> InferenceMsg | None:
        return _best_of(self.audio)


class TimeWindowBuffer:
    """Буфер выравнивания с окном ±epsilon_ms (раздельно по source_id)."""

    def __init__(self, epsilon_ms: float = 80.0, max_keep: int = 256) -> None:
        self._eps_s = max(0.0, epsilon_ms / 1000.0)
        self._max_keep = max_keep
        self._video: dict[str, deque[InferenceMsg]] = defaultdict(lambda: deque(maxlen=self._max_keep))
        self._audio: dict[str, deque[InferenceMsg]] = defaultdict(lambda: deque(maxlen=self._max_keep))

    def _evict_older_than(self, dq: deque[InferenceMsg], cutoff: float) -> None:
        while dq and dq[0].ts < cutoff:
            dq.popleft()

    def add(self, msg: InferenceMsg) -> AlignedWindow:
        """Добавить детекцию; вернуть окно выравнивания вокруг её момента.

        Окно всегда содержит саму добавленную детекцию (в своей модальности) и все
        детекции другой модальности, чьи `ts` лежат в [ts-ε, ts+ε].
        """
        t = msg.ts
        sid = msg.source_id
        if msg.modality == "video":
            self._video[sid].append(msg)
        else:
            self._audio[sid].append(msg)

        t0, t1 = t - self._eps_s, t + self._eps_s
        # подчистка слишком старых записей обеих модальностей
        self._evict_older_than(self._video[sid], t0)
        self._evict_older_than(self._audio[sid], t0)

        win = AlignedWindow(source_id=sid, t0=t0, t1=t1)
        win.video = [m for m in self._video[sid] if t0 <= m.ts <= t1] or (
            [msg] if msg.modality == "video" else []
        )
        win.audio = [m for m in self._audio[sid] if t0 <= m.ts <= t1] or (
            [msg] if msg.modality == "audio" else []
        )
        return win
