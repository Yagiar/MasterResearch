"""TimeWindowBuffer — буфер временного выравнивания детекций (per source_id).

Stateful-компонент: накапливает последние детекции каждой модальности и по новой
детекции возвращает «окно» — набор детекций обеих модальностей, попавших в интервал
[t-ε, t+ε] вокруг момента события. На MVP (video-only) аудио-сторона пуста, окно
содержит только видеодетекцию — этого достаточно для решения.

Реализация: для каждой модальности храним deque последних детекций; при добавлении
детекции с меткой времени t — ищем детекции другой модальности в [t-ε, t+ε].

Политика опоздания (it-33, ревью 2026-09-05 §6.2): записи обеих модальностей
удаляются только старше `t − ε − lateness` (горизонт допустимого опоздания,
`window_lateness_ms` в конфиге), а не `t − ε`. Иначе запаздывающая модальность
(напр. аудио при медленном acoustic-detector) находила окно уже опустошённым:
результат объединения зависел бы от ПОРЯДКА доставки, а не от времени событий.
Порядок доставки теперь не меняет факт образования совместного окна (пока опоздание
в пределах горизонта); превышение горизонта — явное «опоздало» (флаг `late` окна,
метрика `uavdet_late_messages_total`), совместность окна — флаг `joint`
(метрика `uavdet_joint_windows_total`).
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
    late: bool = False   # добавленное сообщение опоздало (ts < последнего − ε − lateness)
    joint: bool = False  # в окне есть детекции обеих модальностей
    # it-35: событийное время триггер-сообщения на медиатаймлайне (None — не отдано источником;
    # окна выравниваются по media_ts, если он есть, иначе по wall-clock ts)
    media_ts: float | None = None

    @property
    def ts_window(self) -> list[float]:
        return [self.t0, self.t1]

    def best_video(self) -> InferenceMsg | None:
        return _best_of(self.video)

    def best_audio(self) -> InferenceMsg | None:
        return _best_of(self.audio)


def _align_ts(msg: InferenceMsg) -> float:
    """Шкала выравнивания сообщения: медиа-время, если источник его отдал, иначе wall-clock (it-35)."""
    return msg.media_ts if msg.media_ts is not None else msg.ts


class TimeWindowBuffer:
    """Буфер выравнивания с окном ±epsilon_ms и горизонтом опоздания lateness_ms (per source_id)."""

    def __init__(self, epsilon_ms: float = 80.0, max_keep: int = 256,
                 lateness_ms: float = 2000.0) -> None:
        self._eps_s = max(0.0, epsilon_ms / 1000.0)
        # горизонт опоздания: сколько истории удерживается сверх ε для запаздывающей модальности
        self._lateness_s = max(0.0, lateness_ms / 1000.0)
        self._max_keep = max_keep
        self._video: dict[str, deque[InferenceMsg]] = defaultdict(lambda: deque(maxlen=self._max_keep))
        self._audio: dict[str, deque[InferenceMsg]] = defaultdict(lambda: deque(maxlen=self._max_keep))
        self._latest: dict[str, float] = defaultdict(float)  # последний ts потока источника (обе модальности)
        # последний ts ПО КАЖДОЙ модальности (it-44): модальность участвует в водяном знаке,
        # только если уже появлялась — иначе стартовый бурст видео «прорезал» бы watermark
        self._last_mod: dict[str, dict[str, float]] = defaultdict(dict)

    @property
    def epsilon_s(self) -> float:
        """Полуширина окна выравнивания ε (сек) — для watermark-релиза (it-44)."""
        return self._eps_s

    def _evict_older_than(self, dq: deque[InferenceMsg], cutoff: float) -> None:
        while dq and dq[0].ts < cutoff:
            dq.popleft()

    def add(self, msg: InferenceMsg) -> AlignedWindow:
        """Добавить детекцию; вернуть окно выравнивания вокруг её момента.

        Выравнивание — по `media_ts` (событийное время), а при его отсутствии по wall-clock
        `ts` (it-35: время доставки ≠ время события, ревью §5.2/§6.1). Окно всегда содержит
        саму добавленную детекцию (в своей модальности) и все детекции другой модальности,
        чьи `ts` лежат в [ts-ε, ts+ε]. История сверх ε удерживается ещё `lateness_ms`,
        чтобы запаздывающая модальность могла образовать совместное окно задним числом.
        """
        t = _align_ts(msg)
        sid = msg.source_id
        # «опоздало» = пришло заметно позади потока источника (за пределами ε + lateness)
        late = self._latest[sid] > 0.0 and t < self._latest[sid] - self._eps_s - self._lateness_s
        self._latest[sid] = max(self._latest[sid], t)
        self._last_mod[sid][msg.modality] = t

        if msg.modality == "video":
            self._video[sid].append(msg)
        else:
            self._audio[sid].append(msg)

        t0, t1 = t - self._eps_s, t + self._eps_s
        # подчистка: старше окна выравнивания минус горизонт опоздания (обе модальности)
        cutoff = t0 - self._lateness_s
        self._evict_older_than(self._video[sid], cutoff)
        self._evict_older_than(self._audio[sid], cutoff)

        win = AlignedWindow(source_id=sid, t0=t0, t1=t1, late=late,
                            media_ts=msg.media_ts)
        win.video = [m for m in self._video[sid] if t0 <= _align_ts(m) <= t1] or (
            [msg] if msg.modality == "video" else []
        )
        win.audio = [m for m in self._audio[sid] if t0 <= _align_ts(m) <= t1] or (
            [msg] if msg.modality == "audio" else []
        )
        win.joint = bool(win.video) and bool(win.audio)
        return win

    # --- watermark-режим (it-44, ревью §6.2: правило завершения окна должно быть явным) ---

    def frontier(self, source_id: str) -> float:
        """Медиа-водяной знак источника: min последних ts модальностей (шкала выравнивания).

        Правило (it-44): пока у источника не appeared ОБЕ модальности, watermark = 0 —
        стартовый бурст видео не должен выпускать mono-решения, пока аудио в принципе
        не подтянулось (исследование it-42: иначе весь стартовый клип уходит в mono).
        После появления обеих — min их последних ts: интервал до watermark закрыт ими обоими.
        Если вторая модальность умерла, watermark замирает и окна уходят в явный
        max_wait-fallback (mono), не в «вечное ожидание».
        """
        mods = self._last_mod.get(source_id)
        if not mods or len(mods) < 2:
            return 0.0
        return min(mods.values())

    def form_window(self, source_id: str, t: float) -> AlignedWindow:
        """Сформировать окно [t-ε, t+ε] из УДЕРЖИВАЕМЫХ записей, не добавляя новых (watermark-режим).

        Отличие от add(): не мутирует состояние, не триггерит evict; окно выпускается вызывающим
        кодом, когда watermark прошёл t+ε (обе модальности успели) либо истёк max_wait
        (явный mono-fallback, помечается late=True).
        """
        t0, t1 = t - self._eps_s, t + self._eps_s
        win = AlignedWindow(source_id=source_id, t0=t0, t1=t1)
        win.video = [m for m in self._video.get(source_id, ()) if t0 <= _align_ts(m) <= t1]
        win.audio = [m for m in self._audio.get(source_id, ()) if t0 <= _align_ts(m) <= t1]
        win.joint = bool(win.video) and bool(win.audio)
        return win
