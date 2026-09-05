"""InferenceConsumer — consumer-сервис fusion-движка.

Поток обработки одного сообщения `inference`:
  1) положить детекцию в TimeWindowBuffer (per source_id) -> получить окно ±ε;
  2) GatingPolicy.weights(окно) -> GatingResult (веса w_v/w_a + сводные показатели качества);
  3) FusionStrategy.fuse(окно, веса, порог) -> FusionOutcome | None;
  4) если решение сформировано -> собрать DecisionMsg (с contributions и gating) -> опубликовать в `decisions`.

Два режима выпуска окон (`fusion.window_release`, it-44, ревью §6.2):
  - `per-message` (по умолчанию, прежнее поведение): решение формируется сразу по каждому
    сообщению; окно содержит то, что успело дойти.
  - `watermark`: сообщение только ПОПОЛНЯЕТ буфер; решение для триггера t выпускается, когда
    медиа-водяной знак (min последних ts обеих модальностей) прошёл t+ε — обе модальности
    успели закрыть интервал, — либо по истечении `window_max_wait_ms` (ЯВНЫЙ mono-fallback,
    помечается как late и счётчиком `uavdet_window_releases_total{reason="max_wait"}`).
    Это устраняет главный источник mono-окон онлайн: бурст одной модальности не «съедает»
    партнёра второй (исследование it-42: mono-video окна давали 274 FP, joint-окна — 0 FP).

e2e-latency = (момент публикации решения − ingest_ts ранней детекции в окне), если доступен.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from uavdet_common.consumer_service import KafkaConsumerService
from uavdet_common.messages import Contributions, DecisionMsg, Gating, InferenceMsg, Topics
from uavdet_common.metrics import (
    DECISIONS_TOTAL,
    DELTA_T_MS,
    E2E_LATENCY,
    JOINT_WINDOWS_TOTAL,
    LATE_MESSAGES_TOTAL,
    MESSAGES_TOTAL,
    WINDOW_RELEASES_TOTAL,
)

from .strategies.base import FusionStrategy
from .temporal import ChannelHealthGate, MedianSmoother, apply_audio_smoothing
from .window_buffer import AlignedWindow, TimeWindowBuffer, _align_ts

_SERVICE = "fusion"


class InferenceConsumer(KafkaConsumerService):
    """inference -> decisions."""

    in_topic = Topics.INFERENCE
    in_model = InferenceMsg

    def __init__(
        self,
        bus,
        *,
        group_id: str,
        strategy: FusionStrategy,
        gating,                              # FixedGating | AdaptiveGating (метод weights(window)->GatingResult)
        window_buffer: TimeWindowBuffer,
        decision_threshold: float = 0.5,
        audio_smoother: MedianSmoother | None = None,  # каузальная медиана p_a (k<=0/None → выключено)
        health_gate: ChannelHealthGate | None = None,  # гейт «тишина vs глухота» (research/it-16, it-19)
        window_release: str = "per-message",           # per-message | watermark (it-44)
        window_max_wait_ms: float = 2000.0,            # watermark: максимум ожидания второй модальности
    ) -> None:
        super().__init__(bus)
        if window_release not in ("per-message", "watermark"):
            raise ValueError(f"неизвестный window_release: {window_release!r}")
        self.group_id = group_id
        self._strategy = strategy
        self._gating = gating
        self._buffer = window_buffer
        self._threshold = float(decision_threshold)
        self._audio_smoother = audio_smoother
        self._health_gate = health_gate
        self._release_mode = window_release
        self._max_wait_s = max(0.0, window_max_wait_ms / 1000.0)
        self._pending: dict[str, deque[tuple[InferenceMsg, float]]] = defaultdict(deque)
        self._pending_cap = 1024  # защита от роста без релизов (источник остановился)

    def on_start(self) -> None:
        self._log.info(
            "fusion: старт",
            mode=getattr(self._strategy, "mode", "?"),
            gating=type(self._gating).__name__,
            threshold=self._threshold,
            audio_temporal_k=self._audio_smoother._k if self._audio_smoother else 0,
            audio_health_gate=self._health_gate is not None,
            window_release=self._release_mode,
            window_max_wait_ms=self._max_wait_s * 1000.0,
        )

    def process(self, key: str | None, msg: InferenceMsg) -> None:  # type: ignore[override]
        window = self._buffer.add(msg)
        if window.late:  # за горизонтом опоздания — наблюдаемость политики буфера (it-33)
            LATE_MESSAGES_TOTAL.labels(service=_SERVICE).inc()
        if self._release_mode == "per-message":
            WINDOW_RELEASES_TOTAL.labels(service=_SERVICE, reason="per-message").inc()
            self._emit(window, msg)
            return
        # watermark: сообщение только пополняет буфер; триггер ждёт закрытия интервала
        self._enqueue_trigger(msg)
        self._release_ready(msg.source_id)

    # --- watermark-выпуск (it-44) ---

    def _enqueue_trigger(self, msg: InferenceMsg) -> None:
        pend = self._pending[msg.source_id]
        pend.append((msg, time.monotonic() + self._max_wait_s))
        while len(pend) > self._pending_cap:  # аварийный mono-fallback при переполнении
            trig, _ = pend.popleft()
            self._emit(self._buffer.form_window(trig.source_id, _align_ts(trig)), trig, reason="max_wait")

    def _release_ready(self, source_id: str) -> None:
        """Выпустить триггеры, чей интервал закрыт watermark'ом, либо у кого истёк max_wait."""
        wm = self._buffer.frontier(source_id)
        eps = self._buffer.epsilon_s
        pend = self._pending[source_id]
        while pend:
            trig, deadline = pend[0]
            t = _align_ts(trig)
            if wm >= t + eps:
                reason = "watermark"
            elif time.monotonic() >= deadline:
                reason = "max_wait"  # вторая модальность не успела — явный mono-fallback (it-44)
            else:
                break
            pend.popleft()
            win = self._buffer.form_window(source_id, t)
            win.media_ts = trig.media_ts
            if reason == "max_wait":
                win.late = True
            self._emit(win, trig, reason=reason)

    # --- формирование и публикация решения ---

    def _emit(self, window: AlignedWindow, trig: InferenceMsg, reason: str = "per-message") -> None:
        WINDOW_RELEASES_TOTAL.labels(service=_SERVICE, reason=reason).inc()
        if window.joint:
            JOINT_WINDOWS_TOTAL.labels(service=_SERVICE).inc()
        raw_a = window.best_audio()                      # до сглаживания: сырой p_a для гейта здоровья
        window = apply_audio_smoothing(window, self._audio_smoother)
        gr = self._gating.weights(window)
        w_a = gr.w_a
        if self._health_gate is not None:
            p_a_raw = (raw_a.confidence if raw_a.label == "drone" else 0.0) if raw_a is not None else None
            rms = raw_a.quality.audio_rms if raw_a is not None else None
            # msg_id — против повторного учёта одного аудио-сообщения через видео-триггеры (it-31)
            w_a *= self._health_gate.scale(trig.source_id, rms, p_a_raw,
                                           msg_id=raw_a.msg_id if raw_a is not None else None)
        outcome = self._strategy.fuse(window, w_v=gr.w_v, w_a=w_a, threshold=self._threshold)
        if outcome is None:
            return

        now = time.time()
        ingest_ts = self._earliest_ingest_ts(window)
        e2e_ms = (now - ingest_ts) * 1000.0 if ingest_ts is not None else 0.0
        self._observe_delta_t(window)

        decision = DecisionMsg(
            source_id=trig.source_id,
            ts=now,
            ts_window=window.ts_window,
            # событийное время решения — для GT-скоринга; None, если источник не отдал media_ts
            media_ts=window.media_ts if window.media_ts is not None else trig.media_ts,
            mode=getattr(self._strategy, "mode", "video-only"),
            decision=outcome.decision,
            p_fused=outcome.p_fused,
            contributions=Contributions(
                p_v=outcome.p_v,
                p_a=outcome.p_a,
                w_v=outcome.w_v,
                w_a=outcome.w_a,
                delta=outcome.delta,
            ),
            gating=Gating(snr_audio=gr.snr_audio, img_quality=gr.img_quality),
            e2e_latency_ms=e2e_ms,
            source_msg_ids=list(outcome.source_msg_ids),
        )
        self.publish(Topics.DECISIONS, trig.source_id, decision)
        MESSAGES_TOTAL.labels(service=_SERVICE, topic=Topics.DECISIONS).inc()
        DECISIONS_TOTAL.labels(service=_SERVICE, decision="drone" if outcome.decision else "non-drone").inc()
        if e2e_ms > 0:
            E2E_LATENCY.labels(service=_SERVICE).observe(e2e_ms / 1000.0)

    def _observe_delta_t(self, window: AlignedWindow) -> None:
        """Записать метрику межмодальной задержки Δt (если в окне есть обе модальности).

        Считается по шкале выравнивания (media_ts, иначе wall-clock ts) — it-35.
        """
        bv, ba = window.best_video(), window.best_audio()
        if bv is not None and ba is not None:
            DELTA_T_MS.labels(service=_SERVICE).observe(abs(_align_ts(bv) - _align_ts(ba)) * 1000.0)

    @staticmethod
    def _earliest_ingest_ts(window: AlignedWindow) -> float | None:
        candidates = [m.ingest_ts for m in (*window.video, *window.audio) if m.ingest_ts is not None]
        return min(candidates) if candidates else None
