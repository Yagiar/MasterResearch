"""InferenceConsumer — consumer-сервис fusion-движка.

Поток обработки одного сообщения `inference`:
  1) положить детекцию в TimeWindowBuffer (per source_id) -> получить окно ±ε;
  2) GatingPolicy.weights(окно) -> GatingResult (веса w_v/w_a + сводные показатели качества);
  3) FusionStrategy.fuse(окно, веса, порог) -> FusionOutcome | None;
  4) если решение сформировано -> собрать DecisionMsg (с contributions и gating) -> опубликовать в `decisions`.

e2e-latency = (момент публикации решения − ingest_ts ранней детекции в окне), если доступен.
"""

from __future__ import annotations

import time

from uavdet_common.consumer_service import KafkaConsumerService
from uavdet_common.messages import Contributions, DecisionMsg, Gating, InferenceMsg, Topics
from uavdet_common.metrics import DECISIONS_TOTAL, DELTA_T_MS, E2E_LATENCY, MESSAGES_TOTAL

from .strategies.base import FusionStrategy
from .temporal import ChannelHealthGate, MedianSmoother, apply_audio_smoothing
from .window_buffer import AlignedWindow, TimeWindowBuffer

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
    ) -> None:
        super().__init__(bus)
        self.group_id = group_id
        self._strategy = strategy
        self._gating = gating
        self._buffer = window_buffer
        self._threshold = float(decision_threshold)
        self._audio_smoother = audio_smoother
        self._health_gate = health_gate

    def on_start(self) -> None:
        self._log.info(
            "fusion: старт",
            mode=getattr(self._strategy, "mode", "?"),
            gating=type(self._gating).__name__,
            threshold=self._threshold,
            audio_temporal_k=self._audio_smoother._k if self._audio_smoother else 0,
            audio_health_gate=self._health_gate is not None,
        )

    def process(self, key: str | None, msg: InferenceMsg) -> None:  # type: ignore[override]
        window = self._buffer.add(msg)
        raw_a = window.best_audio()                      # до сглаживания: сырой p_a для гейта здоровья
        window = apply_audio_smoothing(window, self._audio_smoother)
        gr = self._gating.weights(window)
        w_a = gr.w_a
        if self._health_gate is not None:
            p_a_raw = (raw_a.confidence if raw_a.label == "drone" else 0.0) if raw_a is not None else None
            rms = raw_a.quality.audio_rms if raw_a is not None else None
            w_a *= self._health_gate.scale(msg.source_id, rms, p_a_raw)
        outcome = self._strategy.fuse(window, w_v=gr.w_v, w_a=w_a, threshold=self._threshold)
        if outcome is None:
            return

        now = time.time()
        ingest_ts = self._earliest_ingest_ts(window)
        e2e_ms = (now - ingest_ts) * 1000.0 if ingest_ts is not None else 0.0
        self._observe_delta_t(window)

        decision = DecisionMsg(
            source_id=msg.source_id,
            ts=now,
            ts_window=window.ts_window,
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
        self.publish(Topics.DECISIONS, msg.source_id, decision)
        MESSAGES_TOTAL.labels(service=_SERVICE, topic=Topics.DECISIONS).inc()
        DECISIONS_TOTAL.labels(service=_SERVICE, decision="drone" if outcome.decision else "non-drone").inc()
        if e2e_ms > 0:
            E2E_LATENCY.labels(service=_SERVICE).observe(e2e_ms / 1000.0)

    def _observe_delta_t(self, window: AlignedWindow) -> None:
        """Записать метрику межмодальной задержки Δt (если в окне есть обе модальности)."""
        bv, ba = window.best_video(), window.best_audio()
        if bv is not None and ba is not None:
            DELTA_T_MS.labels(service=_SERVICE).observe(abs(bv.ts - ba.ts) * 1000.0)

    @staticmethod
    def _earliest_ingest_ts(window: AlignedWindow) -> float | None:
        candidates = [m.ingest_ts for m in (*window.video, *window.audio) if m.ingest_ts is not None]
        return min(candidates) if candidates else None
