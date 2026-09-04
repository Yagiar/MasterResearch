"""AudioConsumer — consumer-сервис акустического детектора.

Поток обработки одного сообщения `audio.raw`:
  1) извлечь признаки (FeatureExtractor): PCM -> MFCC/мел-спектрограмма + оценка SNR;
  2) классификация (LightweightCnnDetector): CNN по спектрограмме либо энергетический порог (если нет весов);
  3) сформировать InferenceMsg (modality=audio, quality.snr_db = оценка SNR) -> публикация в `inference`.
"""

from __future__ import annotations

from uavdet_common.consumer_service import KafkaConsumerService
from uavdet_common.messages import AudioRawMsg, InferenceMsg, ModelRef, QualityHint, Topics
from uavdet_common.metrics import DETECT_LATENCY, MESSAGES_TOTAL

import time

from .cnn import LightweightCnnDetector
from .features import FeatureExtractor

_SERVICE = "acoustic-detector"


class AudioConsumer(KafkaConsumerService):
    """audio.raw -> inference (modality=audio)."""

    in_topic = Topics.AUDIO_RAW
    in_model = AudioRawMsg

    def __init__(self, bus, *, group_id: str, extractor: FeatureExtractor, detector: LightweightCnnDetector) -> None:
        super().__init__(bus)
        self.group_id = group_id
        self._extractor = extractor
        self._detector = detector

    def on_start(self) -> None:
        self._log.info(
            "acoustic-detector: старт",
            backend=self._detector.backend,
            arch=self._detector.arch,
            model=self._detector.model_name,
            feature_dim=self._extractor.feature_dim,
            n_frames=self._extractor.n_frames,
        )

    def process(self, key: str | None, msg: AudioRawMsg) -> None:  # type: ignore[override]
        if msg.payload_kind != "pcm" or not msg.payload:
            self._log.warning("acoustic-detector: пропуск сообщения без inline PCM", payload_kind=msg.payload_kind)
            return

        t0 = time.perf_counter()
        src_sr = int(msg.meta.sr or self._extractor._sr)  # noqa: SLF001 - sr источника
        channels = int(msg.meta.channels or 1)

        # маршрутизация: AST принимает СЫРОЙ PCM (свой ASTFeatureExtractor); lwcnn/resnet18 — наши признаки.
        if getattr(self._detector, "expects_pcm", False):
            feats = self._detector.features_from_pcm(msg.payload, src_sample_rate=src_sr, channels=channels)
            snr = float(feats.snr_db)
            rms = float(feats.rms)
            det = self._detector.detect(msg.payload, src_sample_rate=src_sr, channels=channels)
        else:
            feats = self._extractor.from_audio_raw(msg.payload, src_sample_rate=src_sr, channels=channels)
            det = self._detector.detect(feats.array)
            snr = float(feats.snr_db)
            rms = float(feats.rms)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        DETECT_LATENCY.labels(service=_SERVICE).observe(latency_ms / 1000.0)

        inf = InferenceMsg(
            source_id=msg.source_id,
            ts=msg.ts,
            modality="audio",
            label=det.label,
            confidence=det.confidence,
            p_drone=det.p_drone,
            bbox=None,
            track_id=None,
            model=ModelRef(name=self._detector.model_name, ver=self._detector.model_ver),
            det_latency_ms=latency_ms,
            ingest_ts=msg.ts,
            quality=QualityHint(snr_db=snr, audio_rms=rms),
        )
        self.publish(Topics.INFERENCE, msg.source_id, inf)
        MESSAGES_TOTAL.labels(service=_SERVICE, topic=Topics.INFERENCE).inc()
