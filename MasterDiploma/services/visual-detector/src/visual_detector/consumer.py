"""VideoConsumer — consumer-сервис визуального детектора.

Поток обработки одного сообщения video.raw:
  1) декод JPEG-payload -> кадр (numpy BGR);
  2) предобработка (на MVP — no-op);
  3) детекция YOLOv8 -> список боксов с уверенностью;
  4) (опц.) ByteTrack -> track_id;
  5) формирование InferenceMsg (modality=video) -> публикация в топик `inference`.

Стратегия выбора результата на MVP: если есть хотя бы одна детекция «дрон» —
публикуем самую уверенную (label="drone", её bbox/track_id); иначе публикуем одно
сообщение label="non-drone" с confidence=0 (fusion-движку нужен факт «кадр обработан,
дрон не виден», чтобы окно выравнивания не зависало).
"""

from __future__ import annotations

import time

from uavdet_common.consumer_service import KafkaConsumerService
from uavdet_common.messages import InferenceMsg, ModelRef, QualityHint, Topics, VideoRawMsg
from uavdet_common.metrics import DETECT_LATENCY, MESSAGES_TOTAL

from .decode import FrameDecoder
from .detector import Detection, YoloDetector
from .preprocess import PreprocessChain
from .tracker import ByteTrackTracker

_SERVICE = "visual-detector"


class VideoConsumer(KafkaConsumerService):
    """video.raw -> inference (modality=video)."""

    in_topic = Topics.VIDEO_RAW
    in_model = VideoRawMsg

    def __init__(
        self,
        bus,
        *,
        group_id: str,
        detector: YoloDetector,
        tracker: ByteTrackTracker | None,
        decoder: FrameDecoder | None = None,
        preprocess: PreprocessChain | None = None,
    ) -> None:
        super().__init__(bus)
        self.group_id = group_id
        self._detector = detector
        self._tracker = tracker
        self._decoder = decoder or FrameDecoder()
        self._preprocess = preprocess or PreprocessChain.identity()

    def on_start(self) -> None:
        self._log.info(
            "visual-detector: старт",
            model=self._detector.model_name,
            model_ver=self._detector.model_ver,
            tracker="on" if self._tracker else "off",
        )

    def process(self, key: str | None, msg: VideoRawMsg) -> None:  # type: ignore[override]
        if msg.payload_kind != "jpeg" or not msg.payload:
            # claim-check (payload по ссылке) на MVP не поддержан — пропускаем
            self._log.warning("visual-detector: пропуск сообщения без inline JPEG", payload_kind=msg.payload_kind)
            return

        decoded = self._decoder.decode_jpeg_b64(msg.payload)
        sharpness, brightness = self._decoder.quality_hint(decoded)
        quality = QualityHint(img_sharpness=sharpness, img_brightness=brightness)
        frame = self._preprocess(decoded)
        result = self._detector.detect(frame)
        DETECT_LATENCY.labels(service=_SERVICE).observe(result.latency_ms / 1000.0)

        drone_dets = [d for d in result.detections if d.label == "drone"]
        if drone_dets:
            best, track_id = self._pick_best(msg.source_id, drone_dets)
            inf = InferenceMsg(
                source_id=msg.source_id,
                ts=msg.ts,
                modality="video",
                label="drone",
                confidence=best.confidence,
                bbox=best.bbox,
                track_id=track_id,
                model=ModelRef(name=result.model_name, ver=result.model_ver),
                det_latency_ms=result.latency_ms,
                ingest_ts=msg.ts,
                quality=quality,
            )
        else:
            inf = InferenceMsg(
                source_id=msg.source_id,
                ts=msg.ts,
                modality="video",
                label="non-drone",
                confidence=0.0,
                bbox=None,
                track_id=None,
                model=ModelRef(name=result.model_name, ver=result.model_ver),
                det_latency_ms=result.latency_ms,
                ingest_ts=msg.ts,
                quality=quality,
            )

        self.publish(Topics.INFERENCE, msg.source_id, inf)
        MESSAGES_TOTAL.labels(service=_SERVICE, topic=Topics.INFERENCE).inc()

    def _pick_best(self, source_id: str, dets: list[Detection]) -> tuple[Detection, int | None]:
        """Выбрать самую уверенную детекцию; при включённом трекере — добавить track_id."""
        if self._tracker is None:
            best = max(dets, key=lambda d: d.confidence)
            return best, None
        tracked = self._tracker.update(source_id, dets)
        # самая уверенная среди (детекция, track_id)
        best_pair = max(tracked, key=lambda pair: pair[0].confidence)
        return best_pair[0], best_pair[1]
