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
        # it-52: история центров треков для признака движения {(source_id, track_id): [(media_ts, cx, cy), …]}
        self._track_hist: dict[tuple[str, int], list[tuple[float, float, float]]] = {}

    def on_start(self) -> None:
        self._log.info(
            "visual-detector: старт",
            model=self._detector.model_name,
            model_ver=self._detector.model_ver,
            tracker="on" if self._tracker else "off",
        )

    def process(self, key: str | None, msg: VideoRawMsg) -> None:  # type: ignore[override]
        detect_start = time.time()  # it-47: разложение e2e — начало обработки детектором
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
            best, track_id, motion_score = self._pick_best(msg.source_id, drone_dets, msg.media_ts, msg.meta.w)
            if motion_score is not None:
                quality.motion_score = motion_score
            inf = InferenceMsg(
                source_id=msg.source_id,
                ts=msg.ts,
                media_ts=msg.media_ts,
                modality="video",
                label="drone",
                confidence=best.confidence,
                bbox=best.bbox,
                track_id=track_id,
                model=ModelRef(name=result.model_name, ver=result.model_ver),
                det_latency_ms=result.latency_ms,
                ingest_ts=msg.ts,
                detect_start_ts=detect_start,
                detect_done_ts=time.time(),
                quality=quality,
            )
        else:
            inf = InferenceMsg(
                source_id=msg.source_id,
                ts=msg.ts,
                media_ts=msg.media_ts,
                modality="video",
                label="non-drone",
                confidence=0.0,
                bbox=None,
                track_id=None,
                model=ModelRef(name=result.model_name, ver=result.model_ver),
                det_latency_ms=result.latency_ms,
                ingest_ts=msg.ts,
                detect_start_ts=detect_start,
                detect_done_ts=time.time(),
                quality=quality,
            )

        self.publish(Topics.INFERENCE, msg.source_id, inf)
        MESSAGES_TOTAL.labels(service=_SERVICE, topic=Topics.INFERENCE).inc()

    def _pick_best(self, source_id: str, dets: list[Detection], media_ts: float,
                   frame_w: int) -> tuple[Detection, int | None, float | None]:
        """Выбрать самую уверенную детекцию; при включённом трекере — track_id и motion_score.

        motion_score (it-53) = средняя скорость центра, нормированная на ШИРИНУ БОКСА цели
        (скоростей «боксов в секунду»), клампится в [0, 1]. Нормировка на bbox, а не на кадр:
        джиттер детектора масштабируется с размером бокса (близкий план — большой бокс —
        большой джиттер), а реальное перемещение дрона — нет; it-52 показала, что нормировка
        на кадр не отделяет «стоит крупным планом» от «летит». 0 — статичная цель, ~1 —
        цель пролетает собственную длину за секунду.
        """
        if self._tracker is None:
            best = max(dets, key=lambda d: d.confidence)
            return best, None, None
        tracked = self._tracker.update(source_id, dets)
        best_pair = max(tracked, key=lambda pair: pair[0].confidence)
        best, tid = best_pair
        motion: float | None = None
        if tid is not None and media_ts is not None:
            cx, cy = best.bbox[0] + best.bbox[2] / 2.0, best.bbox[1] + best.bbox[3] / 2.0
            hist = self._track_hist.setdefault((source_id, tid), [])
            hist.append((media_ts, cx, cy, max(1.0, best.bbox[2])))
            while len(hist) > 8:
                hist.pop(0)
            if len(hist) >= 2:
                (t0, x0, y0, w0), (t1, x1, y1, w1) = hist[0], hist[-1]
                dt = t1 - t0
                if dt > 1e-6:
                    speed = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt   # px/с
                    bbox_w = max(w0, w1)                                     # масштаб цели
                    motion = max(0.0, min(1.0, speed / max(1.0, bbox_w)))
        return best, tid, motion
