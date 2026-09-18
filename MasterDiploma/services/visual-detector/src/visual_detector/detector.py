"""YoloDetector — обёртка над Ultralytics YOLOv8 (реализует контракт детектора).

Поведение:
  - грузит веса из `weights_path` (целевые `yolov8s-uav.pt` из models/visual/),
    либо, если файла нет, — предобученные `yolov8n.pt` (COCO) как заглушка на MVP;
  - на кадре возвращает список детекций `Detection(label, confidence, bbox=[x,y,w,h])`;
    `label`: "drone" если сработал «дрон-класс» (на спец-весах) — а на COCO-заглушке
    как суррогат используется класс `airplane` (~летящий объект), всё прочее — отбрасываем;
  - класс/уверенность фильтруются по `conf_threshold`.

`bbox` — в пикселях исходного кадра, формат `[x, y, w, h]` (левый-верх + размеры) —
совпадает с тем, что ожидает `InferenceMsg.bbox`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uavdet_common.logging import get_logger

log = get_logger("visual-detector")

# имена COCO-классов, считающихся суррогатом «летящего объекта» для MVP-заглушки
_COCO_SURROGATE_CLASSES = {"airplane", "bird", "kite"}
_FALLBACK_WEIGHTS = "yolov8n.pt"


@dataclass(frozen=True)
class Detection:
    label: str  # "drone" | "non-drone"
    confidence: float
    bbox: list[float]  # [x, y, w, h] в пикселях исходного кадра


@dataclass(frozen=True)
class DetectResult:
    detections: list[Detection]
    latency_ms: float
    model_name: str
    model_ver: str


class YoloDetector:
    """Детектор на Ultralytics YOLOv8 (или совместимые версии YOLO)."""

    def __init__(
        self,
        *,
        weights_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
        drone_class_ids: list[int] | None = None,  # явный список id классов = «дрон» (если None — по имени класса)
        sahi_slice: int = 0,        # it-59: размер среза SAHI (0 = выключено); напр. 640
        sahi_overlap: float = 0.2,  # it-59: перекрытие срезов (доля)
    ) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        path = Path(weights_path)
        if path.exists():
            self._model_name = path.stem
            self._is_surrogate = False
            log.info("visual-detector: загрузка весов", weights=str(path))
            self._model = YOLO(str(path))
        else:
            self._model_name = _FALLBACK_WEIGHTS
            self._is_surrogate = True
            log.warning(
                "visual-detector: веса не найдены — использую предобученные COCO как заглушку",
                requested=str(path),
                fallback=_FALLBACK_WEIGHTS,
            )
            self._model = YOLO(_FALLBACK_WEIGHTS)

        self._conf = float(conf_threshold)
        self._iou = float(iou_threshold)
        self._device = device
        self._imgsz = int(imgsz)
        self._sahi_slice = max(0, int(sahi_slice))
        self._sahi_overlap = min(0.9, max(0.0, float(sahi_overlap)))
        self._names: dict[int, str] = dict(getattr(self._model, "names", {}) or {})
        self._drone_ids: set[int] | None = set(int(i) for i in drone_class_ids) if drone_class_ids else None
        log.info(
            "visual-detector: классы модели",
            names=self._names,
            drone_class_ids=(sorted(self._drone_ids) if self._drone_ids is not None else "по имени класса"),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_ver(self) -> str:
        return "surrogate-coco" if self._is_surrogate else "1"

    def _label_for(self, cls_id: int) -> str | None:
        """Сопоставить id класса YOLO -> метку схемы ("drone"/"non-drone") или None (отбросить).

        Приоритет: явный `drone_class_ids` (если задан) -> иначе имя класса (содержит "drone"/"uav").
        На COCO-заглушке: "летящие" классы (airplane/bird/kite) -> суррогат "drone", остальное -> отбрасываем.
        """
        cls_id = int(cls_id)
        if self._is_surrogate:
            return "drone" if self._names.get(cls_id, "") in _COCO_SURROGATE_CLASSES else None
        if self._drone_ids is not None:
            return "drone" if cls_id in self._drone_ids else "non-drone"
        low = self._names.get(cls_id, "").lower()
        return "drone" if ("drone" in low or "uav" in low) else "non-drone"

    def _detect_frame(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Один прогон детектора на кадре (или срезе); возвращает детекции в координатах входа."""
        t0 = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        detections: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                label = self._label_for(int(cls_id))
                if label is None:
                    continue
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(conf),
                        bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    )
                )
        return detections, latency_ms

    def _sahi_slices(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """Сетка срезов (x, y, w, h) размера slice с перекрытием (it-59).

        Крайний срез прижимается к правому/нижнему краю, чтобы покрыть кадр целиком.
        """
        step = max(1, int(self._sahi_slice * (1.0 - self._sahi_overlap)))

        def starts(total: int) -> list[int]:
            if total <= self._sahi_slice:
                return [0]
            pts = list(range(0, total - self._sahi_slice + 1, step))
            if pts[-1] != total - self._sahi_slice:
                pts.append(total - self._sahi_slice)
            return pts

        return [(x, y, min(self._sahi_slice, w - x), min(self._sahi_slice, h - y))
                for y in starts(h) for x in starts(w)]

    @staticmethod
    def _nms_merge(dets: list[Detection], iou_thr: float) -> list[Detection]:
        """NMS-мердж детекций со всех срезов (координаты уже глобальные)."""
        keep: list[Detection] = []
        for d in sorted(dets, key=lambda d: -d.confidence):
            x1, y1, w, h = d.bbox
            x2, y2 = x1 + w, y1 + h
            dup = False
            for k in keep:
                kx1, ky1, kw, kh = k.bbox
                kx2, ky2 = kx1 + kw, ky1 + kh
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = w * h + kw * kh - inter
                if union > 0 and inter / union >= iou_thr:
                    dup = True
                    break
            if not dup:
                keep.append(d)
        return keep

    def detect(self, frame: np.ndarray) -> DetectResult:
        if self._sahi_slice > 0:
            H, W = frame.shape[:2]
            all_dets: list[Detection] = []
            t0 = time.perf_counter()
            for x, y, sw, sh in self._sahi_slices(W, H):
                tile = frame[y:y + sh, x:x + sw]
                dets, _ = self._detect_frame(tile)
                for d in dets:
                    # координаты среза -> глобальные
                    all_dets.append(Detection(label=d.label, confidence=d.confidence,
                                              bbox=[d.bbox[0] + x, d.bbox[1] + y, d.bbox[2], d.bbox[3]]))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return DetectResult(
                detections=self._nms_merge(all_dets, self._iou),
                latency_ms=latency_ms,
                model_name=self._model_name,
                model_ver=self.model_ver,
            )
        dets, latency_ms = self._detect_frame(frame)
        return DetectResult(
            detections=dets,
            latency_ms=latency_ms,
            model_name=self._model_name,
            model_ver=self.model_ver,
        )
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                label = self._label_for(int(cls_id))
                if label is None:
                    continue
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(conf),
                        bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    )
                )
        return DetectResult(
            detections=detections,
            latency_ms=latency_ms,
            model_name=self._model_name,
            model_ver=self.model_ver,
        )
