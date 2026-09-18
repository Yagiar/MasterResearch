"""Тесты SAHI-нарезки детектора (it-59): сетка срезов, NMS-мердж, координаты."""

from __future__ import annotations

import numpy as np
import pytest

from visual_detector.detector import Detection, YoloDetector


def _det(label: str, conf: float, x: float, y: float, w: float, h: float) -> Detection:
    return Detection(label=label, confidence=conf, bbox=[x, y, w, h])


def test_sahi_grid_covers_frame() -> None:
    d = YoloDetector.__new__(YoloDetector)
    d._sahi_slice = 640
    d._sahi_overlap = 0.2
    d._model_name = "test"
    slices = d._sahi_slices(2560, 960)
    # шаг 512: xs 0..1920 (5), ys 0..320 (2) → 10 срезов; правый/нижний край подрезается
    assert len(slices) == 10
    assert all(sw <= 640 and sh <= 640 for _, _, sw, sh in slices)
    assert sum(x + sw for x, _, sw, _ in slices) >= 2560  # правый край покрыт
    assert sum(y + sh for _, y, _, sh in slices) >= 960   # нижний край покрыт


def test_sahi_grid_small_frame_single_slice() -> None:
    d = YoloDetector.__new__(YoloDetector)
    d._sahi_slice = 640
    d._sahi_overlap = 0.2
    d._model_name = "test"
    slices = d._sahi_slices(500, 400)
    assert slices == [(0, 0, 500, 400)]


def test_nms_merge_dedups_cross_slice() -> None:
    """Одна цель видна в двух срезах: дубль с меньшим conf отбрасывается."""
    dets = [_det("drone", 0.9, 100, 100, 50, 50),
            _det("drone", 0.7, 102, 101, 50, 50),   # IoU > 0.45 с первой
            _det("drone", 0.5, 900, 900, 40, 40)]   # отдельная цель
    out = YoloDetector._nms_merge(dets, iou_thr=0.45)
    assert len(out) == 2
    assert {round(d.confidence, 1) for d in out} == {0.9, 0.5}


def test_nms_merge_keeps_distinct() -> None:
    dets = [_det("drone", 0.8, 0, 0, 30, 30), _det("drone", 0.6, 500, 500, 30, 30)]
    assert len(YoloDetector._nms_merge(dets, iou_thr=0.45)) == 2


def test_detect_sahi_maps_global_coords(monkeypatch) -> None:
    """SAHI-прогон: детекция из среза получает глобальные координаты (offset + NMS)."""
    d = YoloDetector.__new__(YoloDetector)
    d._conf = 0.25
    d._iou = 0.45
    d._device = "cpu"
    d._imgsz = 320
    d._sahi_slice = 320
    d._sahi_overlap = 0.0
    d._names = {0: "drone"}
    d._drone_ids = {0}
    d._is_surrogate = False
    d._model_name = "test"

    seen = []

    def fake2(frame):
        h, w = frame.shape[:2]
        seen.append((w, h))
        # детекция только на правом срезе (локальные координаты 100..140 => глобальные 420..460)
        if w == 320 and seen.count((320, 320)) == 2:
            return [_det("drone", 0.8, 100, 100, 40, 40)], 1.0
        return [], 1.0

    monkeypatch.setattr(d, "_detect_frame", fake2)
    frame = np.zeros((320, 640, 3), dtype=np.uint8)
    res = d.detect(frame)
    assert len(res.detections) == 1
    d0 = res.detections[0]
    assert d0.bbox[0] == 420.0 and d0.bbox[1] == 100.0   # 320 (offset среза) + 100 (локальный)
