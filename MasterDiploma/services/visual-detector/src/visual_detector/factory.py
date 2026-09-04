"""DetectorFactory — сборка YoloDetector и (опц.) ByteTrackTracker по конфигу
(паттерн Factory). Скрывает выбор бэкенда детектора (на пилоте — только YOLOv8).
"""

from __future__ import annotations

from typing import Any

from .detector import YoloDetector
from .tracker import ByteTrackTracker


def build_detector(vd_cfg: dict[str, Any]) -> YoloDetector:
    """Создать детектор по секции `visual_detector` из конфига."""
    drone_ids = vd_cfg.get("drone_class_ids")
    return YoloDetector(
        weights_path=str(vd_cfg.get("weights_path", "/models/visual/yolov8s-uav.pt")),
        conf_threshold=float(vd_cfg.get("conf_threshold", 0.25)),
        iou_threshold=float(vd_cfg.get("iou_threshold", 0.45)),
        device=str(vd_cfg.get("device", "cpu")),
        imgsz=int(vd_cfg.get("imgsz", 640)),
        drone_class_ids=(list(drone_ids) if drone_ids else None),
    )


def build_tracker(vd_cfg: dict[str, Any]) -> ByteTrackTracker | None:
    """Создать трекер, если включён в конфиге; иначе None."""
    if not bool(vd_cfg.get("tracker_enabled", True)):
        return None
    return ByteTrackTracker()
