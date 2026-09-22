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
        sahi_slice=int(vd_cfg.get("sahi_slice", 0)),
        sahi_overlap=float(vd_cfg.get("sahi_overlap", 0.2)),
        drone_class_ids=(list(drone_ids) if drone_ids else None),
        vote_weights_path=(str(vd_cfg["vote_weights_path"]) if vd_cfg.get("vote_weights_path") else None),
        vote_mode=str(vd_cfg.get("vote_mode", "off")),
        vote_floor=float(vd_cfg.get("vote_floor", 0.4)),
    )


def build_tracker(vd_cfg: dict[str, Any]) -> ByteTrackTracker | None:
    """Создать трекер, если включён в конфиге; иначе None."""
    if not bool(vd_cfg.get("tracker_enabled", True)):
        return None
    return ByteTrackTracker()
