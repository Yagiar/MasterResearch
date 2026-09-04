"""ByteTrackTracker — трекинг детекций между кадрами (per source_id), на базе
`supervision.ByteTrack`. Назначает устойчивые `track_id` боксам, что снижает
ложные срабатывания/мерцание (как в ВКР: связка YOLO + ByteTrack + обратная связь).

На MVP трекер можно выключить (`visual_detector.tracker_enabled: false`) — тогда
`track_id` в InferenceMsg остаётся None.
"""

from __future__ import annotations

import numpy as np

from .detector import Detection


class ByteTrackTracker:
    """Обёртка над supervision.ByteTrack с состоянием на каждый source_id."""

    def __init__(self) -> None:
        import supervision as sv  # noqa: PLC0415

        self._sv = sv
        self._trackers: dict[str, "sv.ByteTrack"] = {}

    def _tracker_for(self, source_id: str):
        tr = self._trackers.get(source_id)
        if tr is None:
            tr = self._sv.ByteTrack()
            self._trackers[source_id] = tr
        return tr

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """IoU двух боксов в формате xyxy."""
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def update(self, source_id: str, detections: list[Detection]) -> list[tuple[Detection, int | None]]:
        """Прогнать детекции кадра через трекер источника; вернуть пары (детекция, track_id|None).

        ByteTrack может отбросить/переупорядочить детекции, поэтому сопоставляем входные
        детекции с возвращёнными tracked-боксами по IoU (порог 0.3), а не по индексу.
        """
        if not detections:
            return []
        sv = self._sv
        in_xyxy = np.array(
            [[d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]] for d in detections],
            dtype=float,
        )
        det = sv.Detections(
            xyxy=in_xyxy,
            confidence=np.array([d.confidence for d in detections], dtype=float),
            class_id=np.zeros(len(detections), dtype=int),  # один обобщённый класс «цель»
        )
        tracked = self._tracker_for(source_id).update_with_detections(det)

        # tracked.xyxy / tracked.tracker_id — numpy-массивы (могут быть пустыми)
        tr_xyxy = np.asarray(getattr(tracked, "xyxy", np.empty((0, 4))))
        tr_ids = getattr(tracked, "tracker_id", None)
        tr_ids = np.asarray(tr_ids) if tr_ids is not None else np.empty((0,), dtype=int)

        out: list[tuple[Detection, int | None]] = []
        for i, d in enumerate(detections):
            tid: int | None = None
            if tr_xyxy.shape[0] > 0:
                ious = np.array([self._iou(in_xyxy[i], tr_xyxy[j]) for j in range(tr_xyxy.shape[0])])
                j = int(np.argmax(ious))
                if ious[j] >= 0.3 and j < tr_ids.shape[0]:
                    tid = int(tr_ids[j])
            out.append((d, tid))
        return out
