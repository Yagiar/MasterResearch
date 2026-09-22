"""Тесты кадрового голосования двух моделей (it-81): AND/OR по max_conf дронов."""

from __future__ import annotations

from visual_detector.detector import Detection, YoloDetector


def _det(label: str, conf: float) -> Detection:
    return Detection(label=label, confidence=conf, bbox=[0.0, 0.0, 10.0, 10.0])


def _v(vote_mode: str, floor: float = 0.4) -> YoloDetector:
    d = YoloDetector.__new__(YoloDetector)
    d._vote_mode = vote_mode
    d._vote_floor = floor
    d._model_name = "yolov8s-uav"
    d._voter_name = "uav-yolov8s-bg-best"
    return d


def test_and_pass_keeps_primary() -> None:
    a = [_det("drone", 0.6)]
    b = [_det("drone", 0.45)]
    assert _v("and")._apply_vote(a, b) == a


def test_and_blocks_when_voter_low() -> None:
    a = [_det("drone", 0.9), _det("non-drone", 0.3)]
    b: list[Detection] = []
    out = _v("and")._apply_vote(a, b)
    assert [d.label for d in out] == ["non-drone"]  # drone-детекции срезаны, non-drone остались


def test_and_blocks_when_primary_below_floor() -> None:
    a = [_det("drone", 0.35)]  # ниже floor, но выше conf_threshold
    b = [_det("drone", 0.9)]
    assert _v("and")._apply_vote(a, b) == []


def test_or_adopts_voter_when_primary_silent() -> None:
    a: list[Detection] = [_det("non-drone", 0.3)]
    b = [_det("drone", 0.7)]
    out = _v("or")._apply_vote(a, b)
    assert out == b


def test_or_keeps_primary_when_stronger() -> None:
    a = [_det("drone", 0.8)]
    b = [_det("drone", 0.5)]
    assert _v("or")._apply_vote(a, b) == a


def test_or_negative_when_both_below_floor() -> None:
    a = [_det("drone", 0.3)]
    b = [_det("drone", 0.2)]
    assert _v("or")._apply_vote(a, b) == []


def test_floor_boundary_inclusive() -> None:
    a = [_det("drone", 0.4)]
    b = [_det("drone", 0.4)]
    assert _v("and")._apply_vote(a, b) == a


def test_model_name_reports_composition() -> None:
    assert _v("and").model_name == "yolov8s-uav+uav-yolov8s-bg-best:and@0.4"
