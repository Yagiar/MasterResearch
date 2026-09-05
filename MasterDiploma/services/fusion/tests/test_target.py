"""Тесты целевой переменной fusion: presence | airborne (it-52, ревью GPT-6-Astra §3).

Инвариант it-51 (негативная сессия: дрон стоит, моторы выключены, GT airborne=0):
в режиме airborne статичное видео-подтверждение не должно давать положительного решения.
"""

from __future__ import annotations

import pytest
from fusion.strategies.late import LateFusion
from fusion.target import apply_target_filter, validate_target
from fusion.window_buffer import AlignedWindow
from uavdet_common.messages import InferenceMsg, QualityHint


def _msg(modality: str, label: str, conf: float, media_ts: float, motion: float | None = None) -> InferenceMsg:
    q = QualityHint(motion_score=motion) if modality == "video" else QualityHint()
    return InferenceMsg(source_id="cam-01", modality=modality, label=label, confidence=conf,
                        ts=1000.0 + media_ts, media_ts=media_ts, quality=q,
                        msg_id=f"{modality}@{media_ts}")


def _window(video_motion: float | None, p_a: float) -> AlignedWindow:
    """video_motion=None — видео есть, но motion_score не замерен (трекер выключен)."""
    win = AlignedWindow(source_id="cam-01", t0=41.5, t1=42.7)
    v = InferenceMsg(source_id="cam-01", modality="video", label="drone", confidence=0.86,
                     ts=1042.0, media_ts=42.0, quality=QualityHint(motion_score=video_motion),
                     msg_id="v@42")
    win.video = [v]
    win.audio = [_msg("audio", "non-drone", 0.9, 42.0)]  # p_drone=0.0 (non-drone)
    win.joint = True
    return win


def test_validate_target() -> None:
    assert validate_target("presence") == "presence"
    assert validate_target("airborne") == "airborne"
    with pytest.raises(ValueError):
        validate_target("hovering")


def test_presence_mode_untouched() -> None:
    """presence (дефолт): фильтр не применяется даже при нулевом движении."""
    win = _window(video_motion=0.0, p_a=0.0)
    out = apply_target_filter(win, target="presence", motion_floor=0.15)
    assert out.best_video().label == "drone"
    assert out.best_video().confidence == pytest.approx(0.86)


def test_airborne_static_drone_becomes_non_drone() -> None:
    """airborne: стоящий дрон (motion=0) → видео «не дрон», слияние отвергает."""
    win = _window(video_motion=0.0, p_a=0.0)
    out = apply_target_filter(win, target="airborne", motion_floor=0.15)
    bv = out.best_video()
    assert bv.label == "non-drone" and bv.confidence == pytest.approx(0.0)
    # стратегия на отфильтрованном окне: p_fused = p_a = 0.0 → отрицательное решение
    res = LateFusion().fuse(out, w_v=0.5, w_a=0.5, threshold=0.5)
    assert res.decision is False


def test_airborne_flying_drone_unchanged() -> None:
    """airborne: летящий дрон (motion ≥ floor) — видео-подтверждение сохраняется."""
    win = _window(video_motion=0.6, p_a=0.0)
    out = apply_target_filter(win, target="airborne", motion_floor=0.15)
    assert out.best_video().label == "drone"
    assert out.best_video().confidence == pytest.approx(0.86)


def test_airborne_unknown_motion_not_filtered() -> None:
    """motion_score неизвестен (трекер выключен) — фильтр не рискует утверждать «не летит»."""
    win = _window(video_motion=None, p_a=0.0)
    out = apply_target_filter(win, target="airborne", motion_floor=0.15)
    assert out.best_video().label == "drone"


def test_airborne_audio_only_window_untouched() -> None:
    """Окно без видео фильтр не трогает (акустическая ветка работает как раньше)."""
    win = _window(video_motion=None, p_a=0.0)
    win.video = []
    out = apply_target_filter(win, target="airborne", motion_floor=0.15)
    assert out.best_video() is None and out.best_audio() is not None


def test_it51_scenario_end_to_end() -> None:
    """Сценарий it-51 в обоих режимах: p_v=0.86, стоящий дрон (motion=0.02), сглаженный p_a=0.22."""
    for target, expected_decision in (("presence", True), ("airborne", False)):
        win = AlignedWindow(source_id="cam-01", t0=41.5, t1=42.7)
        win.video = [_msg("video", "drone", 0.86, 42.0, motion=0.02)]
        # сглаженный аудио-канал: p_a=0.22 (после медианы, it-51)
        a = _msg("audio", "drone", 0.22, 42.0)
        win.audio = [a]
        win.joint = True
        out = apply_target_filter(win, target=target, motion_floor=0.15)
        f = LateFusion(delta_conf=0.0, delta_unconf=0.0)  # рабочий контур it-43: Δ=0
        res = f.fuse(out, w_v=0.5, w_a=0.5, threshold=0.5)
        assert res.decision is expected_decision, (target, res.p_fused)
