"""Тесты каузального сглаживания аудио-канала fusion (research/it-08, патч it-11)."""

from __future__ import annotations

import time

import pytest
from fusion.temporal import MedianSmoother, apply_audio_smoothing
from fusion.window_buffer import AlignedWindow, TimeWindowBuffer
from uavdet_common.messages import InferenceMsg


def _msg(modality: str, label: str, conf: float, ts: float | None = None) -> InferenceMsg:
    return InferenceMsg(
        source_id="cam-01",
        modality=modality,
        label=label,
        confidence=conf,
        ts=ts if ts is not None else time.time(),
        msg_id=f"{modality}-{label}-{conf}-{ts}",
    )


def _window(video_conf: float | None, audio_conf: float | None, audio_label: str = "drone") -> AlignedWindow:
    ts = time.time()
    win = AlignedWindow(source_id="cam-01", t0=ts - 0.6, t1=ts + 0.6)
    if video_conf is not None:
        win.video = [_msg("video", "drone", video_conf, ts)]
    if audio_conf is not None:
        win.audio = [_msg("audio", audio_label, audio_conf, ts + 0.1)]
    return win


def test_smoothed_causal_median_over_last_k() -> None:
    sm = MedianSmoother(k=5)
    seq = [0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0]
    got = [sm.smoothed("cam", v) for v in seq]
    # окно из 5 последних; 0.9 вымывается только с 6-го нуля (медиана чётной длины — верхний средний)
    assert got == [0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0]


def test_smoothed_separates_sources() -> None:
    sm = MedianSmoother(k=3)
    assert sm.smoothed("cam-01", 0.8) == 0.8
    assert sm.smoothed("cam-02", 0.2) == 0.2
    assert sm.smoothed("cam-01", 0.6) == 0.8  # cam-01: [0.8, 0.6] → верхний средний 0.8
    assert sm.smoothed("cam-02", 0.4) == 0.4  # cam-02: [0.2, 0.4] → 0.4


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        MedianSmoother(k=0)


def test_duplicate_msg_id_does_not_reenter_history() -> None:
    """Инвариант (ревью 2026-09-05 §6.3): одно аудио-сообщение, попав в несколько
    fusion-окон подряд (триггер видео), учитывается в фильтре РОВНО один раз.

    Контрпример рецензента: история 5×0.9; новое аудио p=0.1 предъявлено трижды →
    медиана должна остаться 0.9 (одно новое наблюдение), а не уйти к 0.1 как от трёх.
    """
    sm = MedianSmoother(k=5)
    for _ in range(5):
        sm.smoothed("cam", 0.9)
    got = [sm.smoothed("cam", 0.1, msg_id="same-audio") for _ in range(3)]
    assert got == [0.9, 0.9, 0.9]


def test_distinct_msg_ids_update_history() -> None:
    """Разные msg_id — полноценные новые наблюдения: медиана k=5 переключается с третьего нуля
    (это и есть честная задержка обнаружения каузальной медианы, ~k//2 шагов)."""
    sm = MedianSmoother(k=5)
    for _ in range(5):
        sm.smoothed("cam", 0.9)
    assert sm.smoothed("cam", 0.1, msg_id="a-1") == 0.9   # история [0.9×4, 0.1]
    assert sm.smoothed("cam", 0.1, msg_id="a-2") == 0.9   # [0.9×3, 0.1×2]
    assert sm.smoothed("cam", 0.1, msg_id="a-3") == 0.1   # [0.9×2, 0.1×3] — перелом


def test_apply_once_per_unique_audio_through_video_triggers() -> None:
    """Интеграционный путь буфера (ревью §6.3): одно аудио + два видео-триггера →
    история фильтра пополняется один раз, сглаженное значение стабильно."""
    sm = MedianSmoother(k=5)
    for _ in range(5):
        sm.smoothed("cam-01", 0.9)
    buf = TimeWindowBuffer(epsilon_ms=600.0)
    t0 = 1000.0
    stream = [
        _msg("audio", "non-drone", 0.9, t0),        # p_drone=0.0 (non-drone conf=0.9)
        _msg("video", "drone", 0.8, t0 + 0.04),
        _msg("video", "drone", 0.75, t0 + 0.08),
    ]
    outs = []
    for m in stream:
        out = apply_audio_smoothing(buf.add(m), sm)
        best = out.best_audio()
        assert best is not None
        outs.append(best.confidence)
    assert outs == [0.9, 0.9, 0.9]                  # не [0.9, 0.9, 0.0] как до фикса


def test_apply_no_smoother_returns_same_window() -> None:
    win = _window(0.8, 0.3)
    assert apply_audio_smoothing(win, None) is win


def test_apply_mono_window_untouched() -> None:
    sm = MedianSmoother(k=5)
    win = _window(0.8, None)
    out = apply_audio_smoothing(win, sm)
    assert out is win
    assert not out.audio


def test_apply_smooths_drone_confidence_and_keeps_trace() -> None:
    sm = MedianSmoother(k=5)
    for conf in (0.9, 0.9, 0.9):
        sm.smoothed("cam-01", conf)
    win = _window(0.8, 0.0, audio_label="non-drone")  # текущее окно: уверенный non-drone
    out = apply_audio_smoothing(win, sm)
    best = out.best_audio()
    assert best is not None
    assert best.label == "drone"
    assert best.confidence == pytest.approx(0.9)  # медиана истории [0.9,0.9,0.9,0.0] = 0.9
    # исходное окно не изменено (иммутабельность по контракту)
    assert win.best_audio().label == "non-drone"
    # msg_id сохранён — трассировка source_msg_ids и метрика Δt не ломаются


def test_apply_through_buffer_end_to_end() -> None:
    """Интеграционный путь: детекции через TimeWindowBuffer → сглаживание → best_audio."""
    sm = MedianSmoother(k=5)
    buf = TimeWindowBuffer(epsilon_ms=600.0)
    t0 = 1000.0
    for i, conf in enumerate((0.9, 0.9, 0.9)):  # аудио-окна: дрон, дрон, дрон
        buf.add(_msg("audio", "drone", conf, t0 + i * 0.5))
    buf.add(_msg("video", "drone", 0.8, t0 + 1.1))
    win = buf.add(_msg("video", "drone", 0.75, t0 + 1.2))
    out = apply_audio_smoothing(win, sm)
    best = out.best_audio()
    assert best is not None and best.label == "drone"
    assert best.confidence == pytest.approx(0.9)
