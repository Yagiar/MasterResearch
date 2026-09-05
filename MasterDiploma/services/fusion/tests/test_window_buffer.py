"""Тесты TimeWindowBuffer: политика опоздания и инвариант порядка доставки (it-33, ревью §6.2)."""

from __future__ import annotations

import pytest
from fusion.window_buffer import TimeWindowBuffer
from uavdet_common.messages import InferenceMsg


def _msg(modality: str, ts: float) -> InferenceMsg:
    return InferenceMsg(source_id="cam-01", modality=modality, label="drone", confidence=0.9,
                        ts=ts, msg_id=f"{modality}@{ts}")


def test_in_order_stream_forms_joint_window() -> None:
    """Обычный (упорядоченный) поток — совместное окно образуется, поведение прежнее."""
    buf = TimeWindowBuffer(epsilon_ms=600.0)
    buf.add(_msg("video", 10.0))
    win = buf.add(_msg("audio", 10.05))
    assert win.joint
    assert win.best_video() is not None and win.best_audio() is not None


def test_arrival_order_does_not_change_joint_windows() -> None:
    """Инвариант (ревью §6.2): перестановка доставки не меняет число совместных окон.

    Контрпример рецензента при ε=0.6: `video@10 → audio@10 → video@12` давал совместное
    окно, а `video@10 → video@12 → audio@10` — нет (видео@10 уже выселен). С горизонтом
    опоздания audio@10 в обоих случаях находит video@10 в удержанной истории.
    """
    counts = []
    for stream in ([("video", 10.0), ("audio", 10.0), ("video", 12.0)],
                   [("video", 10.0), ("video", 12.0), ("audio", 10.0)]):
        buf = TimeWindowBuffer(epsilon_ms=600.0)
        wins = [buf.add(_msg(m, t)) for m, t in stream]
        counts.append(sum(w.joint for w in wins))
    # совместное окно ровно одно в обоих порядках — в т.ч. для опоздавшего audio@10
    assert counts == [1, 1]


def test_message_beyond_lateness_horizon_is_flagged_late() -> None:
    """Опоздание за горизонт: пара уже невосстановима — окно моно, флаг late поднят."""
    buf = TimeWindowBuffer(epsilon_ms=600.0, lateness_ms=2000.0)
    buf.add(_msg("video", 10.0))
    buf.add(_msg("video", 20.0))       # video@10 выселен (cutoff = 20 − 0.6 − 2.0 = 17.4)
    win = buf.add(_msg("audio", 10.0))
    assert win.late
    assert not win.joint


def test_within_lateness_horizon_not_flagged_late() -> None:
    """Опоздание в пределах горизонта — не «опоздало», пара образуется задним числом."""
    buf = TimeWindowBuffer(epsilon_ms=600.0, lateness_ms=2000.0)
    buf.add(_msg("video", 10.0))
    buf.add(_msg("video", 12.0))
    win = buf.add(_msg("audio", 10.0))
    assert not win.late
    assert win.joint
    assert win.best_video().ts == pytest.approx(10.0)


def test_window_interval_stays_plus_minus_epsilon() -> None:
    """Горизонт опоздания расширяет УДЕРЖАНИЕ истории, но не интервал окна [t−ε, t+ε]."""
    buf = TimeWindowBuffer(epsilon_ms=600.0, lateness_ms=5000.0)
    buf.add(_msg("video", 10.0))
    win = buf.add(_msg("video", 10.5))
    assert win.t0 == pytest.approx(10.5 - 0.6)
    assert win.t1 == pytest.approx(10.5 + 0.6)
    assert [m.ts for m in win.video] == [10.0, 10.5]  # оба в интервале
    win2 = buf.add(_msg("video", 11.2))               # video@10 уже вне [t−ε, t+ε]
    assert [m.ts for m in win2.video] == [11.2]


def test_negative_lateness_clamped_to_zero() -> None:
    """lateness_ms < 0 → 0 (поведение эквивалентно прежней эвикции по t−ε)."""
    buf = TimeWindowBuffer(epsilon_ms=600.0, lateness_ms=-1.0)
    buf.add(_msg("video", 10.0))
    win = buf.add(_msg("audio", 10.0))
    assert win.joint                     # в пределах ε пара образуется и так
    win2 = buf.add(_msg("audio", 11.0))  # video@10 выселен сразу
    assert not win2.joint
