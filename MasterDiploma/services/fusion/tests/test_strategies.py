"""Тесты стратегий слияния: маска допущенных каналов (it-32, ревью GPT-6-Astra §6.4).

Канал допущен ⟺ есть детекция в окне И вес после gating/health-gate > 0. Закрытый
канал не влияет ни на сумму, ни на Δ-правила, ни на вероятностное ИЛИ (hybrid);
если не допущен ни один — решение не формируется.
"""

from __future__ import annotations

import pytest
from fusion.strategies.hybrid import HybridFusion
from fusion.strategies.late import LateFusion
from fusion.strategies.video_only import VideoOnly
from fusion.window_buffer import AlignedWindow
from uavdet_common.messages import InferenceMsg


def _msg(modality: str, label: str, conf: float, ts: float = 1000.0) -> InferenceMsg:
    return InferenceMsg(source_id="cam-01", modality=modality, label=label, confidence=conf,
                        ts=ts, msg_id=f"{modality}-{label}-{conf}")


def _window(pv: tuple[str, float] | None, pa: tuple[str, float] | None) -> AlignedWindow:
    win = AlignedWindow(source_id="cam-01", t0=999.5, t1=1000.5)
    if pv is not None:
        win.video = [_msg("video", *pv)]
    if pa is not None:
        win.audio = [_msg("audio", *pa, ts=1000.1)]
    return win


# --- LateFusion ---

def test_late_closed_audio_no_contradiction_penalty() -> None:
    """Контрпример рецензента: p_v=0.55, аудио не-дрон, w_a=0 (гейт закрыл).
    Было: 0.55 + (−0.1) = 0.45 → отрицательное решение. Стало: решение как video-only."""
    out = LateFusion().fuse(_window(("drone", 0.55), ("non-drone", 0.9)),
                            w_v=0.5, w_a=0.0, threshold=0.5)
    assert out is not None
    assert out.p_fused == pytest.approx(0.55)
    assert out.decision is True
    assert out.delta == 0.0
    assert out.p_a is None            # закрытый канал не участвовал
    assert out.w_a == pytest.approx(0.0)


def test_late_closed_audio_no_confidence_bonus() -> None:
    """Закрытое аудио не даёт и бонус подтверждения: p_fused = p_v ровно."""
    out = LateFusion().fuse(_window(("drone", 0.55), ("drone", 0.9)),
                            w_v=0.5, w_a=0.0, threshold=0.5)
    assert out.p_fused == pytest.approx(0.55)
    assert out.delta == 0.0


def test_late_closed_audio_decision_independent_of_its_value() -> None:
    """Инвариант: при закрытом канале значение аудио не меняет решение (любое p_a)."""
    f = LateFusion()
    outs = [f.fuse(_window(("drone", 0.55), ("drone" if lbl else "non-drone", c)),
                   w_v=0.5, w_a=0.0, threshold=0.5)
            for lbl, c in ((1, 0.99), (1, 0.01), (0, 0.99), (0, 0.01))]
    assert {round(o.p_fused, 12) for o in outs} == {0.55}
    assert {o.decision for o in outs} == {True}


def test_late_open_audio_contradiction_penalty_still_applies() -> None:
    """Контраст: канал допущен (w_a=0.5) — штраф противоречия работает как раньше."""
    out = LateFusion().fuse(_window(("drone", 0.55), ("non-drone", 0.9)),
                            w_v=0.5, w_a=0.5, threshold=0.5)
    assert out.p_fused == pytest.approx(0.5 * 0.55 + 0.5 * 0.0 - 0.1)
    assert out.decision is False
    assert out.delta == pytest.approx(-0.1)
    assert out.p_a == pytest.approx(0.0)


def test_late_missing_audio_renormalizes_to_video() -> None:
    """Аудио отсутствует (не закрыто, а молчит) — прежняя семантика перенормировки."""
    out = LateFusion().fuse(_window(("drone", 0.55), None), w_v=0.5, w_a=0.5, threshold=0.5)
    assert out.p_fused == pytest.approx(0.55)
    assert out.p_a is None and out.w_v == pytest.approx(1.0)


def test_late_all_channels_closed_returns_none() -> None:
    """Оба веса нулевые → «недостаточно данных», решение не формируется."""
    assert LateFusion().fuse(_window(("drone", 0.9), ("drone", 0.9)),
                             w_v=0.0, w_a=0.0, threshold=0.5) is None


# --- HybridFusion ---

def test_hybrid_closed_audio_no_penalty_no_or() -> None:
    """Контрпример рецензента: p_v=0.9, w_a=0, аудио не-дрон.
    Было: 0.9·0.5 + 0 + (−0.15) = 0.30 → False. Стало: 0.9 → True (как video-only)."""
    out = HybridFusion().fuse(_window(("drone", 0.9), ("non-drone", 0.9)),
                              w_v=0.5, w_a=0.0, threshold=0.5)
    assert out is not None
    assert out.p_fused == pytest.approx(0.9)
    assert out.decision is True
    assert out.p_a is None


def test_hybrid_closed_audio_excluded_from_prob_or() -> None:
    """Закрытое аудио не участвует в вероятностном ИЛИ: p_fused = p_v без буста."""
    out = HybridFusion().fuse(_window(("drone", 0.9), ("drone", 0.9)),
                              w_v=0.5, w_a=0.0, threshold=0.5)
    assert out.p_fused == pytest.approx(0.9)
    assert out.delta == 0.0


def test_hybrid_open_audio_old_behaviour_preserved() -> None:
    """Оба канала допущены — прежняя арифметика: штраф противоречия, ИЛИ при согласии."""
    out = HybridFusion().fuse(_window(("drone", 0.9), ("non-drone", 0.9)),
                              w_v=0.5, w_a=0.5, threshold=0.5)
    assert out.p_fused == pytest.approx(0.5 * 0.9 - 0.15)   # 0.30
    assert out.decision is False

    out2 = HybridFusion().fuse(_window(("drone", 0.8), ("drone", 0.8)),
                               w_v=0.5, w_a=0.5, threshold=0.5)
    prob_or = 1.0 - 0.2 * 0.2
    assert out2.p_fused == pytest.approx(min(1.0, max(0.8, prob_or) + 0.1))  # 0.96+0.1 → clip 1.0


def test_hybrid_weights_renormalized_over_allowed() -> None:
    """Веса в двухканальной ветке перенормируются (было: сырые w_v/w_a без нормировки)."""
    out = HybridFusion().fuse(_window(("drone", 0.8), ("drone", 0.4)),
                              w_v=0.6, w_a=0.6, threshold=0.5)
    # p_a=0.4 < label_threshold → «не дрон» для Δ-правила: противоречие, base − δ_unconf
    assert out.p_fused == pytest.approx(0.5 * 0.8 + 0.5 * 0.4 - 0.15)


def test_hybrid_all_channels_closed_returns_none() -> None:
    assert HybridFusion().fuse(_window(("drone", 0.9), ("drone", 0.9)),
                               w_v=0.0, w_a=0.0, threshold=0.5) is None


# --- VideoOnly: baseline не зависит от весов/аудио (регресс) ---

def test_video_only_ignores_audio_and_weights() -> None:
    out = VideoOnly().fuse(_window(("drone", 0.55), ("drone", 0.99)),
                           w_v=0.0, w_a=1.0, threshold=0.5)
    assert out is not None
    assert out.p_fused == pytest.approx(0.55)
    assert out.p_a is None
