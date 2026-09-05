"""Тесты аудио-подтверждения в режиме airborne (it-55, политика P1 из it-54)."""

from __future__ import annotations

import pytest
from fusion.strategies.base import FusionOutcome
from fusion.target import apply_audio_confirmation
from uavdet_common.messages import InferenceMsg, QualityHint


def _outcome(decision: bool, p_v: float, p_a: float | None) -> FusionOutcome:
    return FusionOutcome(p_fused=0.54 if decision else 0.11, decision=decision,
                         p_v=p_v, p_a=p_a, w_v=0.5, w_a=0.5, delta=0.0)


def test_confirmed_audio_keeps_decision() -> None:
    """p_a ≥ floor: положительное решение подтверждено — не меняется."""
    out = apply_audio_confirmation(_outcome(True, 0.86, 0.5), floor=0.3)
    assert out.decision is True
    assert out.p_a == pytest.approx(0.5)


def test_unconfirmed_audio_flips_decision() -> None:
    """Сценарий it-51: p_v=0.86, сглаженный p_a=0.22 → p_fused=0.54, но p_a < 0.3 → отмена."""
    out = apply_audio_confirmation(_outcome(True, 0.86, 0.22), floor=0.3)
    assert out.decision is False
    assert out.p_fused == pytest.approx(0.54)   # p_fused сохранён для наблюдаемости


def test_missing_audio_cancels_positive() -> None:
    """Аудио отсутствовало (p_a=None) — подтвердить нечем: решение отменяется."""
    out = apply_audio_confirmation(_outcome(True, 0.86, None), floor=0.3)
    assert out.decision is False


def test_negative_decision_untouched() -> None:
    """Отрицательные решения не изменяются (подтверждение нужно только «дрон»)."""
    out = apply_audio_confirmation(_outcome(False, 0.1, None), floor=0.3)
    assert out.decision is False


def test_floor_zero_disables_confirmation() -> None:
    """floor=0: любое p_a ≥ 0 подтверждает — поведение эквивалентно отсутствию политики."""
    out = apply_audio_confirmation(_outcome(True, 0.86, 0.05), floor=0.0)
    assert out.decision is True


def _msg(modality: str, label: str, conf: float, media_ts: float, motion: float | None = None) -> InferenceMsg:
    q = QualityHint(motion_score=motion) if modality == "video" else QualityHint()
    return InferenceMsg(source_id="cam-01", modality=modality, label=label, confidence=conf,
                        ts=1000.0 + media_ts, media_ts=media_ts, quality=q,
                        msg_id=f"{modality}-{media_ts}-{label}-{conf}")


def test_consumer_airborne_applies_confirmation() -> None:
    """Проводка: consumer в airborne-режиме применяет аудио-подтверждение (политика P1)."""
    import json

    from fusion.consumer import InferenceConsumer
    from fusion.gating import GatingResult
    from fusion.strategies.late import LateFusion
    from fusion.window_buffer import TimeWindowBuffer

    class _FixedGating:
        def weights(self, window):  # noqa: ANN001
            return GatingResult(w_v=0.5, w_a=0.5)

    class _FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        def subscribe(self, topic, group_id):  # noqa: ANN001
            return iter(())

        def publish(self, topic, key, value):  # noqa: ANN001
            self.published.append((topic, value))

        def commit(self) -> None:
            pass

    def _mk(target: str, audio_conf: float) -> InferenceConsumer:
        c = InferenceConsumer(
            _FakeBus(), group_id="t", strategy=LateFusion(delta_conf=0.0, delta_unconf=0.0),
            gating=_FixedGating(), window_buffer=TimeWindowBuffer(epsilon_ms=600.0),
            decision_threshold=0.5, window_release="watermark", window_max_wait_ms=60_000.0,
            target=target, audio_confirm_floor=0.3,
        )
        # летящий дрон (motion=0.5 ≥ floor) + аудио-поток: 10 видео и 10 аудио сообщений
        for i in range(10):
            c.process(None, _msg("video", "drone", 0.86, 42.0 + i * 0.2, motion=0.5))
            c.process(None, _msg("audio", "drone", audio_conf, 42.0 + i * 0.2))
        return c

    def _decisions(c: InferenceConsumer) -> list[dict]:
        return [json.loads(v) for t, v in c._bus.published if t == "decisions"]

    confirmed = _mk("airborne", 0.6)
    dec = _decisions(confirmed)
    assert dec and all(d["decision"] for d in dec)          # аудио подтверждает → «активен»

    unconfirmed = _mk("airborne", 0.1)                      # p_a=0.1 < floor → решения отменяются
    dec2 = _decisions(unconfirmed)
    assert dec2 and all(not d["decision"] for d in dec2)    # аудио-подтверждение не пройдено
