"""Тесты watermark-режима выпуска окон (it-44, ревью GPT-6-Astra §6.2).

Инварианты:
  - при watermark-релизе решение для триггера t не выходит, пока медиа-водяной знак
    (min последних ts модальностей) не прошёл t+ε → бурст одной модальности не даёт
    mono-решений там, где вторая «на подходе»;
  - если вторая модальность не пришла за window_max_wait — ЯВНЫЙ mono-fallback
    (решение выпускается, помечается late, счётчик reason=max_wait);
  - per-message (дефолт) — прежнее поведение: решение на каждое сообщение.
"""

from __future__ import annotations

import time

import pytest
from fusion.consumer import InferenceConsumer
from fusion.strategies.late import LateFusion
from fusion.window_buffer import TimeWindowBuffer
from uavdet_common.messages import InferenceMsg


class _FakeBus:
    """Минимальная шина: publish складывает закодированные сообщения в список."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    def subscribe(self, topic, group_id):  # noqa: ANN001
        return iter(())

    def publish(self, topic: str, key, value: bytes) -> None:  # noqa: ANN001
        self.published.append((topic, value))

    def commit(self) -> None:  # pragma: no cover
        pass


def _consumer(release: str, max_wait_ms: float = 200.0) -> InferenceConsumer:
    return InferenceConsumer(
        _FakeBus(),
        group_id="test",
        strategy=LateFusion(),
        gating=None.__class__ if False else _FixedGating(),
        window_buffer=TimeWindowBuffer(epsilon_ms=600.0),
        decision_threshold=0.5,
        window_release=release,
        window_max_wait_ms=max_wait_ms,
    )


class _FixedGating:
    """Заглушка gating: фиксированные веса 0.5/0.5 (метод weights есть у реальных политик)."""

    def weights(self, window):  # noqa: ANN001
        from fusion.gating import GatingResult

        return GatingResult(w_v=0.5, w_a=0.5)


def _video(media_ts: float, conf: float = 0.8) -> InferenceMsg:
    return InferenceMsg(source_id="cam-01", modality="video", label="drone", confidence=conf,
                        ts=1000.0 + media_ts, media_ts=media_ts, msg_id=f"v@{media_ts}")


def _audio(media_ts: float, conf: float = 0.9, label: str = "drone") -> InferenceMsg:
    return InferenceMsg(source_id="cam-01", modality="audio", label=label, confidence=conf,
                        ts=1000.0 + media_ts, media_ts=media_ts, msg_id=f"a@{media_ts}")


def _decisions(c: InferenceConsumer) -> list[dict]:
    import json

    out = []
    for topic, value in c._bus.published:  # type: ignore[attr-defined]
        assert topic == "decisions"
        out.append(json.loads(value))
    return out


def test_per_message_mode_emits_immediately() -> None:
    """Дефолт: решение на каждое сообщение (прежнее поведение, регресс)."""
    c = _consumer("per-message")
    c.process(None, _video(10.0))
    c.process(None, _audio(10.05))
    assert len(_decisions(c)) == 2


def test_watermark_burst_does_not_release_mono_prematurely() -> None:
    """Бурст видео (сценарий it-42): решения не выпускаются mono, пока водяной знак не прошёл t+ε."""
    c = _consumer("watermark", max_wait_ms=60_000.0)
    for i in range(10):                       # видео-бурст: media 0..1.8
        c.process(None, _video(i * 0.2))
    assert _decisions(c) == []                # аудио нет → водяной знак 0 → ничего не выпущено
    c.process(None, _audio(0.0))              # аудио подтянулось
    c.process(None, _audio(0.5))
    c.process(None, _audio(1.0))
    c.process(None, _audio(1.5))
    c.process(None, _audio(2.0))              # водяной знак = min(video 1.8, audio 2.0) = 1.8
    dec = _decisions(c)
    assert len(dec) >= 6                      # окна t+ε ≤ 1.8 выпущены
    assert all(d["contributions"]["p_a"] is not None for d in dec)   # все СОВМЕСТНЫЕ
    assert all(d["contributions"]["p_v"] is not None for d in dec)


def test_watermark_max_wait_releases_explicit_mono_fallback() -> None:
    """Аудио не приходит вовсе: после max_wait решение выпускается как явный mono-fallback."""
    c = _consumer("watermark", max_wait_ms=1.0)   # ожидание почти нулевое
    c.process(None, _video(10.0))
    time.sleep(0.02)
    c.process(None, _video(10.4))             # следующий видео-триггер будит релиз-цикл
    dec = _decisions(c)
    assert len(dec) >= 1
    assert all(d["contributions"]["p_a"] is None for d in dec)   # mono — честно без p_a


def test_watermark_releases_joint_after_audio_lag() -> None:
    """Сценарий it-42: видео рвётся вперёд на 3 с медиа-времени; watermark+max_wait → joint."""
    c = _consumer("watermark", max_wait_ms=60_000.0)
    for i in range(20):                       # видео до media 3.8
        c.process(None, _video(i * 0.2))
    assert _decisions(c) == []
    # аудио приезжает с лагом (реальный порядок доставки), media_ts корректные
    for i in range(0, 15):
        c.process(None, _audio(i * 0.2))
    dec = _decisions(c)
    joint = [d for d in dec if d["contributions"]["p_a"] is not None]
    assert joint, "после подтяжки аудио должны появиться совместные окна"
    assert all(d["media_ts"] is not None for d in joint)


def test_unknown_release_mode_rejected() -> None:
    with pytest.raises(ValueError):
        _consumer("banana")
