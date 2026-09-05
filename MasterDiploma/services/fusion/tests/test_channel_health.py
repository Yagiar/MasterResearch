"""Тесты гейта здоровья аудиоканала (research/it-16, it-19): «тишина» ≠ «глухота»."""

from __future__ import annotations

import pytest
from fusion.temporal import ChannelHealthGate

RMS_SILENCE = 0.005   # стоянка/тишина (research/it-16: медиана 0.0052)
RMS_LOUD = 0.07       # полёт (медиана 0.0698)


def test_w_lt_2_rejected() -> None:
    with pytest.raises(ValueError):
        ChannelHealthGate(w=1)


def test_healthy_clip_gate_never_closes() -> None:
    """Реалистичный клип: стоянка (тихо, pa=0) → полёт (громко, pa~0.7) → стоянка.

    Гейт не должен закрываться: нули на тишине — норма (RMS низкая), на полёте выход
    изменчив (std высокий).
    """
    g = ChannelHealthGate(w=12)
    seq = ([(RMS_SILENCE, 0.0)] * 20 + [(RMS_LOUD, 0.75)] * 100 + [(RMS_SILENCE, 0.0)] * 24)
    scales = [g.scale("cam", r, p) for r, p in seq]
    assert all(s == 1.0 for s in scales), scales


def test_full_deaf_loud_closes_gate() -> None:
    """Глухота: звук громкий всё время, выход детерминированно 0 → гейт закрывается."""
    g = ChannelHealthGate(w=12)
    scales = [g.scale("cam", RMS_LOUD, 0.0) for _ in range(30)]
    assert all(s == 1.0 for s in scales[:6])       # подозрения копятся (needed=6)
    assert scales.index(0.0) <= 7                   # закрыт не позже 7-го окна
    assert scales[-1] == 0.0


def test_deaf_then_recovery_reopens() -> None:
    """Глухой участок → гейт закрылся; оживление (pa 0.7 при громком звуке) → открыт."""
    g = ChannelHealthGate(w=12)
    for _ in range(25):
        g.scale("cam", RMS_LOUD, 0.0)
    assert g.scale("cam", RMS_LOUD, 0.0) == 0.0
    # оживление: смесь нулей и уверенных drone → std за окном 12 вырастает >= 0.15
    for conf in (0.0, 0.75, 0.0, 0.8, 0.0, 0.7, 0.0, 0.8, 0.0, 0.75, 0.0, 0.8):
        last = g.scale("cam", RMS_LOUD, conf)
    assert last == 1.0


def test_silence_only_stream_stays_open() -> None:
    """Полностью тихий поток: RMS ниже абсолютного пола «звук есть» — подозрений нет,
    гейт остаётся открытым (закрытие на тишине было дефектом it-14/v1)."""
    g = ChannelHealthGate(w=12)
    scales = [g.scale("cam", RMS_SILENCE, 0.0) for _ in range(40)]
    assert all(s == 1.0 for s in scales)


def test_missing_audio_window_freezes_state() -> None:
    """Окно без аудио (rms/pa = None) не меняет состояние гейта."""
    g = ChannelHealthGate(w=12)
    for _ in range(20):
        g.scale("cam", RMS_LOUD, 0.0)
    assert g.scale("cam", RMS_LOUD, 0.0) == 0.0
    frozen = [g.scale("cam", None, None) for _ in range(10)]
    assert all(s == 0.0 for s in frozen)            # состояние не сбросилось
    # и не открылось бы из-за пропусков: громкий ноль продолжает закрывать
    assert g.scale("cam", RMS_LOUD, 0.0) == 0.0


def test_separates_sources() -> None:
    g = ChannelHealthGate(w=12)
    for _ in range(25):
        g.scale("cam-01", RMS_LOUD, 0.0)            # cam-01 глухой
    g.scale("cam-02", RMS_LOUD, 0.7)                # cam-02 здоровый
    assert g.scale("cam-01", RMS_LOUD, 0.0) == 0.0
    assert g.scale("cam-02", RMS_LOUD, 0.7) == 1.0


def test_repeat_msg_id_does_not_count_as_new_observation() -> None:
    """Инвариант (it-31, ревью §6.3): одно аудио-сообщение, протащенное через несколько
    видео-триггеров, заполняет окно гейта один раз — подозрения копятся только от новых msg_id."""
    g = ChannelHealthGate(w=12, suspect_needed=6)
    # 5 новых наблюдений «громко и глухо» + 30 повторов одного msg_id
    for i in range(5):
        assert g.scale("cam", RMS_LOUD, 0.0, msg_id=f"a-{i}") == 1.0
    repeats = [g.scale("cam", RMS_LOUD, 0.0, msg_id="a-4") for _ in range(30)]
    assert all(s == 1.0 for s in repeats)           # гейт НЕ закрылся: реальных наблюдений < suspect_needed
    assert g.scale("cam", RMS_LOUD, 0.0, msg_id="a-5") == 1.0    # 6-е новое → подозрений 6, ещё открыт
    assert g.scale("cam", RMS_LOUD, 0.0, msg_id="a-6") == 0.0    # 7-е новое → закрыт
