"""Тесты общего медиатаймлайна MediaFileAdapter (it-34, ревью GPT-6-Astra §6.1).

Ключевой инвариант: позиции `media_ts` видео и аудио описывают ОДИН таймлайн исходного
медиа; шаг медиа-времени кадров = 1/requested_fps (страйд по файлу), т.е. запрошенный FPS
не растягивает содержание относительно аудио.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf
from source_simulator.adapters.media_file import MediaFileAdapter
from source_simulator.controller import SimulatorController


@pytest.fixture()
def media(tmp_path: Path) -> tuple[str, str]:
    """Видео 25 FPS × 10 кадров (0.4 с) + wav 1.0 с 16 кГц."""
    video = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    assert vw.isOpened()
    for i in range(10):
        vw.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    vw.release()

    audio = tmp_path / "clip.wav"
    sr = 16000
    t = np.arange(sr) / sr
    sf.write(str(audio), (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    return str(video), str(audio)


def test_video_media_ts_advance_matches_requested_fps(media) -> None:
    """Файл 25 FPS при отдаче 5 FPS: каждый 5-й кадр, медиа-шаг 0.2 с (а не растяжение ×5)."""
    video, _ = media
    ad = MediaFileAdapter(source_id="cam-01", video_path=video, fps=5.0, loop=False)
    frames = list(ad.frames())
    assert [f.media_ts for f in frames] == pytest.approx([0.0, 0.2])
    assert [f.seq for f in frames] == [1, 2]


def test_video_media_ts_is_monotonic_across_loops(media) -> None:
    """loop=True: медиа-время продолжается за концом прохода (0.4 с), не сбрасывается."""
    video, _ = media
    ad = MediaFileAdapter(source_id="cam-01", video_path=video, fps=5.0, loop=True)
    gen = ad.frames()
    frames = [next(gen) for _ in range(6)]
    ad.close()
    mts = [f.media_ts for f in frames]
    assert mts == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert all(b > a for a, b in zip(mts, mts[1:], strict=False))


def test_audio_windows_carry_media_start(media) -> None:
    """Аудио-окна: media_ts = позиция НАЧАЛА окна на том же таймлайне; шаг = hop."""
    video, audio = media
    ad = MediaFileAdapter(source_id="cam-01", video_path=video, audio_path=audio,
                          fps=5.0, loop=False, audio_sample_rate=16000,
                          audio_win_ms=250, audio_hop_ms=100)
    wins = list(ad.audio_windows())
    starts = [w.media_ts for w in wins]
    # окно 0.25 с по файлу 1.0 с с шагом 0.1 с → старты 0.0..0.7
    assert starts == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert wins[0].len_ms == 250 and wins[0].hop_ms == 100


def test_controller_passes_media_ts_and_stamps_wall_clock(media) -> None:
    """Контроллер: ts = wall-clock отправки, media_ts проходит без изменений (ревью §6.1)."""
    video, _ = media
    ad = MediaFileAdapter(source_id="cam-01", video_path=video, fps=5.0, loop=False)
    ctrl = SimulatorController(ad, client=None, video_period_s=0.0)  # без ожиданий
    items = list(ctrl._paced_frames())
    assert [it.media_ts for it in items] == pytest.approx([0.0, 0.2])
    assert all(it.ts > 1_500_000_000 for it in items)  # unix wall-clock


def test_grpc_frame_carries_media_ts(media) -> None:
    from source_simulator.grpc_client import GrpcStreamClient

    video, _ = media
    ad = MediaFileAdapter(source_id="cam-01", video_path=video, fps=5.0, loop=False)
    frame = next(iter(ad.frames()))
    pb = GrpcStreamClient._to_frame_pb(frame, "cam-01")
    assert pb.media_ts == pytest.approx(frame.media_ts)
