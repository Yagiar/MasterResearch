"""Тесты MmaudReplayAdapter: media_ts (it-60) и работа без аудио."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf
from source_simulator.adapters.mmaud_replay import MmaudReplayAdapter


@pytest.fixture()
def seq(tmp_path: Path) -> tuple[Path, Path]:
    """Папка из 6 PNG-кадров + wav 1 с."""

    img_dir = tmp_path / "image"
    img_dir.mkdir()
    for i in range(6):
        cv2.imwrite(str(img_dir / f"1692846887.{830421 + i * 300:06d}.png"),
                    np.full((48, 64, 3), i * 40, dtype=np.uint8))
    wav = tmp_path / "audio.wav"
    sr = 16000
    sf.write(str(wav), (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32), sr)
    return img_dir, wav


def test_folder_frames_have_monotonic_media_ts(seq) -> None:
    """media_ts = позиция кадра / fps_nominal, монотонно (it-60)."""
    img_dir, _ = seq
    ad = MmaudReplayAdapter(source_id="cam-01", video_path=str(img_dir), audio_path=None,
                            fps=30.0, loop=False)
    frames = list(ad.frames())
    assert len(frames) == 6
    mts = [f.media_ts for f in frames]
    assert mts == pytest.approx([i / 30.0 for i in range(6)])
    assert all(b > a for a, b in zip(mts, mts[1:], strict=False))


def test_folder_frames_media_ts_continues_across_loops(seq) -> None:
    """loop=True: медиа-время накапливается, не сбрасывается (как в media_file, it-34)."""
    img_dir, _ = seq
    ad = MmaudReplayAdapter(source_id="cam-01", video_path=str(img_dir), audio_path=None,
                            fps=30.0, loop=True)
    gen = ad.frames()
    first_pass = [next(gen).media_ts for _ in range(6)]
    second_start = next(gen).media_ts
    ad.close()
    assert first_pass == pytest.approx([i / 30.0 for i in range(6)])
    assert second_start == pytest.approx(6 / 30.0, abs=1e-6)  # продолжение, не сброс


def test_folder_media_ts_uses_requested_fps_timebase(seq) -> None:
    """it-83: у папки кадров таймбейс = темп отправки (requested_fps), а не номинальные 30 Hz.

    Находка it-82: при fps=2 media_ts рос в 15 раз медленнее wall-clock (240 с → 18,3 с),
    из-за чего media-слоты fusion агрегировали ~15 разных кадров.
    """
    img_dir, _ = seq
    ad = MmaudReplayAdapter(source_id="cam-01", video_path=str(img_dir), audio_path=None,
                            fps=2.0, loop=False)
    mts = [f.media_ts for f in ad.frames()]
    assert mts == pytest.approx([i * 0.5 for i in range(6)])
    assert all(b > a for a, b in zip(mts, mts[1:], strict=False))


def test_folder_media_ts_requested_fps_continues_across_loops(seq) -> None:
    """it-83: при fps=2 продолжение второй петли — 6 кадров / 2 к/с = 3,0 с."""
    img_dir, _ = seq
    ad = MmaudReplayAdapter(source_id="cam-01", video_path=str(img_dir), audio_path=None,
                            fps=2.0, loop=True)
    gen = ad.frames()
    for _ in range(6):
        next(gen)
    assert next(gen).media_ts == pytest.approx(3.0, abs=1e-6)
    ad.close()


def test_folder_media_ts_below_1fps(seq) -> None:
    """it-85: при fps<1 шаг media_ts = 1/fps (кламп max(1.0, ·) сжимал медиа-часы при 0,5 к/с)."""
    img_dir, _ = seq
    ad = MmaudReplayAdapter(source_id="cam-01", video_path=str(img_dir), audio_path=None,
                            fps=0.5, loop=False)
    mts = [f.media_ts for f in ad.frames()]
    assert mts == pytest.approx([i * 2.0 for i in range(6)])


def test_no_audio_yields_no_windows(seq) -> None:
    """audio_path=None → audio_windows() пуст (video-only режим, it-57)."""
    img_dir, _ = seq
    ad = MmaudReplayAdapter(source_id="cam-01", video_path=str(img_dir), audio_path=None,
                            fps=30.0, loop=False)
    assert ad.has_audio is False
    assert list(ad.audio_windows()) == []


def test_with_audio_windows_carry_media_start(seq) -> None:
    """С аудио: окна несут media_ts начала (тот же таймлайн, it-34/60)."""
    img_dir, wav = seq
    ad = MmaudReplayAdapter(source_id="cam-01", video_path=str(img_dir), audio_path=str(wav),
                            fps=30.0, loop=False, audio_sample_rate=16000,
                            audio_win_ms=250, audio_hop_ms=100)
    wins = list(ad.audio_windows())
    assert wins and all(w.media_ts is not None for w in wins)
    starts = [w.media_ts for w in wins]
    assert all(b > a for a, b in zip(starts, starts[1:], strict=False))
