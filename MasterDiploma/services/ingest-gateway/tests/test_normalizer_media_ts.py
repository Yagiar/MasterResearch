"""Тесты нормализатора: проброс media_ts в канонические сообщения (it-34, ревью §6.1)."""

from __future__ import annotations

import pytest
from ingest_gateway.normalizer import audio_to_audio_raw, frame_to_video_raw
from uavdet_proto import ingest_pb2


def test_frame_media_ts_propagated() -> None:
    f = ingest_pb2.Frame(jpeg_bytes=b"x", ts=1000.0, seq=1, source_id="cam-01",
                         width=64, height=48, fps_nominal=25.0, media_ts=12.4)
    msg = frame_to_video_raw(f)
    assert msg.media_ts == pytest.approx(12.4)
    assert msg.ts == pytest.approx(1000.0)  # wall-clock сохраняется отдельно


def test_frame_media_ts_zero_means_absent() -> None:
    f = ingest_pb2.Frame(jpeg_bytes=b"x", ts=1000.0, seq=1, source_id="cam-01")
    assert frame_to_video_raw(f).media_ts is None


def test_audio_window_media_ts_propagated() -> None:
    w = ingest_pb2.AudioWindow(pcm_bytes=b"p", ts=1000.0, seq=1, source_id="cam-01",
                               sample_rate=16000, channels=1, len_ms=1000, hop_ms=500,
                               media_ts=30.5)
    msg = audio_to_audio_raw(w)
    assert msg.media_ts == pytest.approx(30.5)
    assert msg.window.len_ms == 1000


def test_audio_window_media_ts_zero_means_absent() -> None:
    w = ingest_pb2.AudioWindow(pcm_bytes=b"p", ts=1000.0, seq=1, source_id="cam-01")
    assert audio_to_audio_raw(w).media_ts is None
