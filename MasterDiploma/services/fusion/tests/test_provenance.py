"""Тесты provenance-поля решения (research/it-67).

Мотивация: 175 662 записи `data/decisions/decisions.jsonl` не содержат идентификатора модели,
при этом visual-detector при отсутствующем файле весов тихо откатывается на COCO-заглушку
(`detector.py`, `_FALLBACK_WEIGHTS`) — офлайн «наша модель» неотличима от заглушки.
"""

from __future__ import annotations

import time

from fusion.consumer import InferenceConsumer
from fusion.window_buffer import AlignedWindow
from uavdet_common.messages import InferenceMsg, ModelRef


def _msg(modality: str, name: str = "yolov8s-uav", ver: str = "1") -> InferenceMsg:
    ts = time.time()
    return InferenceMsg(
        source_id="cam-01",
        modality=modality,
        label="drone",
        confidence=0.8,
        ts=ts,
        msg_id=f"{modality}-{name}",
        model=ModelRef(name=name, ver=ver),
    )


def _window(*msgs: InferenceMsg) -> AlignedWindow:
    ts = time.time()
    win = AlignedWindow(source_id="cam-01", t0=ts - 0.6, t1=ts + 0.6)
    win.video = [m for m in msgs if m.modality == "video"]
    win.audio = [m for m in msgs if m.modality == "audio"]
    return win


def test_refs_from_both_channels() -> None:
    win = _window(_msg("video"), _msg("audio", name="lwcnn", ver="2"))
    refs = InferenceConsumer._model_refs(win)
    assert refs == {"video": ModelRef(name="yolov8s-uav", ver="1"), "audio": ModelRef(name="lwcnn", ver="2")}


def test_surrogate_fallback_is_visible() -> None:
    """Откат видео-ветки на заглушку обязан читаться из записи о решении."""
    win = _window(_msg("video", name="yolov8n", ver="surrogate-coco"))
    refs = InferenceConsumer._model_refs(win)
    assert refs["video"].ver == "surrogate-coco"


def test_unknown_model_not_recorded() -> None:
    """Старые сообщения без заполненного ModelRef не порят решение фиктивным значением."""
    win = _window(_msg("video", name="unknown"))
    assert InferenceConsumer._model_refs(win) == {}
