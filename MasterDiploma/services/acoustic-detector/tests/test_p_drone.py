"""Тесты p_drone в AudioDetection (research/it-15): вероятность класса вместо бимодальной пары."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from acoustic_detector.cnn import AudioDetection

AST_DIR = Path(__file__).resolve().parents[3] / "models" / "acoustic" / "samid-drone-detector"


def test_audio_detection_p_drone_default_none() -> None:
    d = AudioDetection(label="non-drone", confidence=0.9, backend="cnn")
    assert d.p_drone is None  # lwcnn пока не отдаёт; поле опционально — контракт не ломается


def test_audio_detection_frozen_with_p_drone() -> None:
    import dataclasses

    d = AudioDetection(label="drone", confidence=0.7, backend="ast", p_drone=0.7)
    assert d.p_drone == pytest.approx(0.7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.p_drone = 0.5  # type: ignore[misc]


@pytest.mark.skipif(not (AST_DIR / "config.json").exists(), reason="нет локальных весов AST")
def test_ast_detect_exposes_p_drone_both_sides() -> None:
    """Интеграция: на drone- и non-drone-окнах p_drone присутствует в обоих случаях.

    Смысл: для non-drone окон раньше p(drone) терялась (писали 0.0), из-за чего
    калибровка порога была невозможна (research/it-12 — бимодальность). Теперь
    p_drone несёт «насколько не дрон».
    """
    import librosa
    import numpy as np
    import soundfile as sf
    from acoustic_detector.ast_classifier import AstAudioClassifier

    det = AstAudioClassifier(weights_path=str(AST_DIR), device="cpu", target_sample_rate=16000)
    assert det.backend == "ast"

    pcm, sr = sf.read(
        AST_DIR.parents[2] / "sandboxDataForSimulator" / "sandbox-audio-for-simulator.wav",
        dtype="int16",
    )
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    pcm16k = (librosa.resample(pcm.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000) * 32767).astype("<i2")

    win = 16000
    results = []
    for t0 in (12.0, 70.0):  # 12 c — полёт (ожидаем drone), 70 c — стоянка (ожидаем non-drone)
        chunk = pcm16k[int(t0 * 16000): int(t0 * 16000) + win]
        results.append(det.detect(base64.b64encode(chunk.tobytes()).decode(), src_sample_rate=16000, channels=1))

    for d in results:
        assert d.p_drone is not None, "AST обязан отдавать p_drone"
        assert 0.0 <= d.p_drone <= 1.0
        # согласованность пары (label, confidence) с p_drone
        if d.label == "drone":
            assert d.confidence == pytest.approx(d.p_drone)
        else:
            assert d.confidence == pytest.approx(1.0 - d.p_drone)
            assert d.p_drone < 0.5
    # полёт vs стоянка: p_drone в полёте должна быть заметно выше
    assert results[0].p_drone > results[1].p_drone


def test_inference_msg_p_drone_additive_contract() -> None:
    """it-18: поле p_drone в InferenceMsg аддитивно — старые сообщения без него парсятся,
    новые сериализуют значение; schema_ver не менялся."""
    from uavdet_common.messages import InferenceMsg

    old = InferenceMsg.model_validate({
        "source_id": "cam-01", "modality": "audio", "label": "non-drone",
        "confidence": 0.97, "ts": 1000.0,
    })
    assert old.p_drone is None                       # старое сообщение — поле None
    new = InferenceMsg.model_validate({
        "source_id": "cam-01", "modality": "audio", "label": "drone",
        "confidence": 0.71, "p_drone": 0.71, "ts": 1000.0,
    })
    assert new.p_drone == pytest.approx(0.71)
    data = new.model_dump()
    assert data["p_drone"] == pytest.approx(0.71)    # сериализация сохраняет значение
    # старый потребитель (без поля в схеме) бы проигнорировал ключ — pydantic extra=ignore по умолчанию
