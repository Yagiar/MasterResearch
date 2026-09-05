"""Тесты схем сообщений: round-trip JSON, дефолты, карта топиков."""

from uavdet_common.messages import (
    SCHEMA_VER,
    TOPIC_MODELS,
    AudioRawMsg,
    DecisionMsg,
    InferenceMsg,
    Topics,
    VideoRawMsg,
)
from uavdet_common.serialization import JsonSerializer


def test_video_raw_defaults():
    m = VideoRawMsg(source_id="cam-01", seq=42)
    assert m.schema_ver == SCHEMA_VER
    assert m.msg_id and len(m.msg_id) == 32
    assert m.ts > 0
    assert m.payload_kind == "jpeg"


def test_round_trip_inference():
    ser = JsonSerializer()
    src = InferenceMsg(
        source_id="cam-01",
        modality="video",
        label="drone",
        confidence=0.87,
        bbox=[10.0, 20.0, 30.0, 40.0],
        track_id=7,
    )
    raw = ser.encode(src)
    back = ser.decode(raw, InferenceMsg)
    assert isinstance(back, InferenceMsg)
    assert back.msg_id == src.msg_id
    assert back.modality == "video" and back.label == "drone"
    assert back.bbox == [10.0, 20.0, 30.0, 40.0]
    assert back.track_id == 7


def test_round_trip_decision():
    ser = JsonSerializer()
    src = DecisionMsg(
        source_id="cam-01",
        ts_window=[1.0, 2.0],
        mode="late",
        decision=True,
        p_fused=0.79,
    )
    src.contributions.p_v = 0.87
    src.contributions.w_v = 0.55
    raw = ser.encode(src)
    back = ser.decode(raw, DecisionMsg)
    assert back.mode == "late" and back.decision is True
    assert back.contributions.p_v == 0.87 and back.contributions.w_v == 0.55


def test_topic_models_map():
    assert TOPIC_MODELS[Topics.VIDEO_RAW] is VideoRawMsg
    assert TOPIC_MODELS[Topics.INFERENCE] is InferenceMsg
    assert TOPIC_MODELS[Topics.DECISIONS] is DecisionMsg


def test_media_ts_contract_additive():
    """it-34/35: media_ts аддитивен — старые сообщения (без поля) парсятся (None), новые сериализуются."""
    old = VideoRawMsg(source_id="cam-01", ts=1.0)          # без media_ts (старый продюсер)
    assert old.media_ts is None
    v = VideoRawMsg(source_id="cam-01", ts=1.0, media_ts=12.5)
    assert VideoRawMsg.model_validate_json(v.model_dump_json()).media_ts == 12.5
    a = AudioRawMsg(source_id="cam-01", ts=1.0, media_ts=12.5)
    assert AudioRawMsg.model_validate_json(a.model_dump_json()).media_ts == 12.5
    inf = InferenceMsg(source_id="cam-01", ts=1.0, modality="video", label="drone",
                       confidence=0.9, media_ts=12.5)
    assert InferenceMsg.model_validate_json(inf.model_dump_json()).media_ts == 12.5
    d = DecisionMsg(source_id="cam-01", ts=1.0, ts_window=[12.0, 13.0], media_ts=12.5,
                    mode="late", decision=True, p_fused=0.9)
    assert DecisionMsg.model_validate_json(d.model_dump_json()).media_ts == 12.5
