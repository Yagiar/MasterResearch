"""Тесты InMemoryBus и KafkaConsumerService (на InMemoryBus, без реального брокера)."""

import threading
import time

from uavdet_common.bus import InMemoryBus
from uavdet_common.consumer_service import KafkaConsumerService
from uavdet_common.messages import DecisionMsg, Topics, VideoRawMsg
from uavdet_common.serialization import JsonSerializer


def test_inmemory_bus_publish_drain():
    bus = InMemoryBus()
    ser = JsonSerializer()
    m = VideoRawMsg(source_id="cam-01", seq=1)
    bus.publish(Topics.VIDEO_RAW, "cam-01", ser.encode(m))
    items = bus.drain(Topics.VIDEO_RAW)
    assert len(items) == 1
    key, raw = items[0]
    assert key == "cam-01"
    back = ser.decode(raw, VideoRawMsg)
    assert back.msg_id == m.msg_id


class _EchoConsumer(KafkaConsumerService):
    in_topic = Topics.VIDEO_RAW
    group_id = "test-echo"
    in_model = VideoRawMsg

    def __init__(self, bus):
        super().__init__(bus)
        self.processed = []

    def process(self, key, msg):
        self.processed.append((key, msg.msg_id))
        # «эхо» в decisions
        out = DecisionMsg(
            source_id=msg.source_id, ts_window=[msg.ts, msg.ts], mode="video-only",
            decision=True, p_fused=1.0,
        )
        self.publish(Topics.DECISIONS, msg.source_id, out)


def test_consumer_processes_and_dedups():
    bus = InMemoryBus()
    ser = JsonSerializer()
    svc = _EchoConsumer(bus)

    m1 = VideoRawMsg(source_id="cam-01", seq=1)
    m2 = VideoRawMsg(source_id="cam-01", seq=2)
    bus.publish(Topics.VIDEO_RAW, "cam-01", ser.encode(m1))
    bus.publish(Topics.VIDEO_RAW, "cam-01", ser.encode(m1))  # дубликат — должен пропуститься
    bus.publish(Topics.VIDEO_RAW, "cam-01", ser.encode(m2))

    t = threading.Thread(target=svc.run, daemon=True)
    t.start()
    # дать обработать 3 сообщения, затем остановить
    deadline = time.time() + 3.0
    while len(svc.processed) < 2 and time.time() < deadline:
        time.sleep(0.05)
    svc._stop = True
    # «протолкнуть» цикл (subscribe блокируется на q.get) — публикуем sentinel и ждём
    bus.publish(Topics.VIDEO_RAW, "cam-01", ser.encode(VideoRawMsg(source_id="cam-01", seq=99)))
    time.sleep(0.2)

    ids = [mid for _, mid in svc.processed]
    assert m1.msg_id in ids and m2.msg_id in ids
    # дубликат m1 не должен появиться вторым
    assert ids.count(m1.msg_id) == 1
    decisions = bus.drain(Topics.DECISIONS)
    assert len(decisions) >= 2
