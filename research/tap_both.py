import json, time
from confluent_kafka import Consumer
c = Consumer({"bootstrap.servers":"kafka:9092","group.id":f"tapboth-{time.time()}","auto.offset.reset":"latest"})
c.subscribe(["video.raw", "inference"])
out = open("/probe/motion_tap.jsonl", "w")
n = 0
t_end = time.time() + 420
while time.time() < t_end:
    m = c.poll(0.3)
    if m is None or m.error(): continue
    d = json.loads(m.value())
    rec = {"topic": m.topic(), "msg_id": d.get("msg_id"), "seq": d.get("seq"),
           "motion": (d.get("quality") or {}).get("motion_score") if m.topic() == "inference" else None,
           "media_ts": d.get("media_ts") if m.topic() == "inference" else None,
           "label": d.get("label") if m.topic() == "inference" else None}
    out.write(json.dumps(rec) + "\n")
    n += 1
    if n % 500 == 0:
        out.flush()
out.close(); c.close()
print(f"tapped {n}")
