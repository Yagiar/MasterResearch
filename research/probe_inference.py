#!/usr/bin/env python3
"""Проба топика inference (it-47): сливает сообщения в jsonl для джойна с decisions по msg_id.

Запуск ПАРАЛЛЕЛЬНО пайплайну (group свежий, latest — видит только текущий прогон):
  research/.venv/bin/python research/probe_inference.py <секунды> [выход.jsonl]
"""
import json
import sys
import time

from confluent_kafka import Consumer

ROOT = "/home/otrix/code/GeneralFolderMasterDiploma"
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/research/inference_dump.jsonl"
BOOTSTRAP = sys.argv[3] if len(sys.argv) > 3 else "localhost:9092"

c = Consumer({
    "bootstrap.servers": BOOTSTRAP,
    "group.id": f"probe-inference-{int(time.time())}",
    "auto.offset.reset": "latest",
})
c.subscribe(["inference"])
n = 0
t_end = time.time() + DURATION
with open(OUT, "w", encoding="utf-8") as fh:
    while time.time() < t_end:
        m = c.poll(0.2)
        if m is None or m.error():
            continue
        fh.write(m.value().decode("utf-8", errors="replace").rstrip() + "\n")
        n += 1
c.close()
print(f"probe: {n} inference-сообщений → {OUT}")
