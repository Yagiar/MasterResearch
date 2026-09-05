#!/usr/bin/env python3
"""Разложение e2e-задержки по стадиям конвейера (it-47; ревью §12).

Стадии (по меткам времени сообщений; все величины в мс):
  raw_queue     = detect_start − ingest_ts            — Kafka video.raw/audio.raw + ожидание детектором
  detect        = detect_done  − detect_start         — обработка детектором (декодирование + инференс)
  fusion_wait   = decision.ts  − detect_done          — очередь inference + накопление окна/watermark (включая ожидание второй модальности)
  e2e           = decision.ts  − ingest_ts            — сумма (контроль: raw_queue+detect+fusion_wait ≈ e2e)
Триггер решения — то из source_msg_ids, чьё media_ts ближе к media_ts решения; его метки дают стадию
(для joint-окон вторая модальность вносит свою задержку в fusion_wait — см. медианы по модальностям).

Публикуются медиана и перцентили p90/p99 (требование ревью: не только среднее).
Запуск: research/.venv/bin/python research/e2e_decompose.py <decisions.jsonl> <inference_dump.jsonl>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEC_PATH = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/MasterDiploma/data/decisions/decisions.jsonl"
INF_PATH = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/research/inference_dump.jsonl"

def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]

def stats(vals):
    v = sorted(x for x in vals if x is not None and x == x)
    if not v:
        return "—"
    med, p90, p99 = pct(v, 0.5), pct(v, 0.9), pct(v, 0.99)
    return f"медиана={med:8.0f}  p90={p90:8.0f}  p99={p99:8.0f}"

inf = {}
for line in open(INF_PATH, encoding="utf-8"):
    if line.strip():
        d = json.loads(line)
        inf[d["msg_id"]] = d

dec = [json.loads(l) for l in open(DEC_PATH, encoding="utf-8") if l.strip()]
joined, miss = [], 0
for d in dec:
    if d.get("media_ts") is None:
        continue
    contribs = d["contributions"]
    trig = None
    best_dt = None
    for mid in d.get("source_msg_ids", []):
        m = inf.get(mid)
        if m is None or m.get("media_ts") is None:
            continue
        dt = abs(m["media_ts"] - d["media_ts"])
        if best_dt is None or dt < best_dt:
            best_dt, trig = dt, m
    if trig is None or trig.get("detect_start_ts") is None or trig.get("detect_done_ts") is None:
        miss += 1
        continue
    joined.append(dict(
        joint=contribs["p_a"] is not None and contribs["p_v"] is not None,
        modality=trig["modality"],
        raw_queue=(trig["detect_start_ts"] - trig["ingest_ts"]) * 1000.0,
        detect=(trig["detect_done_ts"] - trig["detect_start_ts"]) * 1000.0,
        fusion_wait=(d["ts"] - trig["detect_done_ts"]) * 1000.0,
        e2e=(d["ts"] - trig["ingest_ts"]) * 1000.0,
        audio_lag=(max(m["media_ts"] for mid in d.get("source_msg_ids", [])
                       if (m := inf.get(mid)) and m.get("media_ts") is not None)
                   - min(m["media_ts"] for mid in d.get("source_msg_ids", [])
                         if (m := inf.get(mid)) and m.get("media_ts") is not None)) * 1000.0
                  if any(inf.get(mid) and inf[mid].get("media_ts") is not None for mid in d.get("source_msg_ids", [])) else None,
    ))

print(f"решений: {len(dec)}, с media_ts: {sum(1 for d in dec if d.get('media_ts') is not None)}, "
      f"джойн с inference: {len(joined)} (без меток: {miss})\n")

def report(sub, title):
    if not sub:
        return
    print(f"--- {title} (n={len(sub)}) ---")
    for stage in ("raw_queue", "detect", "fusion_wait", "e2e"):
        print(f"  {stage:<12} {stats([r[stage] for r in sub])}")

report(joined, "ВСЕ решения")
report([r for r in joined if r["joint"]], "совместные окна (p_v и p_a)")
report([r for r in joined if not r["joint"]], "mono-окна")
report([r for r in joined if r["modality"] == "video"], "триггер = видео")
report([r for r in joined if r["modality"] == "audio"], "триггер = аудио")
jl = [r["audio_lag"] for r in joined if r["joint"] and r["audio_lag"] is not None]
print(f"\nразброс media_ts внутри joint-окон (ожидание второй модальности): {stats(jl)}")
