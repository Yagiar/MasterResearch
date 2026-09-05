#!/usr/bin/env bash
# Валидация it-52: target=airborne с признаком движения (motion_score).
# Этап N: негативная сессия (наземные сегменты, GT airborne=0) → FP должно упасть против it-51 (24.3%).
# Этап P: sandbox-клип (полёт, GT airborne=1 в 11–65 с) → recall не должен упасть.
# Оба: watermark k=5, Δ=0, τ=0.5, target=airborne, motion_floor=0.15.
set -u
cd "$(dirname "$0")/../MasterDiploma"

GPU=${GPU:-1}
WAIT=${WAIT:-120}
BASE_COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml)
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
JSONL="$PWD/data/decisions/decisions.jsonl"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=video_path, $2=audio_path
  cat > infra/docker-compose.it52.yml <<EOF
services:
  source-simulator:
    environment:
      UAVDET_CONFIG: /app/configs/pilot.yaml
      UAVDET_SOURCE__MEDIA_FILE__VIDEO_PATH: "$1"
      UAVDET_SOURCE__MEDIA_FILE__AUDIO_PATH: "$2"
  fusion:
    environment:
      UAVDET_FUSION__WINDOW_RELEASE: "watermark"
      UAVDET_FUSION__WINDOW_MAX_WAIT_MS: "5000"
      UAVDET_FUSION__AUDIO_TEMPORAL_K: "5"
      UAVDET_FUSION__DECISION_THRESHOLD: "0.5"
      UAVDET_FUSION__LATE__DELTA_CONF: "0.0"
      UAVDET_FUSION__LATE__DELTA_UNCONF: "0.0"
      UAVDET_FUSION__TARGET: "airborne"
      UAVDET_FUSION__MOTION_FLOOR: "0.15"
EOF
}

reset_groups() {
  for g in fusion sink sink-inference acoustic-detector visual-detector; do
    docker compose -f infra/docker-compose.yml exec -T kafka \
      /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
      --delete --group "$g" >/dev/null 2>&1 || true
  done
}

stage() {  # $1=имя, $2=video, $3=audio
  echo
  echo "=== ЭТАП $1   $(date +%H:%M:%S) ==="
  write_override "$2" "$3"
  "${BASE_COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  sleep 3
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${BASE_COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  echo "[it52] ждём ${WAIT}с..."
  sleep "$WAIT"
  echo "-- decisions: всего | positive | доля | совместных | avg e2e_ms:"
  q "SELECT count(1), count(*) FILTER (WHERE decision), \
round(count(*) FILTER (WHERE decision)::numeric / GREATEST(count(1),1), 4), \
count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL), \
round(avg(e2e_latency_ms)::numeric, 0) FROM uavdet.decisions;"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${BASE_COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

"${BASE_COMPOSE[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it52] ждём инфраструктуру (30с)..."; sleep 30

stage "N negative-session airborne" "/data/sandbox/negative-session.mp4" "/data/sandbox/negative-session.wav"
stage "P sandbox airborne"          "/data/sandbox/sandbox-video-for-simulator.mp4" "/data/sandbox/sandbox-audio-for-simulator.wav"

rm -f infra/docker-compose.it52.yml
echo "=== валидация it-52 завершена ==="
