#!/usr/bin/env bash
# it-54: сбор чистых срезов для симуляции политик airborne.
# Этап P: sandbox (позитив: 11–65 с airborne=1) → Этап N: negative-session (всё airborne=0).
# Оба: watermark k=5, Δ=0, τ=0.5, target=airborne, motion_floor=0.3.
# Порядок перезапуска этапа (УРОК it-47/53): rm сервисов → сброс групп → TRUNCATE → up.
set -u
cd "$(dirname "$0")/../MasterDiploma"

GPU=${GPU:-1}
WAIT=${WAIT:-100}
BASE_COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml)
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
JSONL="$PWD/data/decisions/decisions.jsonl"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

# УРОК it-52/53: override-файл ОБЯЗАН входить в compose-команду up — иначе
# применяются pilot-дефолты (per-message/presence/Δ=0.1) и «валидация» измеряет не то.
COMPOSE_UP() { docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f infra/docker-compose.it54.yml "$@"; }

write_override() {  # $1=video, $2=audio
  cat > infra/docker-compose.it54.yml <<EOF
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
      UAVDET_FUSION__MOTION_FLOOR: "0.3"
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
  echo "=== ЭТАП $1   $(date +%H:%M:%S) ==="
  write_override "$2" "$3"
  "${BASE_COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  sleep 3
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  COMPOSE_UP up -d $SERVICES 2>&1 | tail -1
  sleep "$WAIT"
  echo "P: $(q 'SELECT count(1), count(*) FILTER (WHERE decision), round(avg(e2e_latency_ms)::numeric,0) FROM uavdet.decisions;')"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${BASE_COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

"${BASE_COMPOSE[@]}" up -d kafka postgres >/dev/null 2>&1 || true
sleep 30

stage "P sandbox"  "/data/sandbox/sandbox-video-for-simulator.mp4" "/data/sandbox/sandbox-audio-for-simulator.wav"
stage "N negative" "/data/sandbox/negative-session.mp4" "/data/sandbox/negative-session.wav"

rm -f infra/docker-compose.it54.yml
echo "=== it-54 сбор завершён ==="
