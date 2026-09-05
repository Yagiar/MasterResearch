#!/usr/bin/env bash
# Негативная синхронная сессия «наземные сегменты» (it-51): клип 17.04 с из наземных
# сегментов sandbox (0–10.5 + 66–72.6, дрон стоит, моторы выключены) — GT: airborne=0 всюду.
# Измеряет FP ПОЛНОЙ системы (fusion, а не только AST) на независимых негативах.
# Этапы: A) per-message k=0 → B) watermark k=5 (оба Δ=0, τ=0.5 — рабочий контур it-43).
# Скоринг: доля FP (decision=true при airborne=0), совместность, e2e — по срезам jsonl.
# Скоринг: research/.venv/bin/python research/score_stages.py --burn-in-s 0 "этап=a:b" c
#          gt_negative_session.csv (см. research/negative_session_score.py)
set -u
cd "$(dirname "$0")/../MasterDiploma"

GPU=${GPU:-1}
WAIT=${WAIT:-100}
COMPOSE_GPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f infra/docker-compose.it51.yml)
COMPOSE_CPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.it51.yml)
if [ "$GPU" = "1" ]; then COMPOSE=("${COMPOSE_GPU[@]}"); else COMPOSE=("${COMPOSE_CPU[@]}"); fi
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
JSONL="$PWD/data/decisions/decisions.jsonl"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=release, $2=k
  cat > infra/docker-compose.it51.yml <<EOF
services:
  source-simulator:
    environment:
      UAVDET_CONFIG: /app/configs/pilot.yaml
      UAVDET_SOURCE__MEDIA_FILE__VIDEO_PATH: "/data/sandbox/negative-session.mp4"
      UAVDET_SOURCE__MEDIA_FILE__AUDIO_PATH: "/data/sandbox/negative-session.wav"
  fusion:
    environment:
      UAVDET_FUSION__WINDOW_RELEASE: "$1"
      UAVDET_FUSION__WINDOW_MAX_WAIT_MS: "5000"
      UAVDET_FUSION__AUDIO_TEMPORAL_K: "$2"
      UAVDET_FUSION__DECISION_THRESHOLD: "0.5"
      UAVDET_FUSION__LATE__DELTA_CONF: "0.0"
      UAVDET_FUSION__LATE__DELTA_UNCONF: "0.0"
EOF
}

reset_groups() {
  for g in fusion sink sink-inference acoustic-detector visual-detector; do
    docker compose -f infra/docker-compose.yml exec -T kafka \
      /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
      --delete --group "$g" >/dev/null 2>&1 || true
  done
}

stage() {  # $1=имя, $2=release, $3=k
  echo
  echo "=== ЭТАП $1: release=$2 k=$3   $(date +%H:%M:%S) ==="
  write_override "$2" "$3"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  sleep 3
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  echo "[neg] ждём ${WAIT}с..."
  sleep "$WAIT"
  echo "-- decisions: всего | FP (decision=true) | доля FP | совместных | avg e2e_ms:"
  q "SELECT count(1), count(*) FILTER (WHERE decision), \
round(count(*) FILTER (WHERE decision)::numeric / GREATEST(count(1),1), 4), \
count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL), \
round(avg(e2e_latency_ms)::numeric, 0) FROM uavdet.decisions;"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

"${COMPOSE[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[neg] ждём инфраструктуру (30с)..."; sleep 30

stage "A permessage k0" per-message 0
stage "B watermark k5"  watermark   5

rm -f infra/docker-compose.it51.yml
"${COMPOSE_CPU[@]}" up -d --force-recreate $SERVICES >/dev/null 2>&1 || true
echo
echo "=== негативная сессия завершена ==="
