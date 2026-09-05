#!/usr/bin/env bash
# Ablation v5 (research/it-46): финальный протокол — длинные этапы (240 с) для полного
# стационарного участка после burn-in 90 с, скоринг NORM (нормировка по медиа-секундам,
# it-45). Δ=0, τ=0.5. Этапы: A) per-message k=0 → B) watermark k=5 → C) watermark k=0
# (отделить эффект watermark от медианы).
# Скоринг: research/.venv/bin/python research/score_stages.py --burn-in-s 90 <этапы из лога>
set -u
cd "$(dirname "$0")/../MasterDiploma"

GPU=${GPU:-1}
WAIT=${WAIT:-240}
COMPOSE_GPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml)
COMPOSE_CPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
if [ "$GPU" = "1" ]; then COMPOSE=("${COMPOSE_GPU[@]}"); else COMPOSE=("${COMPOSE_CPU[@]}"); fi
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
OVR="$PWD/infra/docker-compose.it46.yml"
JSONL="$PWD/data/decisions/decisions.jsonl"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=release, $2=k
  cat > "$OVR" <<EOF
services:
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
  echo "=================================================================="
  echo "=== ЭТАП $1: release=$2 k=$3   $(date +%H:%M:%S) ==="
  echo "=================================================================="
  write_override "$2" "$3"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  echo "[v5] ждём ${WAIT}с..."
  sleep "$WAIT"
  echo "-- decisions: всего | совместных | доля | avg e2e_ms:"
  q "SELECT count(1), count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL), \
round(count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL)::numeric / GREATEST(count(1),1), 3), \
round(avg(e2e_latency_ms)::numeric, 0) FROM uavdet.decisions;"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

"${COMPOSE[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[v5] ждём инфраструктуру (30с)..."; sleep 30

stage "A permessage k0"  per-message 0
stage "B watermark k5"   watermark   5
stage "C watermark k0"   watermark   0

rm -f "$OVR"
"${COMPOSE_CPU[@]}" up -d --force-recreate $SERVICES >/dev/null 2>&1 || true
echo
echo "=== ablation v5 завершён ==="
