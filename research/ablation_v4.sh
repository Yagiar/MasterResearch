#!/usr/bin/env bash
# Ablation v4 (research/it-43): watermark-выпуск окон (it-44) против per-message.
# Δ-правило отключено (it-41: не помогает на честных вероятностях).
# Этапы: A) per-message k=0 (контроль) → B) watermark k=0 → C) watermark k=5
#        → D) watermark k=5 τ=0.55 (офлайн-оптимум для late+медианы, it-40).
# Скоринг: research/score_ablation_v4.sh /tmp/ablation_v4.log
set -u
cd "$(dirname "$0")/../MasterDiploma"

GPU=${GPU:-1}
WAIT=${WAIT:-120}
COMPOSE_GPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml)
COMPOSE_CPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
if [ "$GPU" = "1" ]; then COMPOSE=("${COMPOSE_GPU[@]}"); else COMPOSE=("${COMPOSE_CPU[@]}"); fi
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
OVR="$PWD/infra/docker-compose.it43.yml"
JSONL="$PWD/data/decisions/decisions.jsonl"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=release, $2=maxwait_ms, $3=k, $4=tau
  cat > "$OVR" <<EOF
services:
  fusion:
    environment:
      UAVDET_FUSION__WINDOW_RELEASE: "$1"
      UAVDET_FUSION__WINDOW_MAX_WAIT_MS: "$2"
      UAVDET_FUSION__AUDIO_TEMPORAL_K: "$3"
      UAVDET_FUSION__DECISION_THRESHOLD: "$4"
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

stage() {  # $1=имя, $2=release, $3=maxwait, $4=k, $5=tau
  echo
  echo "=================================================================="
  echo "=== ЭТАП $1: release=$2 maxwait=$3 k=$4 tau=$5   $(date +%H:%M:%S) ==="
  echo "=================================================================="
  write_override "$2" "$3" "$4" "$5"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  echo "[v4] ждём ${WAIT}с..."
  sleep "$WAIT"
  echo "-- decisions: всего | совместных (p_v и p_a) | доля | avg e2e_ms:"
  q "SELECT count(1), count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL), \
round(count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL)::numeric / GREATEST(count(1),1), 3), \
round(avg(e2e_latency_ms)::numeric, 0) FROM uavdet.decisions;"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

"${COMPOSE[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[v4] ждём инфраструктуру (30с)..."; sleep 30

stage "A permessage k0 d0"       per-message 2000 0 0.5
stage "B watermark k0 d0"        watermark   5000 0 0.5
stage "C watermark k5 d0"        watermark   5000 5 0.5
stage "D watermark k5 tau055"    watermark   5000 5 0.55

rm -f "$OVR"
"${COMPOSE_CPU[@]}" up -d --force-recreate $SERVICES >/dev/null 2>&1 || true
echo
echo "=== ablation v4 завершён ==="
