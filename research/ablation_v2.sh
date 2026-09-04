#!/usr/bin/env bash
# Ablation v2 (research/it-30): проверка ВНЕДРЁННЫХ улучшений fusion в живом пайплайне.
# Этапы: B) k=0 (контроль, GPU) → C) k=5 (медиана) → D) k=5+health_gate.
# Отличия от scripts/ablation.sh: env через override-файл (без правки configs/pilot.yaml),
# фиксация offset decisions.jsonl на каждом этапе (урок it-03: jsonl без ротации).
#
# Запуск:  bash research/ablation_v2.sh 2>&1 | tee /tmp/ablation_v2.log
set -u
cd "$(dirname "$0")/../MasterDiploma"

WAIT=${WAIT:-90}
GPU=${GPU:-1}
BASE=/home/otrix/code/GeneralFolderMasterDiploma
JSONL="$BASE/MasterDiploma/data/decisions/decisions.jsonl"
OVR="$BASE/MasterDiploma/infra/docker-compose.it30.yml"

if [ "$GPU" = "1" ]; then
  COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
else
  COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f "$OVR")
fi
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }

write_override() {  # $1=k, $2=gate
  cat > "$OVR" <<EOF
services:
  fusion:
    environment:
      UAVDET_FUSION__AUDIO_TEMPORAL_K: "$1"
      UAVDET_FUSION__AUDIO_HEALTH_GATE: "$2"
EOF
}

offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

stage() {  # $1=имя, $2=k, $3=gate
  echo
  echo "=================================================================="
  echo "=== ЭТАП $1: audio_temporal_k=$2, health_gate=$3   $(date +%H:%M:%S) ==="
  echo "=================================================================="
  write_override "$2" "$3"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  echo "[v2] offset jsonl до этапа: $OFF_BEFORE"
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -2
  echo "[v2] ждём ${WAIT}с..."
  sleep "$WAIT"
  echo "-- decisions (mode|decision|n|avg_p_fused|with_audio|avg_e2e_ms):"
  q "SELECT mode,decision,count(1),round(avg(p_fused)::numeric,3),count(p_a),round(avg(e2e_latency_ms)::numeric,0) FROM uavdet.decisions GROUP BY 1,2 ORDER BY 1,2;"
  echo "-- сводка (n | drone_ratio | with_audio | avg_e2e):"
  q "SELECT count(1),round(avg(decision::int)::numeric,3),count(p_a),round(avg(e2e_latency_ms)::numeric,0) FROM uavdet.decisions;"
  OFF_AFTER=$(offset)
  echo "[v2] offset jsonl после этапа: $OFF_AFTER → срез решений этапа: строки $((OFF_BEFORE+1))..$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

# Этап A (текущий CPU k=0 прогон) уже завершён пользователем — фиксируем offset до GPU-этапов:
echo "=== offset jsonl (включая пользовательский CPU k=0 прогон): $(offset) ==="

stage "B k=0 (GPU контроль)"  0 false
stage "C k=5 (медиана)"       5 false
stage "D k=5+health_gate"     5 true

# вернуть конфиг compose в дефолт (без override-слоя)
rm -f "$OVR"
docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml up -d --force-recreate $SERVICES >/dev/null 2>&1 || true
echo
echo "=== ablation v2 завершён. Метрики этапов выше; GT-скоринг — research/score_fusion_vs_gt.py по offset-срезам ==="
