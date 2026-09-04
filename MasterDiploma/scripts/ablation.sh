#!/usr/bin/env bash
# Ablation-прогон fusion: video-only / audio-only / late / hybrid / late+adaptive gating.
# Перед каждым режимом — TRUNCATE таблиц, после прогона ~WAIT секунд — метрики из PostgreSQL.
# Требует: поднятой инфры (make infra-up && make topics-create && make db-migrate), собранных GPU-образов
# (или они соберутся при первом up). По умолчанию использует GPU-override; для CPU — задать ABLATION_GPU=0.
#
# Запуск:   bash scripts/ablation.sh 2>&1 | tee /tmp/ablation_results.txt
# Дольше прогон каждого режима:  WAIT=180 bash scripts/ablation.sh
#
# ВАЖНО: скрипт ВРЕМЕННО правит configs/pilot.yaml (fusion.mode и fusion.gating.enabled) и в конце
# возвращает дефолт (mode=late, gating off). Если прервать посреди — проверить git diff configs/pilot.yaml.
set -e
cd "$(dirname "$0")/.."

WAIT=${WAIT:-90}
if [ "${ABLATION_GPU:-1}" = "1" ]; then
  COMPOSE="docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml"
  echo "[ablation] GPU-режим (docker-compose.gpu.yml; visual-detector + acoustic-detector на CUDA, FPS=25)"
else
  COMPOSE="docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml"
  echo "[ablation] CPU-режим"
fi
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
PSQL="docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet"

q() { $PSQL -tA -F$'\t' -c "$1"; }

run() {  # $1 = mode, $2 = gating(true|false)
  local mode="$1" gating="${2:-false}"
  echo
  echo "======================================================================"
  echo "=== РЕЖИМ: $mode    (adaptive gating: $gating)    $(date +%H:%M:%S) ==="
  echo "======================================================================"
  sed -i "s/^  mode: .*/  mode: \"$mode\"/" configs/pilot.yaml
  sed -i "s/^    enabled: .*GATING_ENABLED.*/    enabled: $gating  # GATING_ENABLED/" configs/pilot.yaml
  $COMPOSE rm -sf $SERVICES >/dev/null 2>&1 || true
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  echo "[ablation] up --build (первый запуск может тянуть GPU-образы ~5-15 мин)..."
  $COMPOSE up -d --build $SERVICES 2>&1 | tail -3
  echo "[ablation] ждём ${WAIT}с (накопление данных)..."
  sleep "$WAIT"
  echo "-- inference  (modality | label | n | avg_det_latency_ms):"
  q "SELECT modality,label,count(1),round(avg(det_latency_ms)::numeric,1) FROM uavdet.inference GROUP BY 1,2 ORDER BY 1,2;"
  echo "-- decisions  (mode | decision | n | avg_p_fused | with_audio | avg_e2e_ms):"
  q "SELECT mode,decision,count(1),round(avg(p_fused)::numeric,3),count(p_a),round(avg(e2e_latency_ms)::numeric,0) FROM uavdet.decisions GROUP BY 1,2 ORDER BY 2;"
  echo "-- сводка  (n | drone_ratio | avg_p_fused | with_audio | avg_e2e_ms):"
  q "SELECT count(1),round(avg(decision::int)::numeric,3),round(avg(p_fused)::numeric,3),count(p_a),round(avg(e2e_latency_ms)::numeric,0) FROM uavdet.decisions;"
  $COMPOSE rm -sf $SERVICES >/dev/null 2>&1 || true
}

run video-only false
run audio-only false
run late       false
run hybrid     false
run late       true       # late + adaptive gating

# вернуть конфиг в дефолт
sed -i 's/^  mode: .*/  mode: "late"/' configs/pilot.yaml
sed -i 's/^    enabled: .*GATING_ENABLED.*/    enabled: false  # GATING_ENABLED/' configs/pilot.yaml
echo
echo "=== ablation готова. configs/pilot.yaml возвращён в дефолт (mode=late, gating off). ==="
echo "    Проверь: git diff configs/pilot.yaml  (должно быть пусто)"
