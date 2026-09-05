#!/usr/bin/env bash
# Ablation v3 (research/it-42): живой GPU-прогон ПОСЛЕ фиксов ревью (it-31..35) —
# проверка гипотез it-30 о блокере доставки аудио (10% совместных окон).
#
# Новое против v2:
#   - скоринг по media_ts БЕЗ подгонки фазы по GT (it-35);
#   - метрики совместности: count(p_v NOT NULL AND p_a NOT NULL) в PG (it-32/35);
#   - ступенчатый старт: прогрев AST до включения видео (гипотеза (а) it-30);
#   - stage D: расширенный lateness буфера (гипотеза: буфер v1 разрушал пары).
# Этапы: A) контроль as-was (всё сразу, k=0) → B) прогрев AST + k=0 → C) прогрев + k=5
#        → D) прогрев + k=5 + lateness=30 с.
# Запуск:  bash research/ablation_v3.sh 2>&1 | tee /tmp/ablation_v3.log
set -u
cd "$(dirname "$0")/../MasterDiploma"

GPU=${GPU:-1}
BASE="$(pwd -P)"
JSONL="$BASE/data/decisions/decisions.jsonl"
OVR="$BASE/infra/docker-compose.it42.yml"
WAIT=${WAIT:-120}
WARM=${WARM:-90}

COMPOSE_GPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml)
COMPOSE_CPU=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
if [ "$GPU" = "1" ]; then COMPOSE=("${COMPOSE_GPU[@]}"); else COMPOSE=("${COMPOSE_CPU[@]}"); fi
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
APP_ONLY="ingest-gateway visual-detector acoustic-detector fusion sink"   # без источника
q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=k, $2=gate, $3=lateness_ms
  cat > "$OVR" <<EOF
services:
  fusion:
    environment:
      UAVDET_FUSION__AUDIO_TEMPORAL_K: "$1"
      UAVDET_FUSION__AUDIO_HEALTH_GATE: "$2"
      UAVDET_FUSION__WINDOW_LATENESS_MS: "$3"
EOF
}

# Урок этапа A первого прогона: в топиках остаются недопотреблённые сообщения прежних
# прогонов (offset'ы групп персистентны) — старые кадры без media_ts замешиваются в
# измерение. Перед этапом: остановить сервисы, СБРОСИТЬ группы (auto_offset_reset=latest
# возьмёт только свежее), затем TRUNCATE.
reset_groups() {
  for g in fusion sink sink-inference acoustic-detector visual-detector; do
    docker compose -f infra/docker-compose.yml exec -T kafka \
      /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
      --delete --group "$g" >/dev/null 2>&1 || true
  done
}

report_stage() {  # $1=имя этапа (для печатей)
  echo "-- inference по модальностям (проверка потерь, гипотеза (б) it-30):"
  q "SELECT modality, count(1) FROM uavdet.inference GROUP BY 1 ORDER BY 1;"
  echo "-- decisions: всего | с обеими модальностями | доля совместных | avg e2e_ms:"
  q "SELECT count(1), count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL), \
round(count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL)::numeric / GREATEST(count(1),1), 3), \
round(avg(e2e_latency_ms)::numeric, 0) FROM uavdet.decisions;"
  echo "-- решение «дрон» по режимам:"
  q "SELECT mode, decision, count(1) FROM uavdet.decisions GROUP BY 1,2 ORDER BY 1,2;"
}

stage() {  # $1=имя, $2=k, $3=gate, $4=lateness_ms, $5=staged(yes/no)
  echo
  echo "=================================================================="
  echo "=== ЭТАП $1: k=$2 gate=$3 lateness_ms=$4 staged=$5   $(date +%H:%M:%S) ==="
  echo "=================================================================="
  write_override "$2" "$3" "$4"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  echo "[v3] offset jsonl до этапа: $OFF_BEFORE"
  if [ "$5" = "yes" ]; then
    echo "[v3] ступенчатый старт: детекторы+fusion без источника, прогрев ${WARM}с (AST грузит модель)..."
    "${COMPOSE[@]}" up -d $APP_ONLY 2>&1 | tail -1
    sleep "$WARM"
    echo "[v3] прогрев завершён — включаем источник+сток:"
    "${COMPOSE[@]}" up -d source-simulator sink 2>&1 | tail -1
  else
    "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  fi
  echo "[v3] ждём ${WAIT}с..."
  sleep "$WAIT"
  report_stage "$1"
  OFF_AFTER=$(offset)
  echo "[v3] offset jsonl после этапа: $OFF_AFTER → срез: $((OFF_BEFORE+1))..$OFF_AFTER"
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

"${COMPOSE[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[v3] ждём инфраструктуру (30с)..."; sleep 30

stage "A контроль as-was"   0 false 2000  no
stage "B прогрев AST k=0"   0 false 2000  yes
stage "C прогрев AST k=5"   5 false 2000  yes
stage "D прогрев k=5 late30" 5 false 30000 yes

rm -f "$OVR"
"${COMPOSE_CPU[@]}" up -d --force-recreate $SERVICES >/dev/null 2>&1 || true
echo
echo "=== ablation v3 завершён. GT-скоринг срезов: research/.venv/bin/python research/score_stages.py <имя>=<a:b> (media_ts, без подгонки) ==="
