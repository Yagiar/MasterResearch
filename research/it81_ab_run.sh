#!/usr/bin/env bash
# it-81: живой A/B ансамбля old⊕new в видео-ветке конвейера (предрегистрация:
# research/iterations/it-81-live-ensemble-ab-PLANNED.md).
# Каркас — ablation_v5.sh (240 с/этап, burn-in 90, media_ts), НО с исправленным механизмом:
# override-файл передаётся явно через -f "$OVR" (в v3/v5 он был мёртвый — см. аудит-находку it-81),
# и каждый этап печатает env контейнера visual-detector ДО измерений (критерий P5).
# Этапы: A vote=off (baseline, одиночный old) -> B vote=and@0,4 -> C vote=or@0,4 (voter=new).
# Fusion во всех этапах — боевая поставка: watermark, k=5, τ=0,5, Δ=0.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
cd "$MD"

WAIT=${WAIT:-240}
OVR="$MD/infra/docker-compose.it81.yml"
JSONL="$MD/data/decisions/decisions.jsonl"
LOG="$ROOT/research/it81_ab_run.log"
OLD_PT="$MD/models/visual/yolov8s-uav.pt"
VOTER_PT="$MD/models/visual/uav-yolov8s-bg-best.pt"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=vote_mode, $2=vote_floor
  cat > "$OVR" <<EOF
services:
  fusion:
    environment:
      UAVDET_FUSION__WINDOW_RELEASE: "watermark"
      UAVDET_FUSION__WINDOW_MAX_WAIT_MS: "5000"
      UAVDET_FUSION__AUDIO_TEMPORAL_K: "5"
      UAVDET_FUSION__DECISION_THRESHOLD: "0.5"
      UAVDET_FUSION__LATE__DELTA_CONF: "0.0"
      UAVDET_FUSION__LATE__DELTA_UNCONF: "0.0"
  visual-detector:
    environment:
      UAVDET_VISUAL_DETECTOR__VOTE_MODE: "$1"
      UAVDET_VISUAL_DETECTOR__VOTE_FLOOR: "$2"
      UAVDET_VISUAL_DETECTOR__VOTE_WEIGHTS_PATH: "/models/visual/uav-yolov8s-bg-best.pt"
EOF
}

reset_groups() {
  for g in fusion sink sink-inference acoustic-detector visual-detector; do
    docker compose -f infra/docker-compose.yml exec -T kafka \
      /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
      --delete --group "$g" >/dev/null 2>&1 || true
  done
}

stage() {  # $1=имя, $2=vote_mode, $3=floor
  echo
  echo "=================================================================="
  echo "=== ЭТАП $1: vote=$2 floor=$3   $(date +%H:%M:%S) ==="
  echo "=================================================================="
  write_override "$2" "$3"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  sleep 15  # контейнеру — загрузить веса
  # P5: env видеосервиса должен содержать vote-переменные ДО измерений
  echo "-- P5 env uavdet-visual-detector:"
  docker exec uavdet-visual-detector env | grep -E "UAVDET_VISUAL|VOTE" || echo "P5 FAIL: нет vote-env"
  # ранний стрим-гейт: за 25 с inference обязан двинуться, иначе этап пустой — abort
  sleep 25
  NINF=$(q "SELECT count(1) FROM uavdet.inference;" | tr -d '[:space:]')
  [ "${NINF:-0}" -gt 0 ] || { echo "ОТКАЗ: пустой поток inference на этапе '$1' (25 с) — прогон прерван"; exit 1; }
  echo "-- стрим-гейт OK: inference за 25с = $NINF"
  echo "[it81] ждём ${WAIT}с..."
  sleep "$WAIT"
  echo "-- decisions: всего | совместных | доля | avg e2e_ms:"
  q "SELECT count(1), count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL), \
round(count(*) FILTER (WHERE p_v IS NOT NULL AND p_a IS NOT NULL)::numeric / GREATEST(count(1),1), 3), \
round(avg(e2e_latency_ms)::numeric, 0) FROM uavdet.decisions;"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

# --- provenance (P4) ---
exec > >(tee -a "$LOG") 2>&1
echo "=== it-81 A/B $(date '+%F %T') ==="
echo "веса old:   $(sha256sum "$OLD_PT")"
echo "веса voter: $(sha256sum "$VOTER_PT" 2>/dev/null || echo 'НЕТ ФАЙЛА — копировать best.pt it-65 в models/visual/')"
[ -f "$VOTER_PT" ] || { echo "ОТКАЗ: нет $VOTER_PT"; exit 1; }

COMPOSE_INFRA=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
write_override off 0.4  # валидный YAML до первого up infra
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it81] ждём инфраструктуру (30с)..."; sleep 30

# топики (аналог make topics-create; в этом Kafka auto-create выключен — пустой прогон без них)
create_topics() {
  local t have
  for t in video.raw audio.raw inference decisions; do
    "${COMPOSE_INFRA[@]}" exec -T kafka /opt/kafka/bin/kafka-topics.sh \
      --bootstrap-server localhost:9092 --create --if-not-exists \
      --topic "$t" --partitions 1 --replication-factor 1 >/dev/null 2>&1 || true
  done
  have=$("${COMPOSE_INFRA[@]}" exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list 2>/dev/null)
  for t in video.raw audio.raw inference decisions; do
    grep -qx "$t" <<<"$have" || { echo "ОТКАЗ: топик $t не создан"; exit 1; }
  done
  echo "[it81] топики на месте: $(tr '\n' ' ' <<<"$have")"
}
create_topics

stage "A vote off (old-only)"   off 0.4
stage "B AND@0,4"               and 0.4
stage "C OR@0,4"                or  0.4

rm -f "$OVR"
echo
echo "=== it-81 A/B завершён; стек НЕ перезапущен (ритуал it-68: явное решение о down) ==="
echo "скоринг: research/.venv/bin/python research/score_stages.py --burn-in-s 90 <этапы из строк 'SCORE_OFF' лога>"
