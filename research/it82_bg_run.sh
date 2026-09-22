#!/usr/bin/env bash
# it-82: живой фоновый прогон — FP-обещание ансамбля в контуре (предрегистрация:
# research/iterations/it-82-live-bg-fp-PLANNED.md, коммит 96c8cbe).
# Каркас — it81_ab_run.sh (все гейты сохранены: build образа, create_topics, стрим-гейт 25 с,
# P5 env, P3 model_name по ДАННЫМ). Отличия: источник — mmaud_replay на strict400 HD-фонов OI
# (400 кадров без дрона), аудио выключено; метрика — доля положительных решений (FP).
# Этапы: A vote=off -> B and@0,4 -> C or@0,4 (voter=new), 240 с/этап, fusion — боевая поставка.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
cd "$MD"

WAIT=${WAIT:-240}
OVR="$MD/infra/docker-compose.it82.yml"
JSONL="$MD/data/decisions/decisions.jsonl"
LOG="$ROOT/research/it82_bg_run.log"
OLD_PT="$MD/models/visual/yolov8s-uav.pt"
VOTER_PT="$MD/models/visual/uav-yolov8s-bg-best.pt"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=vote_mode, $2=vote_floor
  cat > "$OVR" <<EOF
services:
  source-simulator:
    environment:
      UAVDET_SOURCE__ADAPTER: "mmaud_replay"
      UAVDET_SOURCE__ENABLE_AUDIO: "false"
      UAVDET_SOURCE__MMAUD_REPLAY__VIDEO_PATH: "/data/bgsrc/strict400"
      UAVDET_SOURCE__MMAUD_REPLAY__AUDIO_PATH: ""
      UAVDET_SOURCE__MMAUD_REPLAY__FPS: "2"
      UAVDET_SOURCE__MMAUD_REPLAY__LOOP: "true"
    volumes:
      - ../train/data/_prepared/hi-res-bg-v1:/data/bgsrc:ro
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

stage() {  # $1=имя, $2=vote_mode, $3=floor, $4=ожидаемый model_name (regex, P3-гейт по данным)
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
  sleep 15
  echo "-- P5 env uavdet-visual-detector + source:"
  docker exec uavdet-visual-detector env | grep -E "UAVDET_VISUAL|VOTE" || echo "P5 FAIL: нет vote-env"
  docker exec uavdet-source-simulator env | grep -E "UAVDET_SOURCE" || echo "P5 FAIL: нет source-env"
  sleep 25
  NINF=$(q "SELECT count(1) FROM uavdet.inference;" | tr -d '[:space:]')
  [ "${NINF:-0}" -gt 0 ] || { echo "ОТКАЗ: пустой поток inference на этапе '$1' (25 с) — прогон прерван"; exit 1; }
  echo "-- стрим-гейт OK: inference за 25с = $NINF"
  P4M=$(q "SELECT DISTINCT model_name FROM uavdet.inference WHERE modality='video';" | tr -d '\r')
  echo "-- P3 model_name(video): $(tr '\n' ' ' <<<"$P4M")"
  grep -qE "$4" <<<"$P4M" || { echo "ОТКАЗ P3: ожидаем /$4/, в данных: $(tr '\n' ' ' <<<"$P4M") — прогон прерван"; exit 1; }
  NA=$(q "SELECT count(1) FROM uavdet.inference WHERE modality='audio';" | tr -d '[:space:]')
  echo "-- аудио-строк inference (ожидаем 0): $NA"
  echo "[it82] ждём ${WAIT}с..."
  sleep "$WAIT"
  echo "-- decisions: всего | положительных(FP) | доля | avg e2e_ms:"
  q "SELECT count(1), count(*) FILTER (WHERE decision), \
round(count(*) FILTER (WHERE decision)::numeric / GREATEST(count(1),1), 4), \
round(avg(e2e_latency_ms)::numeric, 0) FROM uavdet.decisions;"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

# --- provenance ---
exec > >(tee -a "$LOG") 2>&1
echo "=== it-82 bg A/B/C $(date '+%F %T') ==="
echo "веса old:   $(sha256sum "$OLD_PT")"
echo "веса voter: $(sha256sum "$VOTER_PT")"
[ -f "$VOTER_PT" ] || { echo "ОТКАЗ: нет $VOTER_PT"; exit 1; }
NS=$(ls "$MD/train/data/_prepared/hi-res-bg-v1/strict400" | wc -l)
[ "$NS" -eq 400 ] || { echo "ОТКАЗ: strict400 содержит $NS != 400"; exit 1; }
echo "strict400 кадров: $NS"

COMPOSE_INFRA=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
write_override off 0.4
echo "[it82] сборка образа visual-detector (кэш слоёв)..."
"${COMPOSE[@]}" build visual-detector 2>&1 | tail -3
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it82] ждём инфраструктуру (30с)..."; sleep 30

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
  echo "[it82] топики на месте: $(tr '\n' ' ' <<<"$have")"
}
create_topics

stage "A vote off (old-only)"   off 0.4 '^yolov8s-uav$'
stage "B AND@0,4"               and 0.4 '^yolov8s-uav\+uav-yolov8s-bg-best:and@0\.4$'
stage "C OR@0,4"                or  0.4 '^yolov8s-uav\+uav-yolov8s-bg-best:or@0\.4$'

rm -f "$OVR"
echo
echo "=== it-82 завершён; стек НЕ перезапущен (ритуал it-68: явное решение о down) ==="
echo "разбор: research/.venv/bin/python research/it82_bg_fp_eval.py <этапы из строк 'SCORE_OFF' лога>"
