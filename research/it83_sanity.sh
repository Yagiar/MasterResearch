#!/usr/bin/env bash
# it-83 sanity: короткий живой A/B (A off, B AND@0,4 по ${WAIT:-120} с) после фикса
# media_ts режима папки (requested_fps). Проверяет: гейты it-82 (env/стрим/model_name) +
#media_ts теперь идёт 1:1 с wall-clock (условие S1) + направление FP(AND)<FP(old)
# не перевернулось (условие S2). Каркас — it82_bg_run.sh (источник: strict400 HD-фонов OI).
# ВНИМАНИЕ (урок it-82): аргументы в heredoc write_override — без экранирования (\$1
# даст литерал в YAML); после генерации оверлея он печатается в лог.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
cd "$MD"

WAIT=${WAIT:-120}
OVR="$MD/infra/docker-compose.it83.yml"
JSONL="$MD/data/decisions/decisions.jsonl"
LOG="$ROOT/research/it83_sanity.log"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
COMPOSE_INFRA=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
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
    "${COMPOSE_INFRA[@]}" exec -T kafka \
      /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
      --delete --group "$g" >/dev/null 2>&1 || true
  done
}

stage() {  # $1=имя, $2=vote_mode, $3=floor, $4=ожидаемый model_name (regex)
  echo
  echo "=== ЭТАП $1: vote=$2 $(date +%H:%M:%S) ==="
  write_override "$2" "$3"
  echo "--- оверлей (самопроверка heredoc):"; cat "$OVR" | grep -E "VOTE_MODE|VIDEO_PATH"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  sleep 15
  echo "-- P5 env:"
  docker exec uavdet-visual-detector env | grep -E "VOTE" || { echo "ОТКАЗ P5"; exit 1; }
  sleep 25
  NINF=$(q "SELECT count(1) FROM uavdet.inference;" | tr -d '[:space:]')
  [ "${NINF:-0}" -gt 0 ] || { echo "ОТКАЗ: пустой стрим на '$1'"; exit 1; }
  echo "-- стрим-гейт OK: $NINF за 25с"
  P4M=$(q "SELECT DISTINCT model_name FROM uavdet.inference WHERE modality='video';" | tr -d '\r')
  grep -qE "$4" <<<"$P4M" || { echo "ОТКАЗ P4: $(tr '\n' ' ' <<<"$P4M")"; exit 1; }
  echo "-- P4-гейт OK: $P4M"
  sleep "$WAIT"
  OFF_AFTER=$(offset)
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$OFF_AFTER"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

exec > >(tee -a "$LOG") 2>&1
echo "=== it-83 sanity $(date '+%F %T') (WAIT=$WAIT) ==="
[ -d "$MD/train/data/_prepared/hi-res-bg-v1/strict400" ] || { echo "ОТКАЗ: нет strict400"; exit 1; }
echo "[it83] пересборка образов (фикс в коде источника):"
write_override off 0.4
"${COMPOSE[@]}" build source-simulator visual-detector 2>&1 | tail -2
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it83] ждём инфраструктуру (30с)..."; sleep 30
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
}
create_topics

stage "A off"     off 0.4 '^yolov8s-uav$'
stage "B AND@0,4" and 0.4 '^yolov8s-uav\+uav-yolov8s-bg-best:and@0\.4$'

rm -f "$OVR"
echo
echo "=== it-83 sanity завершён; разбор: research/.venv/bin/python research/it83_sanity_eval.py <срезы SCORE_OFF> ==="
