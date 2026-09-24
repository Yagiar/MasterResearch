#!/usr/bin/env bash
# it-86 — мультимодальный синхронный негатив V0/V1/V2 на media_file (negative-session, GT «дрона нет»).
# Каркас it84_paired_run.sh: оверлей-файл, env-P5, стрим/P4-гейты, дренажный конец (тишина 45 с).
# ВНИМАНИЕ: 17-с материал => это sanity-обвязка (M0) + дескриптив; M1–M3-вердикты — только на
# материале 10–30 мин (MIN_NEW гейт поднимается вместе с материалом). Heredoc-правило: аргументы
# write_override без экранирования.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
cd "$MD"

OVR="$MD/infra/docker-compose.it86.yml"
JSONL="$MD/data/decisions/decisions.jsonl"
LOG="$ROOT/research/it86_sync_run.log"
NEG_VIDEO="/data/sandbox/negative-session.mp4"
NEG_AUDIO="/data/sandbox/negative-session.wav"
FPS="${FPS:-2}"
MIN_NEW="${MIN_NEW:-5}"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
COMPOSE_INFRA=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=vote_mode, $2=vote_floor, $3=enable_audio, $4=audio_path
  cat > "$OVR" <<EOF
services:
  source-simulator:
    environment:
      UAVDET_SOURCE__ADAPTER: "media_file"
      UAVDET_SOURCE__ENABLE_AUDIO: "$3"
      UAVDET_SOURCE__FPS: "$FPS"
      UAVDET_SOURCE__MEDIA_FILE__VIDEO_PATH: "$NEG_VIDEO"
      UAVDET_SOURCE__MEDIA_FILE__AUDIO_PATH: "$4"
      UAVDET_SOURCE__MEDIA_FILE__LOOP: "false"
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

stage() {  # $1=имя, $2=vote_mode, $3=floor, $4=enable_audio, $5=audio_path, $6=ожидаемый model_name regex
  echo
  echo "=== ЭТАП $1: vote=$2 audio=$4 $(date +%H:%M:%S) ==="
  write_override "$2" "$3" "$4" "$5"
  echo "--- оверлей (самопроверка heredoc):"; grep -E "VOTE_MODE|ENABLE_AUDIO|AUDIO_PATH" "$OVR"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  sleep 20
  echo "-- P5 env:"
  docker exec uavdet-visual-detector env | grep -E "VOTE" || { echo "ОТКАЗ P5"; exit 1; }
  # 17-с материал: source-simulator штатно завершается (exit 0) до этой проверки —
  # env читаем docker inspect (работает и на остановленном контейнере), не exec.
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' uavdet-source-simulator \
    | grep -E "ENABLE_AUDIO|ADAPTER" || { echo "ОТКАЗ P5 source"; exit 1; }
  sleep 30
  NINF=$(q "SELECT count(1) FROM uavdet.inference;" | tr -d '[:space:]')
  [ "${NINF:-0}" -gt 0 ] || { echo "ОТКАЗ: пустой стрим на '$1'"; exit 1; }
  echo "-- стрим-гейт OK: $NINF за ~50с"
  P4M=$(q "SELECT DISTINCT model_name FROM uavdet.inference WHERE modality='video';" | tr -d '\r')
  grep -qE "$6" <<<"$P4M" || { echo "ОТКАЗ P4: $(tr '\n' ' ' <<<"$P4M")"; exit 1; }
  echo "-- P4-гейт OK: $P4M"
  # дренаж: 17 с стрима + тёп-запуск; тишина 45 с, потолок 300 с
  PREV=$(offset); STABLE=0; DEADLINE=$((SECONDS+300))
  while [ $SECONDS -lt $DEADLINE ]; do
    sleep 15
    CUR=$(offset)
    if [ "$CUR" -eq "$PREV" ]; then STABLE=$((STABLE+15)); else STABLE=0; fi
    PREV=$CUR
    [ "$STABLE" -ge 45 ] && break
  done
  NEW=$((CUR-OFF_BEFORE))
  [ "$NEW" -ge "$MIN_NEW" ] || { echo "ОТКАЗ дренажа: '$1' дал $NEW решений (<$MIN_NEW)"; exit 1; }
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$CUR (новых $NEW, тишина ${STABLE}с)"
  NA=$(q "SELECT count(1) FROM uavdet.inference WHERE modality='audio';" | tr -d '[:space:]')
  NV=$(q "SELECT count(1) FROM uavdet.inference WHERE modality='video';" | tr -d '[:space:]')
  echo "TELEMETRY $1 audio_inf=$NA video_inf=$NV"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

exec > >(tee -a "$LOG") 2>&1
echo "=== it-86 sync negative $(date '+%F %T') MIN_NEW=$MIN_NEW fps=$FPS ==="
[ -f "$ROOT/MasterDiploma/sandboxDataForSimulator/negative-session.mp4" ] || { echo "ОТКАЗ: нет negative-session"; exit 1; }
write_override off 0.4 false ""
"${COMPOSE[@]}" build source-simulator visual-detector 2>&1 | tail -2
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it86] ждём инфраструктуру (30с)..."; sleep 30
for t in video.raw audio.raw inference decisions; do
  "${COMPOSE_INFRA[@]}" exec -T kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 --create --if-not-exists \
    --topic "$t" --partitions 1 --replication-factor 1 >/dev/null 2>&1 || true
done
REGEX_A='^yolov8s-uav$'
REGEX_B='^yolov8s-uav\+uav-yolov8s-bg-best:and@0\.4$'

stage "V0 off-audio" off 0.4 false ""            "$REGEX_A"
stage "V1 +audio"    off 0.4 true  "$NEG_AUDIO"  "$REGEX_A"
stage "V2 and+audio" and 0.4 true  "$NEG_AUDIO"  "$REGEX_B"

rm -f "$OVR"
echo
echo "=== it-86 завершён; разбор: research/.venv/bin/python research/it86_sync_negative_eval.py <срезы SCORE_OFF> ==="
