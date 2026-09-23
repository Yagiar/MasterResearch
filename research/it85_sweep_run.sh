#!/usr/bin/env bash
# it-85 — throughput saturation sweep: fps {0,5;1;2;3;4;5} × arms {A off, B AND@0,4}
# на строгих 400 HD-фонах OI (loop=false), каркас гейтов it-81…84. Семплер каждые 10 с
# пишет лаги групп visual-detector/fusion + GPU util/mem в research/it85_samples.csv.
# Heredoc-правило: аргументы write_override — без экранирования; оверлей печатается в лог.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
cd "$MD"

OVR="$MD/infra/docker-compose.it85.yml"
JSONL="$MD/data/decisions/decisions.jsonl"
LOG="$ROOT/research/it85_sweep_run.log"
SAMPLES="$ROOT/research/it85_samples.csv"
FLAG="$ROOT/research/.it85_sampler_on"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
COMPOSE_INFRA=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }
klags() {  # $1=group -> сумма LAG по строкам describe
  "${COMPOSE_INFRA[@]}" exec -T kafka /opt/kafka/bin/kafka-consumer-groups.sh \
    --bootstrap-server localhost:9092 --describe 2>/dev/null \
    | awk -v g="$1" '$1==g && $6 ~ /^[0-9]+$/ {s+=$6} END{print s+0}'
}

write_override() {  # $1=vote_mode, $2=vote_floor, $3=fps
  cat > "$OVR" <<EOF
services:
  source-simulator:
    environment:
      UAVDET_SOURCE__ADAPTER: "mmaud_replay"
      UAVDET_SOURCE__ENABLE_AUDIO: "false"
      UAVDET_SOURCE__MMAUD_REPLAY__VIDEO_PATH: "/data/bgsrc/strict400"
      UAVDET_SOURCE__MMAUD_REPLAY__AUDIO_PATH: ""
      UAVDET_SOURCE__MMAUD_REPLAY__FPS: "$3"
      UAVDET_SOURCE__MMAUD_REPLAY__LOOP: "false"
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

sampler() {  # $1=имя этапа; крутится, пока жив флаг-файл
  touch "$FLAG"
  while [ -f "$FLAG" ]; do
    TS=$(date +%s)
    VD=$(klags visual-detector)
    FU=$(klags fusion)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "$TS,$1,$VD,$FU,${GPU:-0,0}" >> "$SAMPLES"
    sleep 10
  done
}

stage() {  # $1=имя, $2=vote_mode, $3=floor, $4=model_regex, $5=fps
  local fps="$5"
  local dur deadline
  dur=$(awk -v f="$fps" 'BEGIN{printf "%d", 400/f}')
  echo
  echo "=== ЭТАП $1: vote=$2 fps=$fps $(date +%H:%M:%S) (стрим ~${dur}с) ==="
  write_override "$2" "$3" "$fps"
  echo "--- оверлей (самопроверка heredoc):"; grep -E "VOTE_MODE|FPS|LOOP" "$OVR"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $SERVICES 2>&1 | tail -1
  sampler "$1" &
  SAMP_PID=$!
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
  # дренаж: ждём тишины решений 45 с ПОСЛЕ окончания отправки; deadline = стрим + 240 с гейтов/дренажа
  deadline=$((SECONDS + dur + 240))
  START_TS=$(date +%s)
  PREV=$(offset); STABLE=0
  while [ $SECONDS -lt $deadline ]; do
    sleep 15
    CUR=$(offset)
    if [ "$CUR" -eq "$PREV" ]; then STABLE=$((STABLE+15)); else STABLE=0; fi
    PREV=$CUR
    ELAPSED=$(( $(date +%s) - START_TS ))
    if [ "$STABLE" -ge 45 ] && [ "$ELAPSED" -ge $((dur + 60)) ]; then break; fi
  done
  rm -f "$FLAG"; wait "$SAMP_PID" 2>/dev/null || true
  NEW=$((CUR-OFF_BEFORE))
  [ "$NEW" -ge 200 ] || { echo "ОТКАЗ дренажа: '$1' дал $NEW решений (<200)"; exit 1; }
  echo "SCORE_OFF $1 $((OFF_BEFORE+1)):$CUR fps=$fps (новых $NEW, тишина ${STABLE}с)"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

exec > >(tee -a "$LOG") 2>&1
echo "=== it-85 throughput sweep $(date '+%F %T') ==="
[ -d "$MD/train/data/_prepared/hi-res-bg-v1/strict400" ] || { echo "ОТКАЗ: нет strict400"; exit 1; }
NIMG=$(ls "$MD/train/data/_prepared/hi-res-bg-v1/strict400" | wc -l)
echo "[it85] кадров в strict400: $NIMG (ожидаем 400)"
[ "$NIMG" -eq 400 ] || { echo "ОТКАЗ: знаменатель не 400"; exit 1; }
echo "[it85] sha весов: $(sha256sum models/visual/yolov8s-uav.pt models/visual/uav-yolov8s-bg-best.pt | cut -c1-12 | tr '\n' ' ')"
[ -f "$SAMPLES" ] || echo "epoch,stage,vd_lag,fusion_lag,gpu_util,gpu_mem_mib" > "$SAMPLES"
write_override off 0.4 2
"${COMPOSE[@]}" build source-simulator visual-detector 2>&1 | tail -2
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it85] ждём инфраструктуру (30с)..."; sleep 30
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

REGEX_A='^yolov8s-uav$'
REGEX_B='^yolov8s-uav\+uav-yolov8s-bg-best:and@0\.4$'
for f in 0.5 1 2 3 4 5; do stage "A@${f}" off 0.4 "$REGEX_A" "$f"; done
for f in 0.5 1 2 3 4 5; do stage "B@${f}" and 0.4 "$REGEX_B" "$f"; done

rm -f "$OVR"
echo
echo "=== it-85 sweep завершён; разбор: research/.venv/bin/python research/it85_sweep_eval.py <срезы SCORE_OFF> ==="
