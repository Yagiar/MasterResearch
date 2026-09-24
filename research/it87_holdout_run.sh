#!/usr/bin/env bash
# it-87 — frozen final holdout: боевая линейка (AND@0,40 + audio gate) на НОВОМ материале.
# Протокол: research/iterations/it-87-frozen-holdout-PLANNED.md (замеренный стек заморожен).
# Каркас it86_sync_negative_run.sh; отличия: роли сегментов из манифеста holdout-24, loop=false,
# дренаж = 45 с тишины решений (хвост ~10 последних окон не выпускается watermark-политикой —
# зарегистрированная потеря, eval её фиксирует как tail-loss, в знаменатель H1 не входит).
# ЗАПУСКАТЬ ОДНАЖДЫ после записи материала. Двойной запуск того же сегмента = нарушение протокола.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
cd "$MD"

OVR="$MD/infra/docker-compose.it87.yml"
JSONL="$MD/data/decisions/decisions.jsonl"
# HOLD_SUBDIR — песочница прогона внутри sandboxDataForSimulator (bind /data/sandbox:ro).
# Боевое открытие: только дефолт holdout-24. Smoke-прогон обвязки: HOLD_SUBDIR=smoke87 (не боевой замер).
HOLD_SUBDIR="${HOLD_SUBDIR:-holdout-24}"
LOG="$ROOT/research/it87_holdout_run.log"
if [ "$HOLD_SUBDIR" != "holdout-24" ]; then LOG="$ROOT/research/it87_smoke_run.log"; fi
HOLD="$MD/sandboxDataForSimulator/$HOLD_SUBDIR"
MANIFEST="$HOLD/manifest.tsv"
FPS=2
MIN_NEW="${MIN_NEW:-20}"

[ -f "$MANIFEST" ] || { echo "ОТКАЗ: нет $MANIFEST (запись материала не выполнена)"; exit 1; }
COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
COMPOSE_INFRA=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
DOWNSTREAM="ingest-gateway visual-detector acoustic-detector fusion sink"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=video $2=audio (файлы лежат в $HOLD_SUBDIR внутри sandboxDataForSimulator,
  # который уже примонтирован compose как /data/sandbox:ro — новых volume не требуется)
  cat > "$OVR" <<EOF
services:
  source-simulator:
    environment:
      UAVDET_SOURCE__ADAPTER: "media_file"
      UAVDET_SOURCE__ENABLE_AUDIO: "true"
      UAVDET_SOURCE__FPS: "$FPS"
      UAVDET_SOURCE__MEDIA_FILE__VIDEO_PATH: "/data/sandbox/$HOLD_SUBDIR/$1"
      UAVDET_SOURCE__MEDIA_FILE__AUDIO_PATH: "/data/sandbox/$HOLD_SUBDIR/$2"
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
      UAVDET_VISUAL_DETECTOR__VOTE_MODE: "and"
      UAVDET_VISUAL_DETECTOR__VOTE_FLOOR: "0.40"
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

stage() {  # $1=segment_id, $2=video, $3=audio, $4=duration_s
  local id="$1" vid="$2" aud="$3" dur="$4"
  echo
  echo "=== ЭТАП $id (video=$vid dur=${dur}s) $(date +%H:%M:%S) ==="
  write_override "$vid" "$aud"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  # downstream ДО source (урок it-86: подписка-гонка latest-offset)
  "${COMPOSE[@]}" up -d $DOWNSTREAM 2>&1 | tail -1
  sleep 20
  docker exec uavdet-visual-detector env | grep -E "VOTE_MODE=and|VOTE_FLOOR=0.40" \
    || { echo "ОТКАЗ P5 vote"; exit 1; }
  "${COMPOSE[@]}" up -d source-simulator 2>&1 | tail -1
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' uavdet-source-simulator \
    | grep -E "ENABLE_AUDIO=true|ADAPTER=media_file" || { echo "ОТКАЗ P5 source"; exit 1; }
  # ждём конец real-time потока (dur) + 20 с, затем дренаж 45 с тишины решений
  sleep "$(awk "BEGIN{print int($dur)+20}")"
  NINF=$(q "SELECT count(1) FROM uavdet.inference;" | tr -d '[:space:]')
  [ "${NINF:-0}" -gt 0 ] || { echo "ОТКАЗ: пустой стрим на '$id'"; exit 1; }
  P4M=$(q "SELECT DISTINCT model_name FROM uavdet.inference WHERE modality='video';" | tr -d '\r')
  grep -qE '^yolov8s-uav\+uav-yolov8s-bg-best:and@0\.4$' <<<"$P4M" \
    || { echo "ОТКАЗ P4 (ne AND@0,40): $(tr '\n' ' ' <<<"$P4M")"; exit 1; }
  NA=$(q "SELECT count(1) FROM uavdet.inference WHERE modality='audio';" | tr -d '[:space:]')
  [ "${NA:-0}" -gt 0 ] || echo "ВНИМАНИЕ H0: audio_inf=0 на '$id' (оценит eval)"
  PREV=$(offset); STABLE=0; DEADLINE=$((SECONDS+240))
  while [ $SECONDS -lt $DEADLINE ]; do
    sleep 15
    CUR=$(offset)
    if [ "$CUR" -eq "$PREV" ]; then STABLE=$((STABLE+15)); else STABLE=0; fi
    PREV=$CUR
    [ "$STABLE" -ge 45 ] && break
  done
  NEW=$((CUR-OFF_BEFORE))
  [ "$NEW" -ge "$MIN_NEW" ] || { echo "ОТКАЗ дренажа: '$id' дал $NEW решений (<$MIN_NEW)"; exit 1; }
  echo "SCORE_OFF $id $((OFF_BEFORE+1)):$CUR (новых $NEW, тишина ${STABLE}с)"
  echo "TELEMETRY $id audio_inf=$NA video_inf=$NINF"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

exec > >(tee -a "$LOG") 2>&1
echo "=== it-87 holdout $(date '+%F %T') (ОДНОКРАТНОЕ ОТКРЫТИЕ) MIN_NEW=$MIN_NEW ==="
write_override dummy.mp4 dummy.wav
"${COMPOSE[@]}" build source-simulator visual-detector 2>&1 | tail -2
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it87] ждём инфраструктуру (30с)..."; sleep 30
for t in video.raw audio.raw inference decisions; do
  "${COMPOSE_INFRA[@]}" exec -T kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 --create --if-not-exists \
    --topic "$t" --partitions 1 --replication-factor 1 >/dev/null 2>&1 || true
done
# манифест: segment_id<TAB>role<TAB>video<TAB>audio<TAB>duration_s<TAB>gt_start<TAB>gt_end
while IFS=$'\t' read -r id role vid aud dur gs ge; do
  case "$id" in ''|'#'*) continue ;; esac
  stage "$id" "$vid" "$aud" "$dur"
done < "$MANIFEST"
rm -f "$OVR"
echo
echo "=== it-87 завершён; РАЗБОР ОДИН РАЗ: research/.venv/bin/python research/it87_holdout_eval.py --manifest \"$MANIFEST\" <срезы SCORE_OFF> ==="
