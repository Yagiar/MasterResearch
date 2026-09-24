#!/usr/bin/env bash
# it-86-full — мультимодальный синхронный негатив V0/V1/V2 на НЕГАТИВАХ holdout-24 (GT «дрона нет»).
# Протокол: iterations/it-86-multimodal-sync-negative.md (вердикт-правило для 10–30 мин: M1–M3).
# Каркас it87_holdout_run.sh (отказные гейты строже sanity-runner'а): loop=false, тишина-дренаж
# с ОТКАЗОМ, up-front валидация neg-сегментов. ПОРЯДОК: только ПОСЛЕ разбора it-87 (шаг 5
# чек-листа роадмапа) — гейт ниже требует it87_holdout_summary.txt.
# Выход: на плечо — одна строка `SCORE_OFF "<V0 off-audio|V1 +audio|V2 and+audio>" A:B` (объединённый
# срез всех neg-сегментов плеча; имена ровно как ждёт it86_sync_negative_eval.py), плюс per-segment
# `SEGSTAT` для аудита. Аргументы разбора: "V0 off-audio=A:B" и т.д.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
cd "$MD"

OVR="$MD/infra/docker-compose.it86f.yml"
JSONL="$MD/data/decisions/decisions.jsonl"
LOG="$ROOT/research/it86_full_run.log"
# Страж однократности (тот же класс, что в it87): tee -a дописал бы в старый лог — SCORE_OFF
# плечей склеились бы из двух прогонов (первое/последнее решение плеча из разных миров).
# Повтор после прерванного запуска — только с письменной пометкой автора (шаг 6 чек-листа).
if [ -e "$LOG" ]; then
  echo "ОТКАЗ pre-flight: боевой лог $LOG уже существует (след первого или прерванного запуска) — пуск не начнётся; см. шаг 6 чек-листа роадмапа"; exit 1
fi
HOLD_SUBDIR="holdout-24"
HOLD="$MD/sandboxDataForSimulator/$HOLD_SUBDIR"
MANIFEST="$HOLD/manifest.tsv"
FPS=2

[ -f "$ROOT/research/it87_holdout_summary.txt" ] || { echo "ОТКАЗ: разбор it-87 не завершён (нет it87_holdout_summary.txt) — it-86-full идёт ПОСЛЕ него по чек-листу"; exit 1; }
[ -f "$MANIFEST" ] || { echo "ОТКАЗ: нет $MANIFEST"; exit 1; }

# --- up-front: neg-сегменты (файлы, роли, кавычки); суммарный негатив >=600 c ---
NEGDUR=0
declare -a SEGS=()
declare -A SEENID=()
while IFS=$'\t' read -r id role vid aud dur gs ge; do
  case "$id" in ''|'#'*) continue ;; esac
  [[ "$id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ОТКАЗ pre-flight: segment_id '$id' — только буквы/цифры/точка/дефис/подчёркивание (пробел ломает awk-идиому разбора SCORE_OFF)"; exit 1; }
  [[ "$id" == -* ]] && { echo "ОТКАЗ pre-flight: segment_id '$id' — ведущий дефис (тот же namespace-конфликт с eval-флагами, что в it87; манифест у прогонов общий)"; exit 1; }
  [[ -n "${SEENID[$id]:-}" ]] && { echo "ОТКАЗ pre-flight: segment_id '$id' повторяется (дубликат съедал бы сегмент в eval-словаре)"; exit 1; }
  SEENID[$id]=1
  [ "$role" = "pos" ] || [ "$role" = "neg" ] || { echo "ОТКАЗ pre-flight: '$id' — role='$role' (допустимы только pos|neg)"; exit 1; }
  [ "$role" = "neg" ] || continue
  [[ "$vid$aud" == *'"'* || "$vid$aud" == *'$'* || "$vid$aud" == *'`'* || "$vid$aud" == *'\'* ]] && { echo "ОТКАЗ pre-flight: '$id' — кавычка/\$/backtick/обратный слэш в имени (heredoc-оверрай раскрывает \$ и \` молча)"; exit 1; }
  for nf in "$dur" "$gs" "$ge"; do
    [[ "$nf" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "ОТКАЗ pre-flight: '$id' — поле '$nf' не число (точка, не запятая; '95,0' в awk читается как 95)"; exit 1; }
  done
  awk "BEGIN{exit !($dur+0 >= 20)}" || { echo "ОТКАЗ pre-flight: '$id' — dur=$dur < 20 с (MIN_NEW.stage floor=20 решений был бы недостижим в коротком окне — отказ наступил бы ПОСЛЕ tee, частичным логом)"; exit 1; }
  [ -f "$HOLD/$vid" ] || { echo "ОТКАЗ pre-flight: '$id' — нет видео $HOLD/$vid"; exit 1; }
  [ -f "$HOLD/$aud" ] || { echo "ОТКАЗ pre-flight: '$id' — нет аудио $HOLD/$aud"; exit 1; }
  NEGDUR=$(awk "BEGIN{print $NEGDUR+($dur+0)}")
  SEGS+=("$id"$'\t'"$vid"$'\t'"$aud"$'\t'"$dur")
done < <(sed '1s/^\xef\xbb\xbf//' "$MANIFEST" | tr -d '\r')
awk "BEGIN{exit !($NEGDUR >= 600)}" || { echo "ОТКАЗ pre-flight: суммарный негатив ${NEGDUR}с < 600с (спека M1–M3)"; exit 1; }
echo "[it86-full] pre-flight OK: neg-сегментов=${#SEGS[@]}, суммарно ${NEGDUR}с"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml -f "$OVR")
COMPOSE_INFRA=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml)
SERVICES="source-simulator ingest-gateway visual-detector acoustic-detector fusion sink"
DOWNSTREAM="ingest-gateway visual-detector acoustic-detector fusion sink"

q() { docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -tA -c "$1"; }
offset() { wc -l < "$JSONL" 2>/dev/null || echo 0; }

write_override() {  # $1=video $2=audio $3=enable_audio $4=vote_mode $5=vote_floor
  cat > "$OVR" <<EOF
services:
  source-simulator:
    environment:
      UAVDET_SOURCE__ADAPTER: "media_file"
      UAVDET_SOURCE__ENABLE_AUDIO: "$3"
      UAVDET_SOURCE__FPS: "$FPS"
      UAVDET_SOURCE__MEDIA_FILE__VIDEO_PATH: "/data/sandbox/$HOLD_SUBDIR/$1"
      UAVDET_SOURCE__MEDIA_FILE__AUDIO_PATH: "$([ -n "$2" ] && echo "/data/sandbox/$HOLD_SUBDIR/$2")"
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
      UAVDET_VISUAL_DETECTOR__VOTE_MODE: "$4"
      UAVDET_VISUAL_DETECTOR__VOTE_FLOOR: "$5"
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

REGEX_SINGLE='^yolov8s-uav$'
REGEX_AND='^yolov8s-uav\+uav-yolov8s-bg-best:and@0\.4$'

stage() {  # $1=имя "$ARM/$seg", $2=vid, $3=aud, $4=dur, $5=audio_on(true|false), $6=vote, $7=floor, $8=regex
  local name="$1" vid="$2" aud="$3" dur="$4" audon="$5" vote="$6" floor="$7" rx="$8"
  local MIN_NEW
  MIN_NEW=$(awk "BEGIN{d=$dur+0; m=int(d*$FPS*0.3); if (m<20) m=20; print m}")  # нижняя граница покрытия 30 %
  echo
  echo "=== ЭТАП $name vote=$vote audio=$audon dur=${dur}s мин.решений=$MIN_NEW $(date +%H:%M:%S) ==="
  write_override "$vid" "$([ "$audon" = true ] && echo "$aud" || echo "")" "$audon" "$vote" "$floor"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
  reset_groups
  q "TRUNCATE uavdet.inference, uavdet.decisions;" >/dev/null
  OFF_BEFORE=$(offset)
  "${COMPOSE[@]}" up -d $DOWNSTREAM 2>&1 | tail -1
  sleep 20
  docker exec uavdet-visual-detector env | grep -qE "VOTE_MODE=$vote" \
    && docker exec uavdet-visual-detector env | grep -qE "VOTE_FLOOR=$floor" \
    || { echo "ОТКАЗ P5 vote ($name)"; exit 1; }
  "${COMPOSE[@]}" up -d source-simulator 2>&1 | tail -1
  SRC_ENV=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' uavdet-source-simulator)
  grep -qE "ADAPTER=media_file" <<<"$SRC_ENV" && grep -qE "ENABLE_AUDIO=$audon" <<<"$SRC_ENV" \
    || { echo "ОТКАЗ P5 source ($name)"; exit 1; }
  sleep "$(awk "BEGIN{print int($dur)+20}")"
  NINF=$(q "SELECT count(1) FROM uavdet.inference WHERE modality='video';" | tr -d '[:space:]')
  [ "${NINF:-0}" -gt 0 ] || { echo "ОТКАЗ: пустой видео-стрим на '$name'"; exit 1; }
  P4M=$(q "SELECT DISTINCT model_name FROM uavdet.inference WHERE modality='video';" | tr -d '\r')
  grep -qE "$rx" <<<"$P4M" || { echo "ОТКАЗ P4 ($name): $(tr '\n' ' ' <<<"$P4M")"; exit 1; }
  if [ "$audon" = true ]; then
    NA=$(q "SELECT count(1) FROM uavdet.inference WHERE modality='audio';" | tr -d '[:space:]')
    [ "${NA:-0}" -gt 0 ] || { echo "ОТКАЗ M0-подобный: audio_inf=0 на '$name'"; exit 1; }
  else
    NA=0
  fi
  PREV=$(offset); STABLE=0; DEADLINE=$((SECONDS + $(awk "BEGIN{d=$dur+0; w=int(d)+300; if (w<300) w=300; print w}")))
  while [ $SECONDS -lt $DEADLINE ]; do
    sleep 15
    CUR=$(offset)
    if [ "$CUR" -eq "$PREV" ]; then STABLE=$((STABLE+15)); else STABLE=0; fi
    PREV=$CUR
    [ "$STABLE" -ge 45 ] && break
  done
  [ "$STABLE" -ge 45 ] || { echo "ОТКАЗ дренажа: '$name' тишина ${STABLE}с < 45с (обрезанный срез губит M1–M3)"; exit 1; }
  NEW=$((CUR-OFF_BEFORE))
  [ "$NEW" -ge "$MIN_NEW" ] || { echo "ОТКАЗ дренажа: '$name' дал $NEW решений (<$MIN_NEW)"; exit 1; }
  ARM_FIRST="${ARM_FIRST:-$((OFF_BEFORE + 1))}"; ARM_LAST="$CUR"
  echo "SEGSTAT $name $((OFF_BEFORE+1)):$CUR (новых $NEW, тишина ${STABLE}с)"
  echo "TELEMETRY $name audio_inf=$NA video_inf=$NINF"
  "${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
}

exec > >(tee -a "$LOG") 2>&1
echo "=== it-86-full $(date '+%F %T') (V0/V1/V2 × neg-сегменты holdout-24, loop=false) ==="
write_override dummy "" false off 0.4
# EXIT-trap (симметрично it87, тот же паттерн протестирован stub-прогоном 24.09): при ЛЮБОМ
# выходе после этой точки (mid-run ОТКАЗ P4/P5/стрима/дренажа делает exit 1 и хвост не
# выполняет) гасим app-стек и освобождаем GPU. rm -sf $SERVICES, НЕ down (down задел бы
# общую kafka/postgres). Pre-flight-ОТКАЗы до этой точки — trap не стоит, docker не тронут.
trap '"${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true' EXIT
"${COMPOSE[@]}" build source-simulator visual-detector 2>&1 | tail -2
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it86-full] ждём инфраструктуру (30с)..."; sleep 30
for t in video.raw audio.raw inference decisions; do
  "${COMPOSE_INFRA[@]}" exec -T kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 --create --if-not-exists \
    --topic "$t" --partitions 1 --replication-factor 1 >/dev/null 2>&1 || true
done
run_arm() {  # $1=имя плечи (ровно как в eval), $2..$5 = audio_on, vote, floor, regex
  local arm="$1" audon="$2" vote="$3" floor="$4" rx="$5"
  ARM_FIRST=""; ARM_LAST=""
  local row id vid aud dur
  for row in "${SEGS[@]}"; do
    IFS=$'\t' read -r id vid aud dur <<<"$row"
    stage "$arm / $id" "$vid" "$aud" "$dur" "$audon" "$vote" "$floor" "$rx"
  done
  echo "SCORE_OFF $arm $ARM_FIRST:$ARM_LAST (плечо целиком: ${#SEGS[@]} сегментов)"
}

run_arm "V0 off-audio" false off 0.4 "$REGEX_SINGLE"
run_arm "V1 +audio"    true  off 0.4 "$REGEX_SINGLE"
run_arm "V2 and+audio" true  and 0.40 "$REGEX_AND"
# хвостовая уборка app-стека (GPU) — симметрично it87: явная на успешном пути, EXIT-trap
# выше дублирует на любом выходе. Full down отвергнут (снёс бы общую kafka/postgres).
# Если и trap не отработал (kill -9) — ручная уборка: шаг 2.5 чек-листа роадмапа.
"${COMPOSE[@]}" rm -sf $SERVICES >/dev/null 2>&1 || true
rm -f "$OVR"
echo
echo "=== it-86-full завершён; РАЗБОР (идеома сквозно проверена 24.09; агрегация M1–M3 — по сумме плеч): ==="
echo "  mapfile -t FRES < <(awk '\$1==\"SCORE_OFF\" && \$2 ~ /^V[012]\$/ {print \$2 \" \" \$3 \"=\" \$4}' $LOG)"
echo "  research/.venv/bin/python research/it86_sync_negative_eval.py --out=research/it86_full_summary.txt \"\${FRES[@]}\""
echo "  (--out обязателен: дефолт eval — it86_sync_summary.txt, замороженный манифестный артефакт SANITY it-86)"
