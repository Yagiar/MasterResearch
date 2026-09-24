#!/usr/bin/env bash
# it-85 — корректировочный повтор этапов @0,5 (A и B) на пересобранном образце после
# фикса клампа медиа-часов в mmaud_replay (поправка предрега №2). Срезы SCORE_OFF —
# в лог research/it85_rerun_half_fps.log; исходные дефектные срезы A@0,5/B@0,5 пуска №2
# в eval не передаются.
set -u
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
cd "$ROOT/MasterDiploma"

# получаем stage/write_override/sampler/reset_groups/q из основного runner без main'а
RUNNER="$ROOT/research/it85_sweep_run.sh"
IT85_SOURCE_ONLY=1 source "$RUNNER" || true

exec > >(tee -a "$ROOT/research/it85_rerun_half_fps.log") 2>&1
echo "=== it-85 корректировочный @0,5 повтор $(date '+%F %T') ==="
NIMG=$(ls "$MD/train/data/_prepared/hi-res-bg-v1/strict400" | wc -l)
[ "$NIMG" -eq 400 ] || { echo "ОТКАЗ: знаменатель не 400 ($NIMG)"; exit 1; }
write_override off 0.4 0.5   # оверлей нужен до build (main-раннер удаляет файл на finish)
"${COMPOSE[@]}" build source-simulator visual-detector 2>&1 | tail -2
"${COMPOSE_INFRA[@]}" up -d kafka postgres >/dev/null 2>&1 || true
echo "[it85-rerun] ждём инфраструктуру (30с)..."; sleep 30
create_topics
REGEX_A='^yolov8s-uav$'
REGEX_B='^yolov8s-uav\+uav-yolov8s-bg-best:and@0\.4$'
stage "A@0.5" off 0.4 "$REGEX_A" 0.5
stage "B@0.5" and 0.4 "$REGEX_B" 0.5
rm -f "$OVR"
echo "=== корректировочный повтор завершён ==="
