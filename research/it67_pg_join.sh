#!/bin/bash
# it-67: post-hoc-join — какие model_name/model_ver реально писались в uavdet.inference
# в живых аблациях it-42…51 (том uavdet-pgdata; колонок provenance в decisions нет).
# Защиты: запуск только после финальных строк обеих цепочек it-65 и при available >= 2 ГиБ;
# поднимается ТОЛЬКО сервис postgres; остановка — stop (не down), в trap на любой исход.
set -eu
ROOT=/home/otrix/code/GeneralFolderMasterDiploma
MD=$ROOT/MasterDiploma
RUN=$MD/train/runs/visual/uav-yolov8s-bg
OUT=$ROOT/research/it67_pg_join.out

grep -q "цепочка it-65 завершена" "$RUN/it65_chain.log" \
  || { echo "ОТКАЗ: цепочка 1 не завершена — не нагружать машину во время замеров"; exit 1; }
grep -q "цепочка 2 завершена" "$RUN/it65_chain2.log" \
  || { echo "ОТКАЗ: цепочка 2 не завершена"; exit 1; }
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
[ "$avail" -ge 2048 ] || { echo "ОТКАЗ: available ${avail} МиБ < 2048 МиБ"; exit 1; }

cd "$MD"
trap 'docker compose -f infra/docker-compose.yml stop postgres' EXIT
docker compose -f infra/docker-compose.yml up -d postgres

for i in $(seq 1 30); do
  if docker compose -f infra/docker-compose.yml exec -T postgres \
       pg_isready -U uavdet -d uavdet >/dev/null 2>&1; then break; fi
  sleep 2
  [ "$i" = 30 ] && { echo "ОШИБКА: postgres не стал ready за 60 с"; exit 1; }
done

docker compose -f infra/docker-compose.yml exec -T postgres psql -U uavdet -d uavdet -c \
  "SELECT source_id, modality, model_name, model_ver, count(*), min(ingested_at), max(ingested_at)
   FROM uavdet.inference GROUP BY 1,2,3,4 ORDER BY 6" | tee "$OUT"
echo "готово: $OUT (postgres остановлен trap'ом)"
