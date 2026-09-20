# ИТЕРАЦИЯ 67 — Provenance решений: jsonl-лог не содержит модели, породившей решение (найдено и исправлено в коде)

- **Дата:** 2026-09-20
- **Источник:** автономный аудит конфигурации и данных перед it-66 (см. `it-66-fusion-new-weights-PLANNED.md`, раздел «Смежный аудит»); цель — проверить, чем именно измерялись живые прогоны it-42…51.
- **Артефакты:** `libs/common/src/uavdet_common/messages.py` (`DecisionMsg.models`), `services/fusion/src/fusion/consumer.py` (`_model_refs`), `services/fusion/tests/test_provenance.py` (3 теста).

## Находка

1. `MasterDiploma/data/decisions/decisions.jsonl` — **175 662 записи** живых прогонов. Ключи записи:
   `schema_ver, msg_id, source_id, ts, ts_window, mode, decision, p_fused, contributions, gating,
   e2e_latency_ms, source_msg_ids` — **идентификатора модели нет ни в одном поле**.
2. Видео-ветка при отсутствующем файле весов не падает, а тихо грузит COCO-заглушку
   (`services/visual-detector/src/visual_detector/detector.py`: `_FALLBACK_WEIGHTS = "yolov8n.pt"`,
   ветка `else` → `log.warning` + `_is_surrogate = True`). Предупреждение живёт только в логе процесса.
3. Следовательно: ни один живой прогон нельзя из одних данных отличить «наша `yolov8s-uav`» от
   «заглушка `yolov8n` (COCO)». Для офлайн-скоринга по GT это дыра в валидности: таблицы it-42…51
   опираются на утверждение о модели, которое их собственный лог не подтверждает.
4. Частичное спасение: в PostgreSQL `uavdet.inference` колонки `model_name`, `model_ver` пишутся,
   а `decisions.source_msg_ids` ссылается на inference-сообщения → **post-hoc-join возможен**, если
   том `uavdet-pgdata` жив (проверено: том есть). Это единственный путь проверить прошлые прогоны;
   сам join отложен до окончания цепочек it-65 (не нагружать машину во время тренировки/замеров).

## Исправление (сделано в коде, schema_ver = 1, поле опциональное)

- `DecisionMsg.models: dict[str, ModelRef]` — пусто у старых записей, заполняется для video/audio-канала
  из входящих `InferenceMsg.model`; `name == "unknown"` не пишется, чтобы не плодить фиктивные значения.
- Заполняет fusion-консьюмер (`InferenceConsumer._model_refs`), sink пишет `model_dump()` → поле
  автоматически попадает в `decisions.jsonl`. **В PostgreSQL-таблицу `decisions` поле не пишется**
  (INSERT перечисляет колонки явно; Liquibase-миграция сознательно не трогалась — jsonl достаточно,
  колоночную версию можно добавить отдельным шагом при необходимости).
- Тесты: `test_provenance.py` — оба канала; видимость `ver == "surrogate-coco"` (главное свойство);
  непись «unknown». Прогон: `./venv/bin/python -m pytest services/fusion/tests libs/common/tests -q` →
  59 + 63 passed, новых падений нет.

## Что остаётся сделать (не выполнено в этом ходе)

1. Post-hoc-join по `uavdet-pgdata`: какие `model_name` реально были в живых аблациях it-42…51.
   Процедура (выполнять только после окончания цепочек it-65/66 и при available ≥ 2 ГиБ):
   `docker compose -f infra/docker-compose.yml up -d postgres` →
   `docker compose -f infra/docker-compose.yml exec postgres psql -U uavdet -d uavdet -c "SELECT
   source_id, modality, model_name, model_ver, count(*), min(ingested_at), max(ingested_at) FROM
   uavdet.inference GROUP BY 1,2,3,4 ORDER BY 6"` → `docker compose -f infra/docker-compose.yml stop postgres`.
   Поднимаем только сервис `postgres` (том `uavdet-pgdata`), остальные сервисы не трогаем;
   остановка — `stop`, не `down` (данные и контейнер сохраняются).
   Исполнимо: `bash research/it67_pg_join.sh` (все три защиты в скрипте: финальные строки обеих
   цепочек, available ≥ 2048 МиБ, `stop postgres` в trap; отказ проверен вживую 2026-09-21 —
   exit=1 до завершения цепочки 1, docker не тронут; вывод → `research/it67_pg_join.out`).
2. Синхронизация `configs/pilot.yaml` — зарегистрирована как **it-68**
   (`it-68-pilot-config-sync-PLANNED.md`, критерии U1–U4; τ остаётся 0,5 по живому it-43).
3. Оговорка в тексте: частичное закрытие 2026-09-20 — числа живых аблаций перенесены в
   `MasterDiploma/reports/runs_log.md` (прогон №4) с явной оговоркой «поставка per-message/k=0 ≠
   испытанный watermark/k=5»; аудит `reports/НИР-2/otchet.md` чист (цитируется сходимость v5
   0,92–0,93 и FP по обоим режимам, привязки 0,968 к поставляемому конфигу нет). Остаётся
   перенести формулировку в главу диссертации при её написании (или закрыть правкой конфита it-68).
