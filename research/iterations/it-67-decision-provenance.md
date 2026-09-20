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

1. Post-hoc-join по `uavdet-pgdata`: какие `model_name` реально были в живых аблациях it-42…51
   (поднять только `postgres` из `infra/docker-compose.yml`, запрос, остановить).
2. Синхронизация `configs/pilot.yaml` с испытанным протоколом (`audio_temporal_k: 5`,
   `window_release: watermark`, ε/hop, τ) — отдельной итерацией, с прогонным подтверждением,
   после it-65/it-66 (см. пп. 1–3 смежного аудита в отчёте it-66).
3. В тексте диссертации: оговорка, что живые числа относятся к конфигу испытаний, а не к поставляемому
   пилотному конфигу, пока пункт 2 не закрыт.
