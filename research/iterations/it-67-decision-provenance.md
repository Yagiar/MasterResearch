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

## Post-hoc-join по `uavdet-pgdata` — ВЫПОЛНЕН 2026-09-21 06:16 МСК

`bash research/it67_pg_join.sh` (защиты отработали: обе цепочки it-65 финализированы, available ≥ 2 ГиБ;
поднят только `postgres`, остановлен trap'ом). Вывод — `research/it67_pg_join.out` (ключ манифеста).

Результат (`uavdet.inference`, GROUP BY source_id, modality, model_name, model_ver) — ровно одна группа:

| source_id | modality | model_name | model_ver | count | период (UTC) |
|---|---|---|---|---|---|
| cam-01 | video | yolov8s-uav | 1 | 34 918 | 2026-09-18 22:04:51 → 2026-09-19 02:55:44 |

Что это доказывает и чего не доказывает:

1. **Видеоканал — настоящей моделью, не заглушкой.** В коде эпохи живых прогонов (проверено по
   `git show d24010c:…detector.py`, коммит 19.09 00:20 МСК — до первой строки БД) `model_name` —
   это `Path(weights).stem` только при существовании файла весов; при откате записалось бы
   `yolov8n.pt` + `model_ver="surrogate-coco"`. Строка `yolov8s-uav / 1` могла породиться только
   реально загруженным `models/visual/yolov8s-uav.pt`. Дыра находки №3 закрыта для видеоканала
   в покрытом окне.
2. **Аудиоканал join не подтверждает:** в `uavdet.inference` нет ни одной audio-строки за период —
   post-hoc-проверка модели аудио-ветки невозможна (остаточный пробел; фиксируется честно).
3. **Окно покрытия — часть живого периода:** 34 918 строк за ~4,8 ч против 175 662 записей
   `decisions.jsonl` (несколько сессий) — ранние прогоны в БД не попали.
4. `model_ver` одинаков на всём окне — аблации it-42…51 им не различаются; это и не требовалось:
   различие аблаций было в конфигурации fusion (watermark k), модель одна и та же.

## Что остаётся сделать (не выполнено в этом ходе)

1. ~~Post-hoc-join по `uavdet-pgdata`~~ — выполнено выше (2026-09-21).
2. Синхронизация `configs/pilot.yaml` — зарегистрирована как **it-68**
   (`it-68-pilot-config-sync-PLANNED.md`, критерии U1–U4; τ остаётся 0,5 по живому it-43).
3. Оговорка в тексте: частичное закрытие 2026-09-20 — числа живых аблаций перенесены в
   `MasterDiploma/reports/runs_log.md` (прогон №4) с явной оговоркой «поставка per-message/k=0 ≠
   испытанный watermark/k=5»; аудит `reports/НИР-2/otchet.md` чист (цитируется сходимость v5
   0,92–0,93 и FP по обоим режимам, привязки 0,968 к поставляемому конфигу нет). Остаётся
   перенести формулировку в главу диссертации при её написании (или закрыть правкой конфита it-68).
