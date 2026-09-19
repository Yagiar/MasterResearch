# REPRODUCE — воспроизводимый комплект (it-48, ревью §11)

Комплект фиксирует цепочку «данные → сплит → конфигурация → checkpoint → предсказания → метрика»
для опубликованных таблиц. Полный пересчёт — из корня workspace, окружение `research/.venv`.

## 0. Манифест

`research/manifest.json` — sha256 весов моделей, GT и всех результатных CSV; ревизия HF-модели;
версии библиотек. Перегенерация и сверка:

```bash
research/.venv/bin/python research/make_manifest.py   # перегенерировать
git diff research/manifest.json                        # что изменилось
```

Ключевые артефакты (ревизии — в manifest.json):
- визуальная модель: `MasterDiploma/models/visual/yolov8s-uav.pt` (YOLOv8s, дообучена, it-04);
- акустический AST: `MasterDiploma/models/acoustic/samid-drone-detector/`
  (HF `Rashidbm/samid-drone-detector`, ревизия `3a12f618…`; рецепт окон 1 с / шаг 0.5 с +
  медиана — из карточки модели, it-38);
- акустический CNN: `MasterDiploma/models/acoustic/lwcnn.pt`;
- GT: `research/gt_sandbox_video.csv` (двухслойный visible/airborne, it-02).

## 1. Офлайн-детекции (источники чисел)

```bash
# AST по 144 окнам (1с/0.5с) sandbox-аудио — ЧЕСТНЫЙ p_drone (it-36) → ast_windows.csv
research/.venv/bin/python research/ast_windows_eval.py
# YOLO по секундам клипа (imgsz 480) → yolo_sandbox_frames.csv
research/.venv/bin/python research/yolo_frames_eval.py
```

## 2. Основные таблицы → команды пересчёта

| Таблица (CSV) | Команда | Что получает |
|---|---|---|
| `fusion_sim_results.csv` (it-36) | `research/.venv/bin/python research/fusion_sim_full.py` | политики fusion: RAW-методы + симметричные фильтры k=1 / каузальная медиана-5,7 / центрированная (помечена «ОФЛАЙН») |
| `stress_sim_results.csv` (it-36) | `research/.venv/bin/python research/stress_sim.py` | WindowDrop / шум ОЦЕНОК p_a / outage, каузальная медиана, чистый audio-only |
| `threshold_calibration_v2.csv` (it-40) | `research/.venv/bin/python research/threshold_calibration_v2.py` | свип τ∈[0.05;0.95] по 4 методам, свой порог каждому |
| `bootstrap_pairs.csv` (it-41) | `research/.venv/bin/python research/bootstrap_delta_v2.py` | парные блочные бутстрап-интервалы (блок 5 с, 2000 реплик) + перебор Δ |
| `delta_sweep.csv` (it-41) | та же команда | сетка delta_conf × delta_unconf |
| `event_metrics.csv` (событийные метрики) | `research/.venv/bin/python research/event_metrics.py` | события GT, задержка первого обнаружения, FP/час |

Проверка детерминированности: `fusion_sim_full.py` и `bootstrap_delta_v2.py` при перезапуске
дают побайтно те же CSV (сид 20260905 в бутстрапе).

## 3. Живой пайплайн (GPU/Docker) и скоринг решений

```bash
bash research/ablation_v5.sh 2>&1 | tee /tmp/ablation_v5.log   # 240-с этапы; override-файлы создаются скриптом
# скоринг срезов этапов по media_ts БЕЗ подгонки фазы (протокол it-45/46):
bash research/score_ablation_v3.sh /tmp/ablation_v5.log
# стационарный NORM-скоринг (burn-in 90 с):
research/.venv/bin/python research/score_stages.py --burn-in-s 90 <этапы из лога SCORE_OFF>
# разложение e2e по стадиям (it-47): проба inference параллельно прогону, затем
research/.venv/bin/python research/e2e_decompose.py <decisions.jsonl> research/inference_dump.jsonl
```

Порядок живого прогона (уроки it-42/47): **сначала** остановить сервисы, **потом** сбросить
consumer-группы (`kafka-consumer-groups --delete`), TRUNCATE, затем `up`; после правок кода —
ребилд образов; конфиг источников: 5 fps (gpu-overlay с 25 fps удалён, it-47).

## 4. Скоринг из train-контура (evaluate_fusion_jsonl, it-48)

```bash
MasterDiploma/venv/bin/uavtrain eval-fusion-jsonl \
  --decisions MasterDiploma/data/decisions/decisions.jsonl \
  --gt research/gt_sandbox_video.csv \
  --burn-in-s 90 --name pilot
# → MasterDiploma/train/runs/eval/fusion-pilot/metrics.json
#   (RAW и NORM P/R/F1 по модам + __all__, доля совместных окон, счётчики пропусков)
```

## 5. Независимая валидация на MMAUD V1 Mavic3 (it-56..61)

Материал: `MasterDiploma/train/data/mmaud/Mavic3/` (Folder Data: 5091 PNG 2560×960 + лидарный GT)
и `Mavic3.bag` (ROSBag: аудио /audio1..4, 6 кГц/канал). Не версионируются — загрузка OneDrive (капча).

```bash
# видео: recall присутствия по состояниям (стоит/летит по GT-высоте) + SAHI-сравнение
research/.venv/bin/python research/mmaud_visual_eval.py        # полный корпус @960
research/.venv/bin/python research/mmaud_imgsz1920_eval.py     # полный корпус @1920 + SAHI-проба
research/.venv/bin/python research/mmaud_sahi_eval.py 10       # SAHI 640/0.2 vs full @1920 (510 кадров)
# аудио: извлечение каналов из bag + спектральная проверка + AST
research/.venv/bin/python research/mmaud_multichannel.py       # корреляционная проверка каналов (ДЕРЖАТЬ ПЕРВЫМ ШАГОМ — it-62)
research/.venv/bin/python research/mmaud_acoustic_eval.py      # AST на /audio1 против GT-высоты
# live: mmaud_replay + SAHI (override-файл по образцу it-59: env UAVDET_VISUAL_DETECTOR__*)
```

Ожидаемые результаты: SAHI recall 82.7% (full @1920 — 64.1%); live presence recall 100% (857/857);
аудио — корреляция каналов ≈ 0.005 (узел неисправен), AST p_drone ≈ 0.075 везде.

## 6. Известные границы воспроизводимости

- `sandboxDataForSimulator/` (клип + wav) не версионируется — подложить из локального архива;
  контроль — media-длина 72.609 с, 144 окна.
- Живые прогоны детерминированы по контенту (it-46: 96% общих паттернов), но зависят от
  окружения GPU/драйвера — абсолютное совпадение RAW-чисел не гарантировано, NORM-скоринг
  устойчивее.
- Старые прогоны (до it-35) без `media_ts` скорятся только диагностическим `--fit-phase`.
