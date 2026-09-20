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
research/.venv/bin/python research/mmaud_sahi_full.py          # ПОЛНЫЙ корпус: SAHI recall 94.5% (полёт) / 100% (статика)
#   (it-65: у скрипта появился флаг --full1920-csv — мерж full@1920 из mmaud_imgsz1920.csv относится
#    к СТАРЫМ весам; при других весах передавать 'none', иначе conf_union смешивает две модели)
# аудио: извлечение каналов из bag + спектральная проверка + AST
research/.venv/bin/python research/mmaud_multichannel.py       # корреляционная проверка каналов (ДЕРЖАТЬ ПЕРВЫМ ШАГОМ — it-62)
research/.venv/bin/python research/mmaud_acoustic_eval.py      # AST на /audio1 против GT-высоты
# live: mmaud_replay + SAHI (override-файл по образцу it-59: env UAVDET_VISUAL_DETECTOR__*)
```

Ожидаемые результаты: SAHI recall 82.7% (full @1920 — 64.1%); live presence recall 100% (857/857);
аудио — корреляция каналов ≈ 0.005 (узел неисправен), AST p_drone ≈ 0.075 везде.

## 6. Переподготовка визуальной модели с фоновым корпусом (it-65)

Корпус: hf-drone-detection + DUT Anti-UAV (gdown, публичные файлы) + COCO-фоны (HF
`simopippa/reduced_coco_1000_val2017`), негативов 1,3% train / 3,3% val / 1,8% test:

```bash
# сплиты DUT из папок {train,val,test}/{img,xml} + фоны; пересборка корпуса:
MasterDiploma/venv/bin/uavtrain prepare-visual --datasets dut-anti-uav,hf-drone-detection,coco-background
# обучение (6 ГБ GPU: batch 8; resume после обрыва — last.pt того же прогона):
MasterDiploma/venv/bin/uavtrain train-visual --data train/data/_prepared/visual/data.yaml \
  --base-weights yolov8s.pt --epochs 30 --patience 10 --batch 8 --workers 2 --name uav-yolov8s-bg
research/.venv/bin/python research/it65_resume_train.py        # resume=True от weights/last.pt
```

Замеры старой и новой модели одним протоколом (цепочка после тренировки — `it65_chain.sh`,
лог в каталоге прогона, НЕ в /tmp):

```bash
# baseline-замеры старой модели (CPU, щадя режим тренировки):
research/.venv/bin/python research/make_dut_test_subset.py                      # DUT-test600 (seed 65)
research/.venv/bin/python research/make_dut_test_subset.py --prefix hf --tag hf-test600
cd MasterDiploma && OMP_NUM_THREADS=4 ./venv/bin/python -m uavtrain.cli eval-visual \
  --weights models/visual/yolov8s-uav.pt --data train/data/_prepared/visual-dut-test600/data.yaml \
  --imgsz 640 --device cpu --name old-dut600        # и аналогично old-hf600
research/.venv/bin/python research/coco_bg_fp_eval.py --weights <pt> --name <метка>   # FP на 90 COCO-фонах test
research/.venv/bin/python research/session_vis_probe.py --weights <pt> --name <метка> # «стоящий дрон» (presence-TP)
# полной цепочки (eval test → COCO-FP → SAHI MMAUD новыми весами) — bash research/it65_chain.sh
# вторая цепочка (eval новыми весами DUT600/HF600 → session probe → перегенерация манифеста):
bash research/it65_chain2.sh
# пересчёт sandbox-домена и fusion-контура новыми весами (it-66; значение --suffix — только формой с =):
research/.venv/bin/python research/yolo_frames_eval.py --weights <new.pt> --out research/yolo_sandbox_frames_new.csv
for s in fusion_sim_full stress_sim event_metrics threshold_calibration_v2 bootstrap_delta_v2; do
  research/.venv/bin/python research/$s.py --yolo-csv research/yolo_sandbox_frames_new.csv --suffix=-new; done
#   (аудит 2026-09-20: у всех пяти скриптов выходы именованы с суффиксом — старых артефактов не касаются;
#    при запуске it-66 добавить 7 файлов *-new в FILES make_manifest.py, иначе они не попадут в манифест)
# вердикт по предрегистрированным критериям E1–E5 (из артефактов цепочек):
research/.venv/bin/python research/it65_verdict.py
# самотест вердикта (сверка функций с baseline'ами на старых канонах, см. отчёт it-65, ходы 29–30):
#   fp_rate(coco_bg_fp_old-yolov8s-uav.csv) == 25,556%; session_share(old) == 18/18;
#   sahi_flight_recall(mmaud_sahi_full.csv) == 94,520%
#   (монитор готовности вердикта без холостых опросов: bash research/it65_ready_monitor_v2.sh —
#    готовность только когда все 7 шаговых строк OK; v1 считал готовность по наличию файлов и
#    мог выдать ложный ВЕРДИКТ-ГОТОВ на частичном CSV при FAIL инкрементального шага))
```
Манифест (`make_manifest.py`) с хода 35 покрывает также кривую тренировки
`train/runs/visual/uav-yolov8s-bg/results.csv` и все `metrics.json` eval-прогонов E2/E3.

## 7. Известные границы воспроизводимости

- `sandboxDataForSimulator/` (клип + wav) не версионируется — подложить из локального архива;
  контроль — media-длина 72.609 с, 144 окна. Уточнение (сверено 2026-09-20): из содержимого на диске
  есть только `negative-session.mp4/.wav` (нужно для `session_vis_probe.py` — шаг 5 цепочки 2),
  основного sandbox-клипа **нет**; поэтому пересчёт it-66 (`yolo_frames_eval.py`) возможен только по
  уже извлечённым кадрам `research/sandbox_frames/full/f_001…073.jpg` (локально лежат, в git тоже не
  входят), а не из исходного видео. На чистой копии репро-пакета этот шаг невоспроизводим без обоих
  наборов файлов — ограничение честно остаётся открытым.
  Целостность локальной копии таких входов аттестуется сводным хэшем в `manifest.json`
  (блок `gitignored_inputs`: `tree_sha256` каталога кадров + sha256 mp4/wav; генерируется
  `make_manifest.py`, детерминирован по именам файлов).
- Живые прогоны детерминированы по контенту (it-46: 96% общих паттернов), но зависят от
  окружения GPU/драйвера — абсолютное совпадение RAW-чисел не гарантировано, NORM-скоринг
  устойчивее.
- Старые прогоны (до it-35) без `media_ts` скорятся только диагностическим `--fit-phase`.
