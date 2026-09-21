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

```bash
# it-67: post-hoc-join — какие model_name/model_ver были в живых аблациях it-42…51 (том uavdet-pgdata):
bash research/it67_pg_join.sh   # защищён: требует финальных строк обеих цепочек it-65 и available ≥ 2 ГиБ;
                                # поднимает только postgres, останавливает его trap'ом; вывод → research/it67_pg_join.out
# it-68: живой A/B fusion-конфига (A=per-message/k=0 прежняя поставка, B=watermark/k=5 испытанное) на актуальных весах:
#   (правка configs/pilot.yaml на watermark/k=5 уже внесена 2026-09-21, U1; этапы v5 задают оба
#    конфига env-переопределениями UAVDET_FUSION__*, так что порядок «сначала A/B, потом правка» не обязателен)
bash research/it68_ab_run.sh    # этапы ablation_v5.sh + provenance-заголовок (sha весов, строка реестра);
                                # те же защиты; лог research/it68_ab_run.log; скоринг — score_stages.py --burn-in-s 90
```

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
# пересчёт sandbox-домена и fusion-контура новыми весами (it-66) — одной командой
# (отказывается без артефактов it-65; сверяет sha весов с best.pt; логи research/it66_run.log):
bash research/it66_run.sh [путь.к.новым.весам]   # по умолчанию models/visual/yolov8s-uav.pt
#   ВАЖНО (21.09): экспорт it-65 НЕ делался (красный E5) → в models/ лежат СТАРЫЕ веса;
#   для it-66 обязан явный аргумент: bash research/it66_run.sh \
#     MasterDiploma/train/runs/visual/uav-yolov8s-bg/weights/best.pt
#   (вручную то же самое: yolo_frames_eval.py --weights <new.pt> --out research/yolo_sandbox_frames_new.csv,
#    затем для s в fusion_sim_full stress_sim event_metrics threshold_calibration_v2
#    bootstrap_delta_v2: $s.py --yolo-csv research/yolo_sandbox_frames_new.csv --suffix=-new;
#    значение --suffix — только формой с =; 7 файлов *-new уже зарегистрированы в FILES
#    make_manifest.py — добавлять больше не нужно)
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

Итог 2026-09-21 07:01 МСК: тренировка 30/30 эпох (best ep30 mAP50 0,8406); вердикт **exit=2**
(E1 8,889 % OK / E2 0,9059 OK / E3 0,8830 OK / E4 18/18 OK / **E5 SAHI полёт 84,3 % < 89,5 % —
красный**) → экспорт НЕ делался, `models/visual/yolov8s-uav.pt` — прежние майские веса.
Инцидент: шаг 3 цепочки упал на печати метрик (`mmaud_sahi_full.py`: `bisect_left` без префикса
`bisect.`) при полном CSV — исправлено, шаг перезапущен по протоколу частичных файлов
(conf_sahi совпал 1:1 на всех 5091 строках), манифест перегенерирован, вердикт подтверждён.
Подробности и трактовка — отчёт `iterations/it-65-yolo-retrain-bg.md`, раздел «ИТОГ».

```bash
# экспорт при зелёном вердикте (exit 0; красный exit 2 — НЕ экспортировать):
cd MasterDiploma
cp models/visual/yolov8s-uav.pt models/visual/yolov8s-uav-old-neg0.pt   # backup до перезаписи
sha256sum models/visual/yolov8s-uav-old-neg0.pt   # обязан дать 41f3fd55…a262 (старые майские веса)
./venv/bin/uavtrain export-visual \
  --weights train/runs/visual/uav-yolov8s-bg/weights/best.pt \
  --metrics train/runs/eval/visual-bg-new/metrics.json \
  --dataset "hf-drone-detection + DUT Anti-UAV + COCO-фоны (негативы 1,3 % train / 3,3 % val / 1,8 % test), 1 класс drone, it-65, seed 1337"
# флаг --dataset закрыл дефект реестра (без него registry.csv записал бы корпус it-65 как «hf-drone-detection»);
# md-строка печатается в stdout — переносится в models/README.md вручную, старую строку переименовать в -old-neg0.pt
```

Заметки к экспорту: переименование касается таблицы `models/README.md` (строка 2026-05-12) — в
append-only `models/registry.csv` визуальных строк нет вообще (майский вес регистрировался до появления
реестра), export просто добавит первую визуальную запись.

```bash
# E2/E3 контроль устройством (baseline старой модели — CPU, chain2 — GPU; порог E2 узкий):
# если mAP50 visual-new-dut600 попал в [0,715; 0,725] — пересчитать те же веса на CPU:
cd MasterDiploma && ./venv/bin/python -m uavtrain.cli eval-visual \
  --weights train/runs/visual/uav-yolov8s-bg/weights/best.pt \
  --data train/data/_prepared/visual-dut-test600/data.yaml \
  --imgsz 640 --device cpu --name new-dut600-cpu   # отдельный каталог visual-new-dut600-cpu, вердикт не трогает
```
Условие контроля не наступило (зафиксировано 2026-09-21): mAP50 новой на GPU = 0,9059 ∉ [0,715; 0,725],
отрыв от порога E2 (>0,720) двукратный — каталог `visual-new-dut600-cpu` не создавался; постоянный
warning манифеста «не включены: …visual-new-dut600-cpu/metrics.json» ожидаем и безвреден.

# Живой A/B конфига fusion (it-68): A=per-message/k=0 (прежняя поставка), B=watermark/k=5
# (испытанное it-43), C=watermark/k=0 — бонус-контроль; этапы ablation_v5.sh по 240 с, τ=0,5.
# Provenance (U4): веса/ревизия — в заголовке research/it68_ab_run.log (sha256 + строка реестра);
# интервалы этапов — строки `SCORE_OFF <A|B|C> START:END` того же лога (нумерация строк
# MasterDiploma/data/decisions/decisions.jsonl). Скоринг:
research/.venv/bin/python research/score_stages.py --burn-in-s 90 A=START:END B=... C=...
# медиана задержки U3 — по обкатанной команде sed|grep|sort|awk из it-68 PLANNED (U3).
bash research/it68_ab_run.sh   # защиты: финальные строки цепочек it-65, память ≥2 ГиБ; НЕ трогает веса

## 6b. it-69 — независимые подтверждения перекалибровки гейта (2026-09-21)

- **V1 (внешние данные!):** официальный `https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip`
  (path-style — домен `images.cocodataset.org` даёт TLS-перехват сертификата; проверку отключать нельзя),
  sha256 `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`, CRC-тест zip.
  Выборка: 400 кадров из val2017 по возрастанию имени, байт-хеш-исключение 900 использованных
  (`coco-background/val2017`, совпало 899) → `MasterDiploma/train/data/_prepared/coco-bg-v1/` (вне git).
  Замер: `research/.venv/bin/python research/coco_bg_fp_eval.py --weights <вес> --name it69v1-{old,new}
  --images-dir MasterDiploma/train/data/_prepared/coco-bg-v1 --pattern '*.jpg'` →
  `coco_bg_fp_it69v1-{old,new}.csv` (в манифесте), вердикт-сводка `it69_v1_coco400.txt`.
- **V2:** пересчёт из `mmaud_sahi_full{,_new}.csv` (см. `it69_v2_splithalf.txt`); новых прогонов нет.
- **V3:** `bash research/it69_v3_run.sh` — гейт-предфильтр `yolo_sandbox_frames_new_gate04.csv`
  (max_conf<0,4→0; боевой гейт поставки — 0,25, `pilot.yaml:78`) + пять fusion-скриптов
  `--suffix=-new-gate04`; выходы в манифесте. Результат побайтово = it-66 (гейт в контуре no-op).

## 6c. it-70 — рычаг рецепта: разбавление фоновых негативов (2026-09-21→)

- Прунинг (обратим: исходники не тронуты, полный корпус = prepare-visual it-65):
  `research/.venv/bin/python research/it70_prune_negatives.py --apply` — стратифицированный
  keep-list `random.Random(20260921)` (сид 1337 в dry-run дал структурную корреляцию с
  `_split_groups` — не использовать для реза!), фоны train 720→240 (0,42 %), val 90→30,
  test 90→30; артефакт `research/it70_neg_keep.txt` (sha в отчёте).
- Тренировка (cmd it-65 дословно, кроме имени): из `MasterDiploma/`:
  `./venv/bin/uavtrain train-visual --data train/data/_prepared/visual/data.yaml --base-weights yolov8s.pt --epochs 30 --patience 10 --batch 8 --workers 2 --name uav-yolov8s-bg70`
  (лог `train/runs/visual/it70_train.out`; resume-практика it-65 при обрыве).
- Все замеры E1–E8 одной detached-цепочкой: `bash research/it70_measure_run.sh` (watcher ждёт
  «Results saved»; лог `train/runs/visual/uav-yolov8s-bg70/it70_measure.log`; артефакты с
  метками `it70v1`/`bg70`: eval-каталоги `visual-bg70-{dut,hf}600`,
  `coco_bg_fp_it70v1.csv` (те же 400 независимых фонов V1, каталог `coco-bg-v1`),
  `session_vis_probe_bg70.csv`, `mmaud_sahi_full_bg70.csv`,
  `yolo_sandbox_frames_bg70{_gate025,}.csv`, пять `*-bg70.csv`).
- Вердикт по предрегистрации (b43f782): `research/.venv/bin/python research/it70_verdict.py; echo $?`
  (0 — все зелёные, 1 — есть ЖДЁТ, 2 — красный; без пайпа на exit-код).
- Плечо S (полка HF-моделей; веса `research/shelf_models/*.pt` вне git — скачаны с HF, sha в
  `iterations/it-70-…-PLANNED.md`): скрининг `research/.venv/bin/python research/shelf_screen.py`
  (CPU; инкрементальный `research/shelf_screen_results.csv`, финальный маркер
  «SHELF SCREEN DONE»; первая строка — калибровка боевым майским весом, ожидается
  0,7202/0,8673/0,275 бит-в-бит с канонами) → арбитраж независимым MMAUD:
  `bash research/it70_shelf_mmaud.sh` (топ-3 по dut+hf при FP@0,5 < 27,5 %; SAHI 640/0,2 на CPU,
  `--full1920-csv none`; идемпотентен по наличию выходных CSV) → сводка
  `research/.venv/bin/python research/it70_shelf_report.py` (самотест на канонах: old 0,945 /
  new 0,843 SAHI-полёт). Оговорки (leakage/AGPL/CC-BY-NC) — в PLANNED, раздел «Плечо S».

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
