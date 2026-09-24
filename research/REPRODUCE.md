# REPRODUCE — воспроизводимый комплект (it-48, ревью §11)

Комплект фиксирует цепочку «данные → сплит → конфигурация → checkpoint → предсказания → метрика»
для опубликованных таблиц. Полный пересчёт — из корня workspace, окружение `research/.venv`.

## 0. Манифест

`research/manifest.json` — sha256 весов моделей, GT и всех результатных CSV; ревизия HF-модели;
версии библиотек. Перегенерация и сверка:

```bash
research/.venv/bin/python research/make_manifest.py   # перегенерировать
git diff research/manifest.json                        # что изменилось
research/.venv/bin/python research/verify_manifest.py  # сверить sha256 всех 118 файлов (exit 0 = целостно)
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
дают побайтно те же CSV (сид 20260905 в бутстрапе). Аудит 21.09 (после it-75): то же для
`it71_policy_ablation.py`, `it73_contour.py`, `it74_vote_analysis.py`, `it75_contour_vote.py`
и `it74_vote_bootstrap.py` (сид 20260921) — перезапуск против git чисто (DET-CLEAN).

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

## 6d. it-65-union / it-70 Парето / it-71 / it-72 — безобученные доделки (2026-09-21, ноль или GPU-инференс)

```bash
# union-бэклог it65-union-new-weights (GPU-инференс ~8 мин, права на обучающий режим НЕ касается):
research/.venv/bin/python research/mmaud_imgsz1920_eval.py \
  --weights MasterDiploma/train/runs/visual/uav-yolov8s-bg/weights/best.pt \
  --out research/mmaud_imgsz1920_new.csv           # → it65_union_new.out (лог, вне git)
research/.venv/bin/python research/it65_union_analysis.py   # → it65_union_analysis.txt
#   Итог: union new полёт 85,2 % / дальн10-19 70,5 % (+1 п.п. к SAHI), old-контроль 95,5/91,5 →
#   потеря дальних честная, inference-time рычаги исчерпаны.

# Парето-фронт «гейт T → recall/FP» по существующим CSV (ноль инференса):
research/.venv/bin/python research/it70_threshold_tradeoff.py   # → it70_threshold_tradeoff.txt

# it-71: сетка политик fusion 2 веса × 2 гейта × 6 политик (ноль инференса, вход —
#   yolo_sandbox_frames{,_new}.csv + ast_windows.csv + gt_sandbox_video.csv):
research/.venv/bin/python research/it71_policy_ablation.py       # → it71_policy_ablation.{csv,txt}
#   канон-самопроверка в конце лога: old@0,25/D0 F1=0,961 (сверка с it-66). Итог: кандидатов нет.

# it-72: независимая MMAUD-сессия 207 с (кадры+лидар с OneDrive, вне git; ноль инференса —
#   вход mmaud_sahi_full{,_new}.csv + mmaud_acoustic_eval.csv):
research/.venv/bin/python research/it72_independent_e6.py       # → it72_independent_e6.txt
#   Итог: материал непригоден (1 событие, 3 ground-бина, насыщение max-агрегации) — см. отчёт.
```

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

## 6e. it-73 — SAHI-pv как вход контура (2026-09-21, GPU-инференс, вердикт W2 красный)

```bash
# S1: SAHI (640/0,2, инференс-срез 640, conf 0,05, max по ~8 тайлам) по sandbox-кадрам 1920×1080:
research/.venv/bin/python research/it73_sahi_sandbox.py --weights MasterDiploma/models/visual/yolov8s-uav.pt \
  --out research/yolo_sandbox_frames_old_sahi640.csv --device cuda      # и аналогично new (best.pt it-65)
# S2: SAHI-FP на 400 coco-bg-v1 (ДЕГЕНЕРИРУЕТ: все кадры ≤640px ⇒ 1 тайл = кадр ⇒ тождественно V1 it-69;
#     замеры оставлены как зарегистрированная эквивалентность):
research/.venv/bin/python research/it73_coco_sahi_fp.py --weights <.pt> \
  --out research/coco_bg_sahi_fp_it73_<old|new>.csv --device cuda
# контур (каркас it-71, D0 τ=0,5, гейт 0,25; самопроверка = строки it71_policy_ablation.csv):
research/.venv/bin/python research/it73_contour.py
# сводка/вердикт — research/it73_sahi_contour.txt (в манифесте): возврат 5/5 окон, но FP 13>8 → не кандидат.
```

## 6f. it-74 — голосование двух моделей AND/OR (2026-09-21, ноль инференса)

```bash
# чистый join существующих CSV (mmaud_sahi_full{,_new}, coco_bg_fp_it69v1-{old,new},
# yolo_sandbox_frames{,_new} imgsz480, ast_windows); 6 сами-проверок канонов встроены:
research/.venv/bin/python research/it74_vote_analysis.py
# выходы (в манифесте): it74_vote_analysis.txt (сетка+вердикт), it74_vote_grid.csv.
# Итог: AND@0,40 — полёт 94,7 % / FP 7,2 % — первая зелёная пара E5×E1; G4 контур красный
# (R 0,946 < 0,98) → кандидат офлайн-линейки без контурной заявки; ×2-доставка — решение автора.
# дополнение (интервальные запасы, критерии не меняются):
research/.venv/bin/python research/it74_vote_bootstrap.py   # → it74_vote_bootstrap.txt:
# полёт блочный бутстрап (блок 100, сид 20260921) [92,2; 96,9] над 89,5; фоны Уилсон [5,1; 10,2] под 12,8.
```

## 6g. it-75 — контур на pv-рядах ансамбля AND/OR (2026-09-21, ноль инференса)

```bash
research/.venv/bin/python research/it75_contour_vote.py   # → it75_contour_vote.txt (в манифесте)
# ряды {min,max} над каркасом it-71 (гейт 0,25, D0); сами-проверки old/new/AND встроены.
# Итог: OR-ряд — K1–K3 зелёные (R 1,000 / F1 0,965 / FP 8 = набор окон канона); AND (G4 it-74) красен.
# Вывод: ансамбль двухрядный — image=AND@0,4, контур=OR; гипотеза живой проверки (аналог it-68), n=1 материал.
```

## 6h. it-76 — покрытие E-линейки ансамблем (2026-09-21, ноль инференса)

```bash
research/.venv/bin/python research/it76_ensemble_echeck.py   # → it76_ensemble_echeck.txt (в манифесте)
# Ф: FP min-ряда ≥0,4 на 90 старых фонах (coco_bg_fp_{old,new}-yolov8s-*.csv) = 5,6 % (барьер ±8 п.п. с V1 7,2 %);
# С: presence «стоящего» на 18 с negative-session (session_vis_probe_{old,new}.csv) = 18/18;
# матрица E1–E8: пробелы — только E2/E3 (merge боксов) и живой A/B.
```


## 6i. it-77 — ансамбль-NMS mAP на DUT/HF600 (2026-09-21, GPU-инференс без обучения)

```bash
# обязательные флаги окружения (OOM-уроки первого прогона,см. it-77-ensemble-nms-map.md):
# чанки ~150 встроены в скрипт; нужен expandable_segments на 6-GiB GPU:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MasterDiploma/venv/bin/python research/it77_ensemble_map.py
# → it77_ensemble_map.{csv,txt} (в манифесте). Ключи предсказаний — по порядку отдачи stream
# (r.path в 8.4.48 синтетическое imageN.jpg); одиночные проверяются блокатолом против канонов.
# Итог: блокатор зелёный (4/4 в ±0,010); H1 КРАСНЫЙ: merge dut 0,8737 < new 0,9014; hf 0,8798 < 0,8861
# → mAP-ось (E2/E3) для ансамбля закрыта отрицательно; профиль: conf-AND@0,4 + контурный OR, НЕ merge боксов.
```

## 6j. it-78 — p-семейство обобщённого среднего голосования (2026-09-21, ноль инференса)

```bash
research/.venv/bin/python research/it78_gmean_vote.py
# → it78_gmean_vote.{csv,txt} (в манифесте). Материал — канонические CSV it-74 (join 5091 кадр + 400 фонов).
# Скрипт при импорте гоняет _selftest_gmean (гармоника/арифметика/пределы p→±∞); первые строки вывода —
# сами-проверки крайних точек против it-74 (AND@0,40=94,7/7,2; OR@0,35=100,0/49,8).
# Итог: сами-проверки зелёные; D1 КРАСНЫЙ — ни одна из 55 точек (p,τ) не доминирует AND@0,40; D3 не считался
# → вывод it-74 распространяется на всё семейство средних («исчерпано над p-семейством»).
```

## 6k. it-79 — SAHI-FP на 400 HD-фонах Open Images (2026-09-21→22, GPU-инференс без обучения)

```bash
# сбор выборки (CPU, ~10 ч из-за 429-бана Flickr; данные вне git: train/data/_prepared/hi-res-bg-v1):
research/.venv/bin/python research/it79_fetch_bg.py        # resume-безопасно; ACCEPTED=400/400
# два GPU-замера harness'ом it-73 (tile 640/0,2; imgsz 640; device cuda):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MasterDiploma/venv/bin/python \
  research/it73_coco_sahi_fp.py --weights MasterDiploma/models/visual/yolov8s-uav.pt \
  --out research/it79_old.csv --images-dir MasterDiploma/train/data/_prepared/hi-res-bg-v1/images \
  --file-list MasterDiploma/train/data/_prepared/hi-res-bg-v1/filelist.txt --device cuda
#   ...то же с --weights MasterDiploma/train/runs/visual/uav-yolov8s-bg/weights/best.pt --out research/it79_new.csv
cp MasterDiploma/train/data/_prepared/hi-res-bg-v1/{manifest.csv,filelist.txt} research/  # как it79_manifest.csv/it79_filelist.txt
research/.venv/bin/python research/it79_sahi_hd_fp.py      # сами-проверка fp@0.4≡max_conf≥0.40 → X0–X3
# Итог: X0 зелёный (400 кадров, min 768…4032, CC, ¬aerial); X1/X2 КРАСНЫЕ:
# FP(AND@0,40; SAHI-HD)=48,8 % против V1 7,2 % → AND-точка не робастна к HD-режиму (новое ограничение профиля).
```

## 6l. it-80 — однородный HD-профиль ансамбля (2026-09-22, ноль инференса, только закоммиченные CSV)

```bash
research/.venv/bin/python research/it80_hd_profile.py
# → it80_hd_profile.{csv,txt} (в манифесте). Материал целиком из git:
#   R-ряд — conf_sahi из mmaud_sahi_full.csv / _new.csv (it-74; GT-полёт z>1 → n=5018),
#   FP-ряд — max_conf из it79_old.csv / it79_new.csv (400 HD-фонов OI из §6k).
# Сами-проверки exact (h0-блокатор): old@0,5 R=94,5; new@0,5 R=84,3; FP(AND@0,40)=48,8; join 5091/400/5018/2351.
# Итог: H1 КРАСНЫЙ — первое FP(AND;HD)≤12,8 % при τ=0,70 (9,8 %), но R(AND@0,70)=25,3 % ≪ 85 %;
# H2 КРАСНЫЙ — там же ΔR=−6,4 п.п. (AND хуже лучшего одиночного). Вывод: пара 94,7/7,2 (it-74) —
# смешанный режим (R из SAHI-HD, FP из 640-px COCO); однородной HD-точки в сетке τ≤0,70 нет.
```

## 6m. Перепрогон-аттестация цикла ансамбля (2026-09-22, ноль инференса)

```bash
# все детерминированные CPU-анализаторы §6f–6j/6l прогнаны заново из закоммиченных входов:
for s in it74_vote_analysis it74_vote_bootstrap it75_contour_vote it76_ensemble_echeck \
         it78_gmean_vote it79_sahi_hd_fp it80_hd_profile; do
  research/.venv/bin/python research/$s.py || echo "FAIL $s"
done
research/.venv/bin/python research/verify_manifest.py   # 111/111 sha256 ok
# Итог: 7/7 exit 0; git status чист — артефакты (.csv/.txt) совпали байт-в-байт с закоммиченными.
# it-77 (§6i) в аттестацию не входит: содержит инференс-компоненту (ultralytics), перепрогон не бесплатен.
```

## 6n. Регресс кодового контура MasterDiploma (2026-09-22, после цикла it-64…80)

```bash
cd MasterDiploma && venv/bin/python -m pytest libs/common/tests services
# Итог: 88 passed in 8.66s — цикл (provenance-правок it-67 включительно) тестовый контур не сломал.
# make lint (ruff) локально не проверялся: ruff отсутствует и в venv проекта, и в системе
# (venv/bin/python -m ruff → No module named ruff; which ruff пуст) — фиксация факта, не результат.
# ПЕРЕПРОВЕРКА 24.09 (it-83): та же команда → 98 passed in 10.85s. Расхождение с 88 — +8 тестов,
# добавленных в контур после фиксации §6n (в т.ч. media_ts-файл it-60), и +2 теста it-83;
# прежняя строка 22.09 не переписывается, перемерка фиксируется отдельно.
# ПЕРЕПРОВЕРКА 24.09 (pre-flight it-87): та же команда → 99 passed (--collect-only SUM=99).
# +1 — регресс-тест клампа лага из it-85; счётчик INDEX приведён к 99.
```

## 6o. Живой A/B ансамбля it-81 (2026-09-22, GPU-docker, ~17 мин)

```bash
# Требует: nvidia-container-toolkit, веса models/visual/{yolov8s-uav.pt (41f3fd55…),
# uav-yolov8s-bg-best.pt (7042602a…)}, материал sandbox (source-simulator).
cd MasterDiploma && setsid nohup bash ../research/it81_ab_run.sh &   # A off -> B and@0,4 -> C or@0,4
# Runner сам: пересобирает образ visual-detector (код в образе, не mount!), создаёт топики
# (Kafka без auto-create), на каждом этапе гейтит P5 (env) -> стрим (inference>0 за 25 с) ->
# P4 по ДАННЫМ (DISTINCT model_name = ожидаемый состав) — до 240-с окна; при ОТКАЗе любой
# гейт прерывает всё. Интервалы SCORE_OFF взять из research/it81_ab_run.log.
cd ../research && .venv/bin/python score_stages.py --burn-in-s 90 "A=S1:E1" "B=S2:E2" "C=S3:E3"
# Фиксация (третий пуск, канон): NORM A=B=C: P 0,927 / R 1,000 / F1 0,962; SCORE_OFF
# A 184980:186803, B 186808:188635, C 188636:190463; e2e-медианы 2,516/2,470/2,523 с.
# НЕ воспроизводится байт-в-байт: живой поток + wall-clock окна (в отличие от §6f-6j/6l).
# Предыстория: два первых пуска не засчитаны (пустой Kafka; старый pip-код в образе) —
# operational-заметки в it-81-live-ensemble-ab.md.
```

## 6p. Живой фоновый прогон it-82 (2026-09-22→23, GPU-docker, ~18 мин)

```bash
# Требует: §6k-материал (train/data/_prepared/hi-res-bg-v1 + каталог strict400 из 400 симлинков
# по filelist.txt — вне git), веса как в §6o, nvidia-container-toolkit.
cd MasterDiploma && setsid nohup bash ../research/it82_bg_run.sh &   # A off -> B and@0,4 -> C or@0,4
# Runner — адаптация it81_ab_run.sh: оверлей источника mmaud_replay на /data/bgsrc/strict400
# (fps=2, без аудио, loop), все гейты it-81 (build образа, топики, P5 env, стрим-гейт 25 с,
# P3 model_name по ДАННЫМ). Интервалы SCORE_OFF — из research/it82_bg_run.log (gitignored).
cd ../research && .venv/bin/python it82_bg_fp_eval.py --burn-in-s 60 "A=S1:E1" "B=S2:E2" "C=S3:E3"
# Канон (пуск №2): SCORE_OFF A 190464:190765, B 190766:191068, C 191069:191371;
# FP-кадр: A 0,893 [0,847;0,927] / B(AND@0,4) 0,699 [0,632;0,758] / C(OR@0,4) 1,000 — P0/P1/P2 🟢.
# Единица — отправленный кадр (уникальный media_ts): media_ts режима папки идёт по _DEFAULT_FPS=30
# при темпе 2 к/с (mmaud_replay.py:121) → таймлайн сжат ×15, предрег. слот 0,5 с — union ~15
# картинок (приведён в артефактах как контроль). Пуск №1 не засчитан: heredoc `\$1` в оверлее.
# НЕ воспроизводится байт-в-байт (живой поток); артефакты it82_bg_fp.csv / it82_bg_summary.txt.
```

## 6q. Фикс media_ts + живой sanity it-83 (2026-09-24, GPU-docker, ~9 мин)

```bash
# Фикс: mmaud_replay режим папки — таймбейс media_ts = requested_fps (было _DEFAULT_FPS=30
# при темпе отправки 2 к/с → сжатие таймлайна ×15, находка it-82; §6p). Коммит кода 4dbc99e.
cd MasterDiploma && venv/bin/python -m pytest services/source-simulator -q   # 11 passed
# sanity-харнес (каркас it82_bg_run.sh, этапы A off / B and@0,4 по WAIT=120 с):
cd MasterDiploma && setsid nohup bash ../research/it83_sanity.sh &           # лог research/it83_sanity.log
cd ../research && .venv/bin/python it83_sanity_eval.py --burn-in-s 60 "A=191372:191538" "B=191539:191703"
# Канон: S1 post-burn-in ratio A 1,007 / B 0,993 🟢 (буквальный полный-срез B 1,248 — стартовый
# залп тёп-запуска, зафиксирован и атрибут); S2 кадровая FP B 0,161 < A 0,418 🟢 (не переворот).
# Срез A расширен до дренажной границы (191538) — 3 хвостовых решения этапа A.
# НЕ воспроизводится байт-в-байт (живой поток); артефакт it83_sanity_summary.txt.
```

## 6r. Exact paired replay it-84 (2026-09-24, GPU-docker, ~14 мин)

```bash
# Требует: материал/веса §6p, код с фиксом it-83 (таймбейс папки = requested_fps).
cd MasterDiploma && setsid nohup bash ../research/it84_paired_run.sh &   # A off -> B and@0,4, loop=false
# Отличия от it82_bg_run.sh: строгий конечный прогон 400 кадров (loop=false), конец этапа
# по ДРЕНАЖУ (offset decisions.jsonl не растёт 45 с; максимум 340 с; <200 решений = ОТКАЗ).
cd ../research && .venv/bin/python it84_paired_eval.py --expect-frames 180 "A=S1:E1" "B=S2:E2"
# Канон: SCORE_OFF A 191704:191912, B 191913:192129; C1 🟢 (общих 209, only_B=7 — стартовые
# окна холодного A); P1 🟢: b=53, c=6, McNemar exact p=1,75·10⁻¹⁰; FP(A) 0,330 [0,270;0,396] →
# FP(B AND@0,4) 0,105 [0,071;0,154]; cluster-bootstrap Δ [−0,292;−0,163] (seed 8404, 10k).
# P2 🔴 по букве: FP(A) вне корки [0,5;1,0] — абсолют it-82 инфлирован дофиксной единицей
# (окно fusion 1,2 медиа-с ≈ 36 картинок при сжатии ×15); пофреймовый канон FP — этот §.
# НЕ воспроизводится байт-в-байт (живой поток); артефакты it84_paired.csv/it84_paired_summary.txt.
```

## 6s. Гейт it-89: skyguard третий voter, zero-inference (2026-09-24)

```bash
research/.venv/bin/python research/it89_skyguard_gate.py
# Входы (закоммиченные CSV, ноль инференса): mmaud_sahi_full{,_new,_skyguard-v11}.csv (5091),
#   coco_bg_fp_it69v1-{old,new}.csv + coco_bg_fp_skyguard-v11.csv (400), GT — mmaud ground_truth/*.npy.
# Выход: Q0 зелёный (каноны recall@0,5 94,5/84,3/90,9 ±0,2 п.п.; AND₃⊆AND₂⊆MAJ₃ на множествах);
#   Q1 🔴 (best prec 0,95 @τ=0,5 при introduced_FP=24, порог ≤2 не пройден ни на одной τ);
#   Q2 🔴 (MAJ₃@0,4: ΔR=+4,0 п.п. против ΔFP=+11,5 п.п.). Итог: закрытие отрицательно без GPU.
# Артефакты: research/it89_skyguard_gate.csv, research/it89_skyguard_gate.txt. Воспроизводится детерминированно.
```

## 6t. Throughput saturation sweep it-85 (2026-09-24, GPU-docker, свип ~90 мин + повтор @0,5 ~30 мин)

```bash
# Прогон 12 этапов A(от)/B(AND@0,4) × fps{0,5;1;2;3;4;5} на strict400 (loop=false, аудио off):
bash research/it85_sweep_run.sh            # лог research/it85_sweep_run.log; семплы it85_samples.csv
# Дефект @0,5 (кламп max(1,fps) в mmaud_replay сжимал медиа-часы ×(1/fps)) — поправка предрега №2:
# этапы @0,5 пуска №2 не засчитываются; корректировочный повтор на пересобранном образце:
bash research/it85_rerun_half_fps.sh       # build с фиксом → A@0,5/B@0,5; лог it85_rerun_half_fps.log
# Разбор (срезы 1-based строки decisions.jsonl из SCORE_OFF; @0,5 — из rerun-лога):
research/.venv/bin/python research/it85_sweep_eval.py \
  "A@0.5=194861:195080" "A@1=192502:192722" "A@2=192723:192939" "A@3=192940:193152" \
  "A@4=193153:193362" "A@5=193363:193569" "B@0.5=195081:195303" "B@1=193793:194013" \
  "B@2=194014:194230" "B@3=194231:194443" "B@4=194444:194653" "B@5=194654:194860"
# Канон: sustainable(fps,arm) := drain≤60с ∧ p95_steady≤10с (первые 10с потока решений исключены —
#   тёп-бэклог холодного старта, поправка предрега №3; полный p95 сохранён в Δp95(P2)) ∧ cov≥0,45.
#   Все 12 этапов sustainable 🟢; sustainable_fps(A)=sustainable_fps(B)=5 (потолок сеткой НЕ достигнут,
#   нижняя оценка «≥5»). C0 🟢 (дренаж 206…223≥200); C1 🟢 cov(A@2)=0,540; P0 🟢 (A@0,5 повтором:
#   maxlag 1, p50 0,2с); P2 🟢 sustainable_fps(B)≤(A) и Δp95(полный,B−A)≥0 во всех 6 точках
#   Δ∈[0,00;3,21]с. Гипотеза «×2 снижает потолок» НЕ подтверждена в 0,5…5 к/с; цена ×2 = +70 MiB
#   VRAM (607→677) и GPU-util p95 до 12% против 3%. НЕ воспроизводится байт-в-байт (живой поток).
# Артефакты: research/it85_sweep.csv, research/it85_sweep_summary.txt, research/it85_samples.csv.
# Фикс измерителя: services/source-simulator/.../mmaud_replay.py + регресс test_mmaud_replay_media_ts.py
#   (fps=0,5 → шаг media_ts 2,0с).
```

## 6u. Tile-aware SAHI агрегация it-88 (2026-09-24, GPU-дампы + ноль-инференс разбор)

```bash
# 4 GPU-дампа per-tile конфиденций (строго после it-85; гейт — наличие it85_sweep_summary.txt):
bash research/it88_dump_run.sh            # bg400×{old,new}=400; mmaud stride=10×{old,new}=502 flight
# Ноль-инференс разбор политик (каноны recall/FP из GT, без инференса):
research/.venv/bin/python research/it88_tile_aggregate.py   # → it88_tile_agg.{csv,txt}
# Канон: сами-проверки блокаторы — SC1 MAX-AND@0,40 FP=48,8% (≈it-79 48,8±3) 🟢;
#   SC2 MAX-old recall@0,5=95,8% (≈94,5±3) 🟢. Гейт T1 🔴: ни одна не-MAX политика
#   (CONS2/CONS3/NULL/FULL) не собирает угол FP(AND@0,40)≤12,8 ∧ recall≥85 → exploratory-негатив;
#   граница применимости HD/SAHI не снимается, перекалибровка τ не выполнена (стоп-критерий).
#   CONS3@0,40: FP 19,0/recall 69,9; CONS2@0,40: FP 30,5/recall 86,7; NULL@0,40: FP 1,8/recall 55,6.
# Воспроизводится детерминированно (дампы закоммичены). Артефакты: it88_bg400_{old,new}.csv,
#   it88_mmaud_{old,new}.csv, it88_tile_agg.csv, it88_tile_agg.txt.
```

## 6v. Мультимодальный синхронный негатив it-86 (2026-09-24, живой контур, sanity-обвязка)

```bash
# Три плеча V0/V1/V2 на negative-session (cam+mic, GT «дрона нет»), живой Kafka-пайплайн, GPU:
bash research/it86_sync_negative_run.sh      # SCORE_OFF/TELEMETRY-срезы → it86_sync_run.log
# Разбор (M0-блокатор + дескриптив; срезы — из SCORE_OFF):
research/.venv/bin/python research/it86_sync_negative_eval.py \
  "V0 off-audio=195319:195969" "V1 +audio=195970:197208" "V2 and+audio=197211:198453"
# Канон: M0 🟢 ПРОЙДЕН (audio_inf 631/631; p_a≠null во всех окнах V1/V2 — аудио течёт).
#   V0 FA 651/651=100% (p_v≈0,83 на тепличном фоне), V1 199/1239=16,1%, V2 200/1243=16,1% —
#   знак M2 = подавление вкладом аудио-гейта; СИЛА и headline FA/hour НЕ заявлены: материал
#   17,04 с зациклен loop=true ×~18 (автокорреляция; watermark-релиз сообщения-дрiven теряет
#   100% окон одноразового 17-с прохода — 2 ОТКАЗа дренажа, диагноз: симуляция consumer.py).
#   M3-дескриптив: AND@0,4 не отличим от V1 на этом материале. Full-прогон — после записи
#   автором 10–30 мин синхронного негатива. Артефакты: it86_sync_summary.txt (+ run/eval).
```
