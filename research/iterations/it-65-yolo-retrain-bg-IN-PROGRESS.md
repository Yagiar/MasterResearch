# ИТЕРАЦИЯ 65 — Переподготовка YOLO с фоновым корпусом (IN PROGRESS)

- **Дата:** 2026-09-06 — СТАТУС: тренировка запущена, идёт (~15-20 ч).
- **Источник:** it-64 аудит (0,002% негативов в визуальном корпусе); ревью §3.

## Выполнено до запуска тренировки

1. **DUT Anti-UAV скачан автономно** (gdown с Google Drive, БЕЗ капчи — публичные файлы): train 5200 / val 2600 / test 2200 кадров, VOC XML (класс UAV, 10109 боксов, негативов 3).
2. **COCO val2017 фоны**: images.cocodataset.org недоступен (connection reset); через HF `simopippa/reduced_coco_1000_val2017` (parquet) извлечено **900 фоновых изображений** без дрона (train 800 / val 100 / test 100).
3. **Парсер _parse_dut переписан**: явные сплиты из папок {train,val,test}/{img,xml}, размеры из <size> XML, кадры без <object> → негативы. Новый парсер `coco-background` (все изображения — негативы, каждый кадр — своя группа для случайного распределения по сплитам).
4. **Корпус пересобран** (`prepare-visual --datasets dut-anti-uav,hf-drone-detection,coco-background`): train 57366 (негативов 1,3%), val 2690 (3,3%), test 4915 (1,8%) — против 0,002% в старом.

## Запущенная тренировка

- Отвязанный процесс (nohup, PID 1929404, переживает сессию): `train-visual --data _prepared/visual/data.yaml --base-weights yolov8s.pt --epochs 30 --patience 10 --batch 8 --workers 2 --name uav-yolov8s-bg`
- Лог: `/tmp/yolo_bg_train.log`; прогресс: `train/runs/visual/uav-yolov8s-bg/results.csv`
- Первый запуск (batch 16) упал по CUDA OOM (6 ГБ) → batch 8.
- Реальная скорость (замер): 1,1 it/s → **~1,8 ч/эпоха**; полные 30 эпох ≈ 54 ч, с early stopping реально ~25-30 ч. Процесс отвязан — переживает сессию; проверка прогресса: `wc -l train/runs/visual/uav-yolov8s-bg/results.csv` (строки = завершённые эпохи).

## По завершении (продолжение цикла)

1. `eval-visual` нового корпуса (test: 4915 кадров, 1,8% негативов) — сравнить mAP со старой моделью.
2. Офлайн-оценка на MMAUD (`mmaud_sahi_full.py --weights <новые> --out research/mmaud_sahi_full_new.csv`) — сравнить SAHI-recall с 94,5% (it-63).
3. Замер FP на фонах COCO-test (`coco_bg_fp_eval.py --weights <новые> --name new`) и негативной сессии.
4. Если качество не упало, а FP на фонах снизился — экспорт в `models/visual/` и live-прогон.
5. Отчёт it-65 финальный + INDEX + коммит весов (models/ не в git — только метрики).

## Подготовка (2026-09-20, пока тренировка идёт)

- **Базовый замер FP СТАРОЙ модели на COCO-фонах test выполнен** (`coco_bg_fp_eval.py`, CPU, 90 кадров, imgsz 640):

  | порог | кадров с FP (old yolov8s-uav) |
  |---|---|
  | 0.25 | 53 (58,9%) |
  | 0.40 | 30 (33,3%) |
  | 0.50 | 23 (25,6%) |
  | 0.70 | 4 (4,4%) |

  медиана max_conf = 0.284. Числовое подтверждение «всегда-положительного» видео (дефект it-64) — критерий для сравнения с новой моделью. CSV: `research/coco_bg_fp_old-yolov8s-uav.csv`.
- `mmaud_sahi_full.py` параметризован `--weights/--out` (по умолчанию — прежнее поведение).
- **Базовый замер СТАРОЙ модели на новом домене DUT (test)** — детерминированная подвыборка 600 из 2200 DUT-кадров test (`make_dut_test_subset.py`, seed=65, symlink-каталог `visual-dut-test600`), eval-visual на CPU, imgsz 640:

  | модель | Images | Instances | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|---|---|
  | old yolov8s-uav (обучена без DUT/фонов) | 600 | 614 | 0.857 | 0.675 | 0.720 | 0.387 |

  Артефакты: `train/runs/eval/visual-old-dut600`. После завершения тренировки прогнать ту же подвыборку с новыми весами на GPU (`--weights <new best.pt> --device 0 --name new-dut600`) — прямое сопоставление «старая vs новая» на DUT-домене одним протоколом.
- Состояние тренировки на момент подготовки: эпоха 23/30 идёт (~1,6 it/s), завершённые эпохи 21–22: P≈0.81–0.82, R≈0.94, mAP50≈0.93.
