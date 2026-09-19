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
- Оценка длительности: ~40 мин/эпоха × 30 (early stop раньше) ≈ 15-20 ч.

## По завершении (продолжение цикла)

1. `eval-visual` нового корпуса (test: 4915 кадров, 1,8% негативов) — сравнить mAP со старой моделью.
2. Офлайн-оценка на MMAUD (`mmaud_visual_eval.py` с новыми весами) — сравнить recall с it-56 (18,3% @960).
3. Замер FP на фонах COCO-test (90 негативов) и негативной сессии.
4. Если качество не упало, а FP на фонах снизился — экспорт в `models/visual/` и live-прогон.
5. Отчёт it-65 финальный + INDEX + коммит весов (models/ не в git — только метрики).
