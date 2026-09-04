"""uavtrain — обучающий пайплайн системы обнаружения БПЛА (НИР-2).

Отдельное приложение в монорепо (не сервис): скачка датасетов -> подготовка ->
обучение (YOLOv8 для видео, lightweight CNN для аудио) -> eval (mAP/F1/precision/recall +
confusion matrix) -> export лучших весов в `../models/{visual,acoustic}/`.

Запуск: ноутбук `train/notebooks/pipeline.ipynb` (секции вызывают функции отсюда)
или CLI `python -m uavtrain.cli <команда>` (см. `uavtrain/cli.py`).

Модули:
  - `config`        — пути (корень датасетов, runs, models) и общие константы;
  - `datasets`      — реестр датасетов + download() с проверкой sha256;
  - `prepare_visual`— конвертация в YOLO-формат + train/val/test split + data.yaml;
  - `prepare_audio` — извлечение MFCC/спектрограмм + формирование пар «дрон / не-дрон» + split;
  - `train_visual`  — обучение/дообучение YOLOv8 (Ultralytics API);
  - `train_acoustic`— обучение lightweight CNN (PyTorch);
  - `evaluate`      — метрики на test + графики, отчёт в json+png;
  - `export`        — копирование лучших весов в ../models + запись метаданных в models/README.md.
"""

__version__ = "0.1.0"
