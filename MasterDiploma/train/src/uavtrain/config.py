"""Пути и константы обучающего пайплайна.

Все каталоги — относительно корня `train/` (этот файл лежит в `train/src/uavtrain/`),
кроме `models/`, который общий для репо (`<repo>/models/`).
"""

from __future__ import annotations

from pathlib import Path

# train/src/uavtrain/config.py -> train/  -> <repo>/
_THIS = Path(__file__).resolve()
TRAIN_ROOT = _THIS.parents[2]            # <repo>/train
REPO_ROOT = TRAIN_ROOT.parent            # <repo>

DATA_DIR = TRAIN_ROOT / "data"           # скачанные датасеты (в git не коммитим)
RUNS_DIR = TRAIN_ROOT / "runs"           # выводы обучения / чекпойнты (в git не коммитим)
PREPARED_DIR = DATA_DIR / "_prepared"    # подготовленные данные (YOLO-формат / спектрограммы)

MODELS_DIR = REPO_ROOT / "models"
MODELS_VISUAL_DIR = MODELS_DIR / "visual"
MODELS_ACOUSTIC_DIR = MODELS_DIR / "acoustic"

# единая таксономия меток для итоговых моделей (классы датасетов сводятся к ней)
VISUAL_CLASSES = ["drone"]               # на пилоте — бинарно «дрон / фон»; птица/самолёт — негативы
AUDIO_CLASSES = ["non-drone", "drone"]   # порядок = индекс класса для CNN

DEFAULT_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 1337


def ensure_dirs() -> None:
    """Создать рабочие каталоги, если их нет."""
    for d in (DATA_DIR, RUNS_DIR, PREPARED_DIR, MODELS_VISUAL_DIR, MODELS_ACOUSTIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
