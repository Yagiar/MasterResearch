"""Обучение/дообучение визуального детектора (YOLOv8, Ultralytics API).

Базовая стратегия (по рекомендации из НИР-1): дообучение `yolov8s` на едином UAV-датасете
(`prepare_visual.convert()` -> data.yaml). Гиперпараметры — из аргументов / конфига;
обучение запускается на сервере с GPU (RTX 3090). Веса лучшей эпохи и метрики оседают
в `train/runs/visual/<name>/` (Ultralytics) — оттуда их забирает `export.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import RANDOM_SEED, RUNS_DIR


@dataclass
class VisualTrainConfig:
    data_yaml: Path                       # путь к data.yaml из prepare_visual.convert()
    base_weights: str = "yolov8s.pt"      # стартовые веса (предобучены на COCO)
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"                     # "0" — первая GPU; "cpu" — CPU
    patience: int = 20                    # early stopping
    workers: int = 4                      # воркеров DataLoader (меньше — меньше RAM/файловых дескрипторов)
    cache: bool = False                   # кешировать изображения в RAM (False — безопасно для больших датасетов)
    fraction: float = 1.0                 # доля датасета (0<f<=1) — для быстрых прогонов/слабых машин
    name: str = "uav-yolov8s"
    seed: int = RANDOM_SEED


def _drop_broken_caches(data_yaml: Path) -> None:
    """Удалить .cache-файлы Ultralytics, которые не открываются (битые после прерванного запуска).

    Прерывание обучения Ctrl+C может оставить недописанный `labels/<split>.cache` -> при следующем
    запуске `np.load` падает EOFError. Удаляем такие — Ultralytics пересоздаст их сам.
    """
    try:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return
    base = Path(cfg.get("path") or data_yaml.parent)
    for key in ("train", "val", "test"):
        rel = cfg.get(key)
        if not rel:
            continue
        img_dir = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        # Ultralytics кладёт <split>.cache рядом с labels/<split>/ (т.е. labels/<split>.cache)
        cache = img_dir.parent.parent / "labels" / f"{img_dir.name}.cache"
        if not cache.exists():
            # запасной вариант — рядом с самой папкой
            cache = img_dir.with_suffix(".cache")
        if cache.exists():
            try:
                import numpy as np  # noqa: PLC0415

                np.load(str(cache), allow_pickle=True).item()
            except Exception:  # noqa: BLE001 — битый кеш
                print(f"[train-visual] удаляю битый кеш: {cache}")
                cache.unlink(missing_ok=True)


def train(cfg: VisualTrainConfig) -> Path:
    """Запустить обучение YOLOv8; вернуть путь к лучшим весам (best.pt)."""
    from ultralytics import YOLO  # noqa: PLC0415

    _drop_broken_caches(cfg.data_yaml)

    project = RUNS_DIR / "visual"
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg.base_weights)
    model.train(
        data=str(cfg.data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        patience=cfg.patience,
        workers=cfg.workers,
        cache=cfg.cache,
        fraction=cfg.fraction,
        project=str(project),
        name=cfg.name,
        seed=cfg.seed,
        verbose=True,
    )
    best = project / cfg.name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"обучение завершилось, но {best} не найден — проверьте логи Ultralytics")
    return best
