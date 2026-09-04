"""Экспорт лучших весов в общую папку `models/` + обновление `models/README.md`.

После обучения:
  - визуальная модель: `runs/visual/<name>/weights/best.pt` -> `models/visual/yolov8s-uav.pt`
    (опц. экспорт в ONNX/TensorRT для инференса на ноутбуке);
  - акустическая модель: `runs/acoustic/<name>/lwcnn.pt` -> `models/acoustic/lwcnn.pt`.

Дополнительно в `models/README.md` дописывается строка таблицы: модель, файл, датасет,
ключевые метрики (из `runs/eval/.../metrics.json`), дата. Это «реестр весов» — сами
файлы весов в git не коммитятся (.gitignore), хранятся вне репозитория.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

from .config import MODELS_ACOUSTIC_DIR, MODELS_DIR, MODELS_VISUAL_DIR

# Машиночитаемый append-only лог экспортов (не трогаем models/README.md — он курируется вручную).
_REGISTRY_CSV = MODELS_DIR / "registry.csv"
_CSV_HEADER = "date,model,file,dataset,metrics\n"


def _append_registry_row(model: str, rel_file: str, dataset: str, metrics: dict[str, float], date: str) -> None:
    metrics_str = "; ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else ""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    new = not _REGISTRY_CSV.exists()
    # csv: поля экранируем кавычками (в dataset/metrics бывают запятые/точки с запятой)
    def q(s: str) -> str:
        return '"' + str(s).replace('"', '""') + '"'
    line = ",".join([q(date), q(model), q(rel_file), q(dataset), q(metrics_str)]) + "\n"
    with _REGISTRY_CSV.open("a", encoding="utf-8") as fh:
        if new:
            fh.write(_CSV_HEADER)
        fh.write(line)
    # и просто печатаем строку в формате md-таблицы — её можно вставить в models/README.md вручную
    print(f"| {model} | `{rel_file}` | {dataset} | {metrics_str or '—'} | {date} |")


def _load_metrics(metrics_json: Path | None) -> dict[str, float]:
    if metrics_json and metrics_json.exists():
        try:
            return {k: float(v) for k, v in json.loads(metrics_json.read_text(encoding="utf-8")).items()}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def export_visual(best_pt: Path, *, dataset: str = "hf-drone-detection (DroneDetectionDataset, Pawełczyk & Wojtyra 2020)", metrics_json: Path | None = None, out_name: str = "yolov8s-uav.pt") -> Path:
    """Скопировать лучшие веса YOLOv8 в models/visual/ и дописать строку в models/README.md."""
    MODELS_VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_VISUAL_DIR / out_name
    shutil.copy2(best_pt, dst)
    _append_registry_row(
        model="YOLOv8s (visual)",
        rel_file=f"visual/{out_name}",
        dataset=dataset,
        metrics=_load_metrics(metrics_json),
        date=_dt.date.today().isoformat(),
    )
    return dst


def export_acoustic(best_pt: Path, *, dataset: str = "DroneAudioDataset (Al-Emadi) yes_drone+bebop_1+membo_1 vs ESC-50; MFCC 40×64, окна 1с/0.5с", metrics_json: Path | None = None, out_name: str = "lwcnn.pt") -> Path:
    """Скопировать лучшие веса CNN в models/acoustic/ и дописать строку в models/README.md."""
    MODELS_ACOUSTIC_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_ACOUSTIC_DIR / out_name
    shutil.copy2(best_pt, dst)
    _append_registry_row(
        model="lightweight CNN (acoustic)",
        rel_file=f"acoustic/{out_name}",
        dataset=dataset,
        metrics=_load_metrics(metrics_json),
        date=_dt.date.today().isoformat(),
    )
    return dst
