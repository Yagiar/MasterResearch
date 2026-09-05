"""Оценка обученных моделей на тестовых сплитах + сохранение отчёта.

Визуальная модель (YOLOv8): mAP@0.5 / mAP@0.5:0.95, precision, recall — через
`YOLO.val()` на test-сплите; дополнительно — confusion matrix (Ultralytics строит сам).
Акустическая модель (CNN): accuracy / precision / recall / F1 (macro) + confusion matrix
по test-набору (sklearn). Результаты — в `train/runs/eval/<name>/`: `metrics.json` + png-графики.

Используется как для финальной оценки лучших весов, так и для ablation (этап 6:
сравнение video-only / audio-only / late / hybrid — слой fusion оценивается отдельно,
по jsonl с решениями из sink, против ground truth датасета).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import RUNS_DIR


@dataclass(frozen=True)
class EvalReport:
    metrics: dict[str, float]
    artifacts_dir: Path


def _save_report(name: str, metrics: dict[str, float]) -> Path:
    out_dir = RUNS_DIR / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    return out_dir


def evaluate_visual(weights: Path, data_yaml: Path, *, imgsz: int = 640, device: str = "0", name: str = "uav-yolov8s") -> EvalReport:
    """Оценить YOLOv8 на test-сплите; вернуть метрики + каталог артефактов."""
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights))
    res = model.val(data=str(data_yaml), split="test", imgsz=imgsz, device=device, verbose=True)
    # res.box.* — агрегированные метрики Ultralytics
    box = getattr(res, "box", None)
    metrics = {
        "map50": float(getattr(box, "map50", 0.0)) if box else 0.0,
        "map50_95": float(getattr(box, "map", 0.0)) if box else 0.0,
        "precision": float(getattr(box, "mp", 0.0)) if box else 0.0,
        "recall": float(getattr(box, "mr", 0.0)) if box else 0.0,
    }
    out_dir = _save_report(f"visual-{name}", metrics)
    return EvalReport(metrics=metrics, artifacts_dir=out_dir)


def evaluate_acoustic(weights: Path, features_npz: Path, *, device: str = "cpu", name: str = "uav-lwcnn",
                      arch: str = "lwcnn") -> EvalReport:
    """Оценить акустический классификатор на test-сплите: accuracy / precision / recall / F1 (macro) + confusion matrix.

    Веса — `state_dict` модели `audio_models.build_audio_model(arch)` (= архитектура acoustic-detector).
    """
    import torch  # noqa: PLC0415
    from sklearn.metrics import (  # noqa: PLC0415
        ConfusionMatrixDisplay,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    from .audio_models import build_audio_model  # noqa: PLC0415
    from .config import AUDIO_CLASSES  # noqa: PLC0415
    from .prepare_audio import load_split, resolve_features_dir  # noqa: PLC0415

    Xte, yte = load_split(features_npz, 2)
    if len(Xte) == 0:
        raise RuntimeError(f"в {features_npz} пустой test-сплит")

    # feature_dim/n_frames из meta.json (для resnet18 не критично, для согласованности)
    fd = nf = 64
    meta_p = resolve_features_dir(features_npz) / "meta.json"
    if meta_p.exists():
        try:
            import json  # noqa: PLC0415
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            fd = int(meta.get("feature_dim") or (meta.get("feature_params") or {}).get("n_mels") or 64)
            nf = int(meta.get("n_frames") or (meta.get("feature_params") or {}).get("n_frames") or 64)
        except Exception:  # noqa: BLE001
            pass

    model = build_audio_model(arch, feature_dim=fd, n_frames=nf, n_classes=len(AUDIO_CLASSES), pretrained=False)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.eval().to(device)
    from .train_acoustic import (
        _predict,  # noqa: PLC0415 — батчевый инференс (целый test на GPU может не влезть)
    )

    pred = _predict(model, Xte, device)

    acc = float((pred == yte).mean())
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    metrics = {"accuracy": acc, "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}
    out_dir = _save_report(f"acoustic-{name}", metrics)

    cm = confusion_matrix(yte, pred, labels=list(range(len(AUDIO_CLASSES))))
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=AUDIO_CLASSES)
        disp.plot(cmap="Blues")
        plt.title(f"acoustic confusion matrix — {name}")
        plt.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close()
    except Exception:  # noqa: BLE001 - график не критичен
        pass
    return EvalReport(metrics=metrics, artifacts_dir=out_dir)


def _fusion_prf(preds: list[tuple[int, int]], weights: list[float] | None = None) -> dict[str, float]:
    """P/R/F1 по парам (предсказание, GT); weights=None — каждое решение весит 1 (RAW, it-45)."""
    if weights is None:
        weights = [1.0] * len(preds)
    tp = sum(w for w, (p, t) in zip(weights, preds, strict=True) if p and t)
    fp = sum(w for w, (p, t) in zip(weights, preds, strict=True) if p and not t)
    fn = sum(w for w, (p, t) in zip(weights, preds, strict=True) if not p and t)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0}


def _fusion_prf(preds: list[tuple[int, int]], weights: list[float] | None = None) -> dict[str, float]:
    """P/R/F1 по парам (предсказание, GT); weights=None — каждое решение весит 1 (RAW, it-45)."""
    if weights is None:
        weights = [1.0] * len(preds)
    tp = sum(w for w, (p, t) in zip(weights, preds, strict=True) if p and t)
    fp = sum(w for w, (p, t) in zip(weights, preds, strict=True) if p and not t)
    fn = sum(w for w, (p, t) in zip(weights, preds, strict=True) if not p and t)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0}


def evaluate_fusion_jsonl(
    decisions_jsonl: Path,
    ground_truth_csv: Path,
    *,
    name: str = "fusion-pilot",
    burn_in_s: float = 0.0,
    clip_len_s: float = 72.609,
    gt_column: str = "airborne",
) -> EvalReport:
    """Оценить слой fusion офлайн: decisions.jsonl (из sink) против GT-разметки по секундам.

    Протокол it-45/46 (ревью §10/§12): выравнивание по событийному времени `media_ts`
    (clip_second = media_ts % clip_len_s; фаза НЕ подбирается по GT); вес решения = 1/(число
    решений в его медиа-секунде) — единица оценки = медиа-секунда (NORM), параллельно отдаётся
    RAW; `burn_in_s` исключает переходный режим старта (бурст/догоняние аудио, it-42/43).
    Решения без media_ts (прогоны до it-35) пропускаются со счётчиком.
    GT: CSV со столбцами second,drone_visible,airborne (research/gt_sandbox_video.csv).
    """
    import csv
    from collections import Counter

    with open(ground_truth_csv, encoding="utf-8") as fh:
        gt = {int(r["second"]): int(r[gt_column]) for r in csv.DictReader(fh)}

    rows: list[dict] = []
    skipped_no_media = 0
    t_min: float | None = None
    with open(decisions_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("media_ts") is None:
                skipped_no_media += 1
                continue
            t_min = d["ts"] if t_min is None else min(t_min, d["ts"])
            rows.append(dict(
                ts=d["ts"],
                media_second=int(d["media_ts"] % clip_len_s),
                mode=d.get("mode", "?"),
                decision=int(bool(d["decision"])),
                gt=gt[int(d["media_ts"] % clip_len_s)],
                joint=d.get("contributions", {}).get("p_v") is not None
                and d.get("contributions", {}).get("p_a") is not None,
            ))
    if not rows:
        raise ValueError(f"в {decisions_jsonl} нет решений с media_ts — прогон до it-35? Скоринг невозможен")
    if burn_in_s > 0:
        rows = [r for r in rows if r["ts"] - t_min >= burn_in_s]
        if not rows:
            raise ValueError(f"после burn-in {burn_in_s} с не осталось решений")

    ms_cnt = Counter(r["media_second"] for r in rows)
    metrics: dict[str, float] = {
        "n_decisions": len(rows),
        "n_skipped_no_media": skipped_no_media,
        "n_media_seconds": len(ms_cnt),
    }
    for mode in sorted({r["mode"] for r in rows}) + ["__all__"]:
        rs = rows if mode == "__all__" else [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        key = "all" if mode == "__all__" else mode
        metrics[f"{key}_n"] = len(rs)
        metrics[f"{key}_joint_share"] = round(sum(1 for r in rs if r["joint"]) / len(rs), 4)
        metrics[f"{key}_raw"] = _fusion_prf([(r["decision"], r["gt"]) for r in rs])
        metrics[f"{key}_norm"] = _fusion_prf(
            [(r["decision"], r["gt"]) for r in rs],
            [1.0 / ms_cnt[r["media_second"]] for r in rs],
        )

    out_dir = _save_report(f"fusion-{name}", metrics)
    return EvalReport(metrics=metrics, artifacts_dir=out_dir)
