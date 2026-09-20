#!/usr/bin/env python3
"""Генератор манифеста воспроизводимости (it-48, ревью §11): research/manifest.json.

Фиксирует: sha256 весов моделей и GT/результатных CSV, ревизию HF-модели samid-drone-detector,
версии ключевых библиотек. Перегенерируется командой
  research/.venv/bin/python research/make_manifest.py
Детерминирован: одинаковые файлы → одинаковый manifest.json (кроме блока versions/текущее время).
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"

FILES = {
    # --- веса моделей, использованные в опубликованных таблицах ---
    "models/visual/yolov8s-uav.pt": MD / "models/visual/yolov8s-uav.pt",
    "models/acoustic/lwcnn.pt": MD / "models/acoustic/lwcnn.pt",
    "models/acoustic/samid-drone-detector/model.safetensors": MD / "models/acoustic/samid-drone-detector/model.safetensors",
    "models/acoustic/samid-drone-detector/config.json": MD / "models/acoustic/samid-drone-detector/config.json",
    "models/acoustic/samid-drone-detector/preprocessor_config.json": MD / "models/acoustic/samid-drone-detector/preprocessor_config.json",
    # --- ground truth и источники чисел ---
    "research/gt_sandbox_video.csv": ROOT / "research/gt_sandbox_video.csv",
    "research/ast_windows.csv": ROOT / "research/ast_windows.csv",
    "research/yolo_sandbox_frames.csv": ROOT / "research/yolo_sandbox_frames.csv",
    # --- результатные таблицы ---
    "research/fusion_sim_results.csv": ROOT / "research/fusion_sim_results.csv",
    "research/stress_sim_results.csv": ROOT / "research/stress_sim_results.csv",
    "research/bootstrap_pairs.csv": ROOT / "research/bootstrap_pairs.csv",
    "research/delta_sweep.csv": ROOT / "research/delta_sweep.csv",
    "research/event_metrics.csv": ROOT / "research/event_metrics.csv",
    "research/confuser_eval.csv": ROOT / "research/confuser_eval.csv",
    "research/confuser_eval_all.csv": ROOT / "research/confuser_eval_all.csv",
    "research/confuser_eval_summary.csv": ROOT / "research/confuser_eval_summary.csv",
    # --- независимая валидация MMAUD (it-56..60) ---
    "research/mmaud_visual_eval.csv": ROOT / "research/mmaud_visual_eval.csv",
    "research/mmaud_imgsz1920.csv": ROOT / "research/mmaud_imgsz1920.csv",
    "research/mmaud_sahi_eval.csv": ROOT / "research/mmaud_sahi_eval.csv",
    "research/mmaud_sahi_motion.csv": ROOT / "research/mmaud_sahi_motion.csv",
    "research/mmaud_acoustic_eval.csv": ROOT / "research/mmaud_acoustic_eval.csv",
    "research/gt_negative_session.csv": ROOT / "research/gt_negative_session.csv",
    "research/airborne_policy_sim.csv": ROOT / "research/airborne_policy_sim.csv",
    "research/threshold_calibration_v2.csv": ROOT / "research/threshold_calibration_v2.csv",
    # --- it-65: базовые замеры старой модели и результаты новой (по мере появления) ---
    "research/coco_bg_fp_old-yolov8s-uav.csv": ROOT / "research/coco_bg_fp_old-yolov8s-uav.csv",
    "research/session_vis_probe_old-yolov8s-uav.csv": ROOT / "research/session_vis_probe_old-yolov8s-uav.csv",
    "research/session_vis_probe_new-yolov8s-bg.csv": ROOT / "research/session_vis_probe_new-yolov8s-bg.csv",
    "research/coco_bg_fp_new-yolov8s-bg.csv": ROOT / "research/coco_bg_fp_new-yolov8s-bg.csv",
    "research/mmaud_sahi_full.csv": ROOT / "research/mmaud_sahi_full.csv",
    "research/mmaud_sahi_full_new.csv": ROOT / "research/mmaud_sahi_full_new.csv",
    # --- it-65: источники чисел E2/E3 и кривая тренировки (метрики eval-прогонов) ---
    "train/runs/visual/uav-yolov8s-bg/results.csv": MD / "train/runs/visual/uav-yolov8s-bg/results.csv",
    "train/runs/eval/visual-old-dut600/metrics.json": MD / "train/runs/eval/visual-old-dut600/metrics.json",
    "train/runs/eval/visual-old-hf600/metrics.json": MD / "train/runs/eval/visual-old-hf600/metrics.json",
    "train/runs/eval/visual-bg-new/metrics.json": MD / "train/runs/eval/visual-bg-new/metrics.json",
    "train/runs/eval/visual-new-dut600/metrics.json": MD / "train/runs/eval/visual-new-dut600/metrics.json",
    "train/runs/eval/visual-new-hf600/metrics.json": MD / "train/runs/eval/visual-new-hf600/metrics.json",
}

HF_MODELS = {
    "Rashidbm/samid-drone-detector": "акустический AST-детектор (research/it-05, 15, 18)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_revisions() -> dict[str, str]:
    out = {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for repo in HF_MODELS:
            out[repo] = api.model_info(repo).sha or "unknown"
    except Exception as exc:  # noqa: BLE001 - офлайн: ревизии фиксируются вручную
        print(f"предупреждение: HF недоступен ({type(exc).__name__}); подставьте ревизии вручную", file=sys.stderr)
        out[repo] = "unknown"
    return out


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "librosa", "ultralytics", "numpy", "huggingface_hub"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "not-installed"
    return out


def main() -> int:
    manifest: dict = {
        "schema": "uavdet-repro-manifest/1",
        "hf_revisions": hf_revisions(),
        "versions": versions(),
        "files": {},
    }
    missing = []
    for key, path in FILES.items():
        if path.exists():
            manifest["files"][key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        else:
            missing.append(key)
    out = ROOT / "research/manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {out} ({len(manifest['files'])} файлов)")
    if missing:
        print("отсутствуют (не включены):", ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
