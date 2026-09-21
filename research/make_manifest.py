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
    # --- it-66: пересчёт fusion-контура новыми весами (выходы с суффиксом -new) ---
    "research/yolo_sandbox_frames_new.csv": ROOT / "research/yolo_sandbox_frames_new.csv",
    "research/fusion_sim_results-new.csv": ROOT / "research/fusion_sim_results-new.csv",
    "research/stress_sim_results-new.csv": ROOT / "research/stress_sim_results-new.csv",
    "research/bootstrap_pairs-new.csv": ROOT / "research/bootstrap_pairs-new.csv",
    "research/delta_sweep-new.csv": ROOT / "research/delta_sweep-new.csv",
    "research/event_metrics-new.csv": ROOT / "research/event_metrics-new.csv",
    "research/threshold_calibration_v2-new.csv": ROOT / "research/threshold_calibration_v2-new.csv",
    # --- it-67: вывод post-hoc-join по uavdet-pgdata (появится после запуска скрипта) ---
    "research/it67_pg_join.out": ROOT / "research/it67_pg_join.out",
    "research/it68_ab_summary.txt": ROOT / "research/it68_ab_summary.txt",
    "research/it69_thresh_shift_analysis.txt": ROOT / "research/it69_thresh_shift_analysis.txt",
    # --- it-69: V1 расширенный независимый фон + артефакты вердикта ---
    "research/it69_v1_coco400.txt": ROOT / "research/it69_v1_coco400.txt",
    "research/it69_v2_splithalf.txt": ROOT / "research/it69_v2_splithalf.txt",
    "research/coco_bg_fp_it69v1-old.csv": ROOT / "research/coco_bg_fp_it69v1-old.csv",
    "research/coco_bg_fp_it69v1-new.csv": ROOT / "research/coco_bg_fp_it69v1-new.csv",
    # --- it-69 V3: контур new-весов с гейтом 0,4 (выходы с суффиксом -new-gate04) ---
    "research/yolo_sandbox_frames_new_gate04.csv": ROOT / "research/yolo_sandbox_frames_new_gate04.csv",
    "research/fusion_sim_results-new-gate04.csv": ROOT / "research/fusion_sim_results-new-gate04.csv",
    "research/stress_sim_results-new-gate04.csv": ROOT / "research/stress_sim_results-new-gate04.csv",
    "research/bootstrap_pairs-new-gate04.csv": ROOT / "research/bootstrap_pairs-new-gate04.csv",
    "research/delta_sweep-new-gate04.csv": ROOT / "research/delta_sweep-new-gate04.csv",
    "research/event_metrics-new-gate04.csv": ROOT / "research/event_metrics-new-gate04.csv",
    "research/threshold_calibration_v2-new-gate04.csv": ROOT / "research/threshold_calibration_v2-new-gate04.csv",
    # --- it-70: прунинг негативов + плечо S (полка HF-моделей; арбитр MMAUD — по мере появления) ---
    "research/it70_neg_keep.txt": ROOT / "research/it70_neg_keep.txt",
    # тренировочное плечо (приостановлено; артефакты появятся после возобновления)
    "research/coco_bg_fp_it70v1.csv": ROOT / "research/coco_bg_fp_it70v1.csv",
    "research/session_vis_probe_bg70.csv": ROOT / "research/session_vis_probe_bg70.csv",
    "research/mmaud_sahi_full_bg70.csv": ROOT / "research/mmaud_sahi_full_bg70.csv",
    "research/yolo_sandbox_frames_bg70.csv": ROOT / "research/yolo_sandbox_frames_bg70.csv",
    "research/yolo_sandbox_frames_bg70_gate025.csv": ROOT / "research/yolo_sandbox_frames_bg70_gate025.csv",
    "research/fusion_sim_results-bg70.csv": ROOT / "research/fusion_sim_results-bg70.csv",
    "research/stress_sim_results-bg70.csv": ROOT / "research/stress_sim_results-bg70.csv",
    "research/bootstrap_pairs-bg70.csv": ROOT / "research/bootstrap_pairs-bg70.csv",
    "research/delta_sweep-bg70.csv": ROOT / "research/delta_sweep-bg70.csv",
    "research/event_metrics-bg70.csv": ROOT / "research/event_metrics-bg70.csv",
    "research/threshold_calibration_v2-bg70.csv": ROOT / "research/threshold_calibration_v2-bg70.csv",
    "train/runs/eval/visual-bg70-dut600/metrics.json": MD / "train/runs/eval/visual-bg70-dut600/metrics.json",
    "train/runs/eval/visual-bg70-hf600/metrics.json": MD / "train/runs/eval/visual-bg70-hf600/metrics.json",
    "train/runs/visual/uav-yolov8s-bg70/results.csv": MD / "train/runs/visual/uav-yolov8s-bg70/results.csv",
    # плечо S: полка HF-моделей
    "research/shelf_screen_results.csv": ROOT / "research/shelf_screen_results.csv",
    "research/it70_recall_by_distance.txt": ROOT / "research/it70_recall_by_distance.txt",
    "research/it70_threshold_tradeoff.txt": ROOT / "research/it70_threshold_tradeoff.txt",
    "research/it71_policy_ablation.txt": ROOT / "research/it71_policy_ablation.txt",
    "research/it71_policy_ablation.csv": ROOT / "research/it71_policy_ablation.csv",
    "research/it72_independent_e6.txt": ROOT / "research/it72_independent_e6.txt",
    "research/it65_union_analysis.txt": ROOT / "research/it65_union_analysis.txt",
    "research/mmaud_imgsz1920_new.csv": ROOT / "research/mmaud_imgsz1920_new.csv",
    "research/mmaud_sahi_full_doguilmak-v11x.csv": ROOT / "research/mmaud_sahi_full_doguilmak-v11x.csv",
    "research/mmaud_sahi_full_doguilmak-v8x.csv": ROOT / "research/mmaud_sahi_full_doguilmak-v8x.csv",
    "research/mmaud_sahi_full_skyguard-v11.csv": ROOT / "research/mmaud_sahi_full_skyguard-v11.csv",
    "research/mmaud_sahi_full_ruju-v12.csv": ROOT / "research/mmaud_sahi_full_ruju-v12.csv",
    "research/mmaud_sahi_full_danivelikova-v26n.csv": ROOT / "research/mmaud_sahi_full_danivelikova-v26n.csv",
    "research/mmaud_sahi_full_iris-v8s.csv": ROOT / "research/mmaud_sahi_full_iris-v8s.csv",
    "research/mmaud_sahi_full_noah-v8s.csv": ROOT / "research/mmaud_sahi_full_noah-v8s.csv",
    # --- it-65: источники чисел E2/E3 и кривая тренировки (метрики eval-прогонов) ---
    "train/runs/visual/uav-yolov8s-bg/results.csv": MD / "train/runs/visual/uav-yolov8s-bg/results.csv",
    "train/runs/eval/visual-old-dut600/metrics.json": MD / "train/runs/eval/visual-old-dut600/metrics.json",
    "train/runs/eval/visual-old-hf600/metrics.json": MD / "train/runs/eval/visual-old-hf600/metrics.json",
    "train/runs/eval/visual-bg-new/metrics.json": MD / "train/runs/eval/visual-bg-new/metrics.json",
    "train/runs/eval/visual-new-dut600/metrics.json": MD / "train/runs/eval/visual-new-dut600/metrics.json",
    "train/runs/eval/visual-new-dut600-cpu/metrics.json": MD / "train/runs/eval/visual-new-dut600-cpu/metrics.json",
    "train/runs/eval/visual-new-hf600/metrics.json": MD / "train/runs/eval/visual-new-hf600/metrics.json",
}

# плечо S: артефакты полного протокола финалистов (генерируются post-arbitrage)
for _lbl in ("skyguard-v11", "doguilmak-v8x"):
    for _k in (f"research/coco_bg_fp_{_lbl}.csv", f"research/session_vis_probe_{_lbl}.csv",
               f"research/yolo_sandbox_frames_{_lbl}.csv", f"research/yolo_sandbox_frames_{_lbl}_gate025.csv",
               f"research/fusion_sim_results-{_lbl}.csv", f"research/stress_sim_results-{_lbl}.csv",
               f"research/bootstrap_pairs-{_lbl}.csv", f"research/delta_sweep-{_lbl}.csv",
               f"research/event_metrics-{_lbl}.csv", f"research/threshold_calibration_v2-{_lbl}.csv"):
        FILES[_k] = ROOT / _k

HF_MODELS = {
    "Rashidbm/samid-drone-detector": "акустический AST-детектор (research/it-05, 15, 18)",
}

# Входы, которые по политике не попадают в git (см. AGENTS.md / .gitignore). Отдельным блоком:
# их нельзя проверить по репозиторию, но можно — по целостности локальной копии (it-66 упирается
# именно в них: sandbox-клип отсутствует, пересчёт идёт по извлечённым кадрам).
GITIGNORED = {
    "research/sandbox_frames/full/": (ROOT / "research/sandbox_frames/full", "dir"),
    "MasterDiploma/sandboxDataForSimulator/negative-session.mp4": (MD / "sandboxDataForSimulator/negative-session.mp4", "file"),
    "MasterDiploma/sandboxDataForSimulator/negative-session.wav": (MD / "sandboxDataForSimulator/negative-session.wav", "file"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(directory: Path) -> tuple[str, int]:
    """Сводный sha256 каталога: по отсортированным «имя:sha256» файлов (детерминирован)."""
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    agg = hashlib.sha256()
    for p in files:
        agg.update(f"{p.relative_to(directory)}:{sha256(p)}\n".encode())
    return agg.hexdigest(), len(files)


def hf_revisions() -> dict[str, str]:
    out = {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for repo in HF_MODELS:
            out[repo] = api.model_info(repo).sha or "unknown"
    except Exception as exc:  # noqa: BLE001 - офлайн: ревизии фиксируются вручную
        print(f"предупреждение: HF недоступен ({type(exc).__name__}); подставьте ревизии вручную", file=sys.stderr)
        for repo in HF_MODELS:
            out.setdefault(repo, "unknown")
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
    manifest["gitignored_inputs"] = {}
    for key, (path, kind) in GITIGNORED.items():
        if kind == "dir" and path.is_dir():
            dig, n = tree_digest(path)
            manifest["gitignored_inputs"][key] = {"tree_sha256": dig, "files": n}
        elif kind == "file" and path.is_file():
            manifest["gitignored_inputs"][key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        else:
            manifest["gitignored_inputs"][key] = {"state": "absent"}
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {out} ({len(manifest['files'])} файлов)")
    if missing:
        print("отсутствуют (не включены):", ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
