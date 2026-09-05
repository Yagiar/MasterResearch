# ИТЕРАЦИЯ 48 — Воспроизводимый комплект: манифест, evaluate_fusion_jsonl в train-контуре, REPRODUCE.md (P2 ревью §11)

- **Дата:** 2026-09-05
- **Источник:** ревью GPT-6-Astra §11 («нужен небольшой самодостаточный комплект: конфигурация, версии зависимостей, идентификаторы весов, манифест сплит, разметка, предсказания и команда пересчёта каждой основной таблицы»; «универсальная функция evaluate_fusion_jsonl в обучающем контуре ещё не реализована»).
- **Артефакты:** `research/make_manifest.py` + `research/manifest.json`, `research/REPRODUCE.md`, `train/src/uavtrain/evaluate.py::evaluate_fusion_jsonl` (реализована; была заглушкой с NotImplementedError), команда `uavtrain eval-fusion-jsonl`, `train/tests/test_evaluate_fusion.py` (+4).

## Состав комплекта

1. **`research/manifest.json`** (генератор `make_manifest.py`, детерминирован): sha256 + размер 14 артефактов — веса (yolov8s-uav.pt, lwcnn.pt, AST safetensors/config/preprocessor), GT (`gt_sandbox_video.csv`), источники чисел (`ast_windows.csv`, `yolo_sandbox_frames.csv`), все результатные таблицы (fusion_sim_results, stress_sim_results, threshold_calibration_v2, bootstrap_pairs, delta_sweep, event_metrics). Ревизия HF-модели `Rashidbm/samid-drone-detector` = `3a12f618dd8aebf180945bf04ebfeef262d65795` (запрашивается через `huggingface_hub`, офлайн — вручную). Версии: python 3.12.3, torch 2.14.0+cu130, transformers 5.16.1, librosa 1.0.0, ultralytics 8.4.139.
2. **`evaluate_fusion_jsonl`** в `uavtrain.evaluate` + CLI `uavtrain eval-fusion-jsonl --decisions … --gt … [--burn-in-s] [--clip-len-s]`: тот же протокол, что `score_stages.py` (it-45/46) — выравнивание по `media_ts % CLIP` без подгонки фазы, NORM-веса (1/решений на медиа-секунду), burn-in, счётчик решений без media_ts; метрики RAW и NORM по модам + `__all__`, доля совместных окон; отчёт в `train/runs/eval/fusion-<name>/metrics.json`.
3. **`research/REPRODUCE.md`**: команды пересчёта КАЖДОЙ основной таблицы (офлайн-детекции → 6 CSV → живой пайплайн и скоринг → train-скоринг), уроки протокола живых прогонов (порядок сброса групп, ребилд образов), границы воспроизводимости (sandbox-медиа вне git; NORM устойчивее RAW при смене GPU).

## Проверки

- `train/tests/test_evaluate_fusion.py`: метрики на синтетике (perfect-классификатор → F1=1.0 RAW/NORM), пропуск решений без media_ts со счётчиком, burn-in, ValueError без media_ts. **67/67** по монорепо (добавлен `train/tests` в pytest testpaths).
- Smoke на реальном накопленном jsonl: `eval-fusion-jsonl --burn-in-s 90` → RAW 0.9196 / NORM 0.9054 (данные нескольких эпох протокола вперемешку — ожидаемо ниже чистого v6d).
- Команды из REPRODUCE.md прогнаны: `score_ablation_v3.sh` на логе v4, `make_manifest.py` — воспроизводятся.

## Остатки (зафиксированы)

- Манифест покрывает артефакты пилотного клипа; для будущих сплитов (MMAUD/сессии) генератор расширяется списком FILES.
- `sandboxDataForSimulator/` остаётся вне git (медиа) — REPRODUCE.md фиксирует контрольные признаки (72.609 с, 144 окна).
