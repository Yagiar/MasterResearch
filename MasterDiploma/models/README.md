# models/ — реестр обученных весов

Сюда обучающий пайплайн (`train/`) и/или ручная загрузка с HF Hub кладёт лучшие веса:
- `visual/yolov8s-uav.pt` — визуальный детектор (YOLO, дообучен на UAV-датасете) — используется `visual-detector`;
- `acoustic/samid-drone-detector/` — **AST** (Audio Spectrogram Transformer) — рабочий бэкенд на «грязном» звуке из видео; скачивается с HF: `huggingface-cli download Rashidbm/samid-drone-detector --local-dir models/acoustic/samid-drone-detector` (или `git clone https://huggingface.co/Rashidbm/samid-drone-detector models/acoustic/samid-drone-detector`);
- `acoustic/lwcnn.pt` — lightweight CNN (наш train, lwcnn-архитектура); использовать через `acoustic_detector.arch=lwcnn`;
- `visual/VKR_*.pt` — модели YOLO из бакалаврской ВКР (можно подложить как `weights_path` в конфиге visual-detector — см. ниже).

**Файлы весов в git не коммитятся** (`.gitignore` — `models/visual/*`, `models/acoustic/*`, кроме `.gitkeep`).
Они хранятся вне репозитория и подкладываются в контейнеры детекторов через volume (`../models` → `/models`).
Таблица ниже курируется вручную; `python -m uavtrain.cli export-visual` / `export-acoustic` копирует веса в
`models/<...>`, дописывает строку в машиночитаемый `models/registry.csv` (тоже не коммитится) и **печатает**
готовую md-строку — её можно вставить сюда.

Если файла весов нет — `visual-detector` автоматически грузит предобученные `yolov8n.pt` (COCO) как заглушку —
см. `services/visual-detector/README.md`.

## Обученные веса (текущий пилот)

| Модель | Файл | Датасет / признаки | Метрики | Дата |
|---|---|---|---|---|
| YOLOv8n → UAV (visual) | `visual/yolov8s-uav.pt` | DroneDetectionDataset (Pawełczyk & Wojtyra, IEEE Access 2020; HF-зеркало `pathikg/drone-detection-dataset`), ~54k кадров, 1 класс `drone`, imgsz 480, 20 эпох, дообучение от `yolov8n.pt` | **test**: precision=0.906, recall=0.813, mAP@0.5=0.881, mAP@0.5:0.95=0.430 _(val на epoch 20: P=0.988, R=0.970, mAP@0.5=0.986, mAP@0.5:0.95=0.762)_ | 2026-05-12 |
| lightweight CNN (acoustic, DroneAudioDataset) | `acoustic/lwcnn.pt` | DroneAudioDataset (Al-Emadi): `Binary/yes_drone` + `Multiclass/bebop_1` + `membo_1` (positives) vs ESC-50 (negatives); MFCC 40×64, окна 1 с / шаг 0.5 с; ~20.2k окон, split 0.8/0.1/0.1 по файлу; взвеш. CrossEntropy, выбор по balanced_acc | **test**: accuracy=0.993, precision_macro=0.970, recall_macro=0.992, f1_macro=0.980 (balanced_acc≈0.989; recall: drone≈0.986 / non-drone≈0.992). НА sandbox-аудио (звук дрона из видео): p(drone)≤0.026 — domain shift, см. `scripts/probe_audio.py` | 2026-05-12 |
| lightweight CNN (acoustic, DADS, melspec) | `acoustic/lwcnn-13-05-2026.pt` | DADS (`geronimobasso/drone-audio-detection-samples`, 6 источников дронов + 4 не-дрон), мел-спектр 64×64, окна 0.5 с / шаг 0.25 с, max 10 окон/wav (балансировка); ~328k окон; lwcnn-архитектура | **test**: accuracy=0.995, precision_macro=0.995, recall_macro=0.995, f1_macro=0.995 (DADS-домен). НА sandbox-аудио: p(drone) тот же ≈0 — domain shift не закрыт большим/более разнообразным датасетом | 2026-05-13 |
| **AST (Audio Spectrogram Transformer, samid-drone-detector)** — backend по умолчанию | `acoustic/samid-drone-detector/` (каталог HF-репо: config.json + preprocessor_config.json + model.safetensors) | Rashidbm/samid-drone-detector — AST дообучен на DADS + DroneAudioSet (ahlab-drone-project) с агрессивными аугментациями: codec round-trip, синтетический RIR, random EQ, FilterAugment, Patchout, SpecAugment, Mixup; urban-noise overlay поверх drone-класса. Backbone — `MIT/ast-finetuned-audioset-10-10-0.4593` (предобучен на AudioSet, ~2M реальных звуков) | НА sandbox-аудио (целевой домен — звук дрона из видео): **рабочая** классификация, p(drone) 0.5–0.82 на дрон-сегментах, 0.5–0.95 уверенный no_drone на пустых сегментах; latency ≈80 мс/окно, ~345 МБ весов | 2026-05-13 |

## Веса YOLO из бакалаврской ВКР (для visual-detector «как времянка» / сравнения)

| Файл | Классы (`names`) | как использовать |
|---|---|---|
| `visual/VKR_united_datasets_airplane_birds_drone_11-03-2025.pt` | `{0:Bird, 1:drone, 2:Airplane}` | в `configs/pilot.yaml` → `visual_detector.weights_path` + `drone_class_ids: [1]` |
| `visual/VKR_BPLA_model_10-11-2024.pt` | `{0:drone}` | `weights_path` на него; `drone_class_ids` не нужен (имя класса `drone`) |
| `visual/VKR_AOD_detection_model_yolo12n.pt` | `{0:airplane, 1:helicopter, 2:drone, 3:bird}` | `weights_path` на него; опц. `drone_class_ids: [2]` |

## Область действия живого A/B (it-68, 2026-09-21)

Числа живой A/B-проверки конфигураций fusion (A: per-message/k=0; B: watermark/k=5; C: watermark/k=0)
отнесены к весам `visual/yolov8s-uav.pt` sha `41f3fd55…` (не изменены — вердикт it-65 запретил экспорт)
и ревизии `configs/pilot.yaml` из коммита `0b08e1b` (U4-provenance). Отчёт с вердиктами U1–U4:
`research/iterations/it-68-pilot-config-sync.md`; сырые строки (интервалы SCORE_OFF, sha весов,
строка реестра): `research/it68_ab_summary.txt`.
