# ИТЕРАЦИЯ 01 — Веб-ресёрч: что может оптимизировать и развить диплом

- **Дата:** 2026-09-04
- **Цель:** по статьям/форумам найти конкретные, применимые улучшения для НИР-2/диссертации (видео YOLOv8, акустика, fusion, датасеты).
- **Метод:** веб-поиск (3 запроса-направления), перекрёстная сверка с аудитом от 2026-09-04 (`Глубокая обратная связь — критический аудит`).

## 1. Видеоветка: малые объекты (дрон далеко/мелко)

Находки (2025–2026 литература):

| Приём | Эффект (по источникам) | Стоимость внедрения |
|---|---|---|
| **P2-голова** (детекция со stride 4) | ≈ **+6% mAP** на UAV-сценах ([MDPI Appl. Sci., систематическая оценка YOLOv8](https://www.mdpi.com/2076-3417/16/7/3559); [LPAE-YOLOv8, Sci. Reports](https://www.nature.com/articles/s41598-025-28741-9); [arXiv:2507.12727](https://arxiv.org/html/2507.12727v1)) | переобучение, +латентность |
| **imgsz 640→1280** | ≈ **+25%** ([тот же MDPI]; консенсус [форума Ultralytics по VisDrone](https://community.ultralytics.com/t/standard-epochs-and-imgsz-for-training-yolo11-yolov12-on-visdrone-dataset/1614): 1280 лучший, 800–960 компромисс) | переобучение, память |
| **SAHI** (нарезка кадров на инференсе) | «significantly improves detection of small, distant drones» ([IEEE, Khorsand 2025](https://ieeexplore.ieee.org/document/10930186/)); **без переобучения** ([официальный гайд](https://docs.ultralytics.com/guides/sahi-tiled-inference)) | только инференс; ↑латентность |

**Вывод для диплома:** наш тест mAP@0.5:0.95=0.43 — не потолок. План: (а) переобучить на `imgsz=640` (было 480 — прямая ошибка для мелких дронов), (б) замерить SAHI на инференсе как отдельную строку ablation (латентность vs mAP — красивая кривая для главы «Эксперименты»), (в) P2-голова — как этап диссертации.

## 2. Акустика: реалистичные негативы (закрытие P0.3 аудита)

- Подтверждено литературой: негативы ESC-50 **не содержат акустических конфьюзеров** (вертолёт, другие пропеллерные ЛА) — наша «легкая» постановка типична для работ на DroneAudioDataset ([репо датасета](https://github.com/saraalemadi/DroneAudioDataset) — «propeller noise indoor + ESC-50»).
- Источники «тяжёлых» негативов: классы helicopter/aircraft/propeller **AudioSet**; [DADS](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples) (уже используется); [Kaggle: drone-audio-dataset (Anagnost)](https://www.kaggle.com/datasets/aranagnost/drone-audio-dataset) — «drones vs everyday background noise» + классификация по типу мотора.
- Практика пассивной акустики: полосовой фильтр **200 Гц–10 кГц**, MEMS-массивы ([Acta Acustica 2026](https://acta-acustica.edpsciences.org/articles/aacus/full_html/2026/01/aacus250134/aacus250134.html)); физический предел дальности акустики 300–500 м ([Robin Radar](https://www.robinradar.com/blog/acoustic-sensors-drone-detection)) — полезные числа для главы «ограничения».

**Вывод:** retrain lwcnn/AST-eval с конфьюзерами — обязательный эксперимент диссертации; ожидаемо F1 упадёт с 0.99 до реалистичных значений — это и есть честный результат.

## 3. Fusion: литературное обоснование entropy/consensus-gating

Наш аудит рекомендовал «gating по энтропии выхода детектора» — это **прямо соответствует современному направлению**:

- [Adaptive audiovisual fusion по prediction confidence (MDPI Appl. Sci. 16(10):5113)](https://www.mdpi.com/2076-3417/16/10/5113) — **параметр-фри** стратегия: вес модальности = f(уверенность предсказания) по каждому образцу. Прямой аналог того, что нужно нам (вес аудио падает, когда p_a неуверенно).
- [Belief Entropy-Based Evidential Fusion (MDPI Entropy 28(3):343, 2026)](https://www.mdpi.com/1099-4300/28/3/343) — гейтинг по энтропии убеждений как альтернатива Шеннону.
- [Confidence-Aware Gated Multimodal Fusion (MDPI Sensors 26(8):2454)](https://www.mdpi.com/1424-8220/26/8/2454) — лог-уверенность как attention-bias.
- Семейство evidential (Dempster–Shafer) fusion: [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253523004293).

**Вывод:** формула `w_i ∝ exp(−λ·H(p_i))` (энтропия бинарного выхода канала) — литературно подкреплённая доработка; можно цитировать MDPI 5113 как ближайшего родственника и позиционировать наш вариант (с окном выравнивания и Δ-правилом) как потоковую модификацию.

## 4. Датасет для согласованных модальностей (P0.2 аудита)

- **MMAUD** (NTU ARIS): [arXiv:2402.03706](https://arxiv.org/html/2402.03706v1), [github.com/ntu-aris/MMAUD](https://github.com/ntu-aris/MMAUD) — >1700 с синхронных данных: RGB-стерео + 4-канальное аудио + LiDAR + радар, 6 типов компактных дронов (Mavic2/3, Avata, Phantom4...), задачи: детекция/классификация типа/траектория. Это **готовый датасет с согласием модальностей** — то, чего в sandbox нет (agree=0 во всех наших прогонах).
- Референс-метод на нём: **AV-FDTI** (Audio-Visual Fusion for Drone Threat Identification, Machine Learning with Applications) — fusion image+audio на MMAUD; использовать как baseline для сравнения.

## 5. Приоритизированный бэклог (impact/усилия)

| # | Действие | Закрывает | Усилия |
|---|---|---|---|
| 1 | GT-разметка sandbox-клипа + честные P/R/F1 fusion-политик (офлайн, из jsonl) | P0.1 | часы |
| 2 | Симуляция weighting-политик (w_v=0.7, entropy-gating) на реальных окнах jsonl | P1.5, раздел «Энтропийное взвешивание» диссертации | часы |
| 3 | Переобучение YOLO imgsz=640 (+SAHI-замер) | P1.6, видео-таблица | GPU-часы |
| 4 | Акустика с конфьюзерами (AudioSet/Kaggle) | P0.3 | GPU-часы |
| 5 | Подключение MMAUD (`mmaud_replay`) | P0.2 | дни |
| 6 | P2-голова | диссертация | недели |

## Источники (все проверены существованием на 2026-09-04, ссылки в тексте)
MDPI Appl. Sci. 16(7):3559; Nature Sci. Rep. s41598-025-28741-9; arXiv:2507.12727; arXiv:2402.03706; IEEE 10930186; Ultralytics SAHI guide + community; github.com/saraalemadi/DroneAudioDataset; HuggingFace DADS; Kaggle Anagnost; Acta Acustica aacus250134; MDPI Appl. Sci. 16(10):5113; MDPI Entropy 28(3):343; MDPI Sensors 26(8):2454; Information Fusion S1566253523004293; Robin Radar blog.
