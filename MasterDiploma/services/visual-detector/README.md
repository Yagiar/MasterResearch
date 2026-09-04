# visual-detector — визуальный детектор БПЛА

Consumer-сервис: топик `video.raw` → декод JPEG → предобработка → **YOLOv8 (Ultralytics)**
→ (опц.) **ByteTrack** → публикация `InferenceMsg` (`modality=video`) в топик `inference`.

Веса берутся из `models/visual/` (монтируется в контейнер как `/models`); путь — `visual_detector.weights_path`. Варианты:
- `yolov8s-uav.pt` — наша модель, обученная пайплайном `train/` (`python -m uavtrain.cli export-visual`);
- `VKR_united_datasets_airplane_birds_drone_11-03-2025.pt` — модель из бакалаврской ВКР (классы `{0:Bird, 1:drone, 2:Airplane}`) — годится как рабочая модель, пока обучается новая;
- `VKR_BPLA_model_10-11-2024.pt` — модель из ВКР (1 класс `{0:drone}`).

Какие id классов модели считать «дроном» — задаётся `visual_detector.drone_class_ids` (список индексов; остальные классы → `non-drone`); если не задано — детектор смотрит на **имя** класса (содержит `drone`/`uav`). Например для `VKR_united` (`Bird=0, drone=1, Airplane=2`) → `drone_class_ids: [1]`.

Если файла весов нет — сервис автоматически грузит предобученные `yolov8n.pt` (COCO) как **заглушку** (суррогат «дрона» — COCO-классы `airplane`/`bird`/`kite`), чтобы сквозной путь работал без модели.

## Место в пайплайне
```
video.raw  -->  [visual-detector]  -->  inference (modality=video)  -->  fusion
```

## Конфиг (`configs/pilot.yaml`)
```yaml
kafka:
  bootstrap_servers: kafka:9092
visual_detector:
  group_id: visual-detector
  weights_path: /models/visual/VKR_united_datasets_airplane_birds_drone_11-03-2025.pt  # нет файла -> fallback на yolov8n.pt (COCO)
  drone_class_ids: [1]                          # для VKR_united (Bird=0,drone=1,Airplane=2); убрать -> по имени класса
  conf_threshold: 0.25
  iou_threshold: 0.45
  imgsz: 640
  device: cpu                                   # "cuda" для GPU-сборки
  tracker_enabled: true
  auto_offset_reset: latest
  metrics_port: 0                               # >0 -> /metrics
```

## Запуск
```bash
make run-pipeline      # в составе MVP-пайплайна (Docker)
make run-visual        # отдельно (Docker)

# локально (нужен поднятый Kafka):
pip install -e libs/common -e services/visual-detector
python -m visual_detector --config configs/pilot.yaml
```

## Структура
| Модуль | Назначение |
|---|---|
| `consumer.py` | `VideoConsumer(KafkaConsumerService)` — оркестрация обработки кадра |
| `decode.py` | `FrameDecoder` — base64 JPEG → numpy BGR |
| `preprocess.py` | `PreprocessChain` — цепочка предобработки (на MVP пустая) |
| `detector.py` | `YoloDetector` — Ultralytics YOLOv8 + маппинг классов в `drone`/`non-drone` |
| `tracker.py` | `ByteTrackTracker` — трекинг (per `source_id`) на `supervision.ByteTrack` |
| `batch.py` | `BatchCollector` — микро-батчинг (заготовка, вне MVP) |
| `factory.py` | `DetectorFactory` — сборка детектора/трекера по конфигу |
| `__main__.py` | точка входа (`python -m visual_detector`) |

## Проверка
```bash
docker compose -f infra/docker-compose.yml exec kafka \
  /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic inference --from-beginning --max-messages 3
```
