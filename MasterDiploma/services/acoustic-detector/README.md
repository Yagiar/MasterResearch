# acoustic-detector — акустический детектор БПЛА

Consumer-сервис: топик `audio.raw` → декод PCM → извлечение признаков (**MFCC** / **мел-спектрограмма**)
→ **lightweight CNN** («дрон / не-дрон») → публикация `InferenceMsg` (`modality=audio`, с подсказкой
качества `quality.snr_db`) в топик `inference`.

Веса (`lwcnn.pt` — `state_dict` модели `LightweightAudioCNN`) кладёт обучающий пайплайн
(`train/`: `python -m uavtrain.cli train-acoustic` → `export-acoustic`) в `models/acoustic/`.
**Если весов нет** — детектор работает в режиме **энергетического порога** (заглушка: «дрон», если
средняя энергия аудио-окна выше порога) — чтобы сквозной мультимодальный путь работал без обучения.

> Архитектура `LightweightAudioCNN` (`cnn.py`) синхронизирована с `train/src/uavtrain/train_acoustic.py`
> (веса — `state_dict`). Параметры признаков по умолчанию — с `train/src/uavtrain/prepare_audio.py`.

## Место в пайплайне
```
audio.raw  -->  [acoustic-detector]  -->  inference (modality=audio)  -->  fusion (late / hybrid)
```

## Конфиг (`configs/pilot.yaml`)
```yaml
kafka:
  bootstrap_servers: kafka:9092
acoustic_detector:
  group_id: acoustic-detector
  weights_path: /models/acoustic/lwcnn.pt   # нет файла -> режим энергетического порога
  feature: mfcc                              # mfcc | melspec
  sample_rate: 16000
  n_mfcc: 40
  n_mels: 64
  n_frames: 64                               # ширина спектрограммы по времени (паддинг/обрезка)
  device: cpu                                # "cuda" для GPU-сборки
  energy_threshold: 0.01                     # порог энергии для режима-заглушки
  auto_offset_reset: latest
  metrics_port: 0
```

## Запуск
```bash
# в составе пайплайна с аудио (Docker, профиль `audio`):
docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml --profile audio up acoustic-detector

# локально (нужен поднятый Kafka):
pip install -e libs/common -e services/acoustic-detector
python -m acoustic_detector --config configs/pilot.yaml
```
На пилоте источник `dataset_replay` отдаёт только видео — чтобы протестировать акустическую ветку,
нужен источник с аудио (адаптер с `audio_windows()` — реализуется при подготовке fusion-данных) или
ручная публикация `AudioRawMsg` в `audio.raw`.

## Структура
| Модуль | Назначение |
|---|---|
| `consumer.py` | `AudioConsumer(KafkaConsumerService)` — `audio.raw` → `inference` |
| `features.py` | `FeatureExtractor` — PCM → MFCC/мел-спектрограмма (фикс. ширина, z-score) + оценка SNR |
| `cnn.py` | `LightweightAudioCNN` (Conv-BN-ReLU×3 + GAP + Linear) + `LightweightCnnDetector` (CNN или энергетический порог) |
| `factory.py` | `DetectorFactory` — сборка экстрактора/детектора по конфигу |
| `__main__.py` | точка входа (`python -m acoustic_detector`) |

## Проверка
```bash
docker compose -f infra/docker-compose.yml exec kafka \
  /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic inference --from-beginning --max-messages 5 | grep '"modality":"audio"'
```
