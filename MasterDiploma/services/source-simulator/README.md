# source-simulator — имитатор источников данных

Сервис-имитатор: читает данные через **сменный адаптер источника**, опционально пропускает
поток через **канал деградации** (`DegradationChannel` — Decorator: рассинхрон Δt, дроп кадров/окон,
шум в аудио, отказ канала, дрожание резкости) и **стримит кадры/аудио-окна в `ingest-gateway` по gRPC**
(контракт — `libs/proto/ingest.proto`). Взаимозаменяем с будущим `sensor-driver` (реальные сенсоры) —
оба реализуют один клиентский gRPC-контракт.

Адаптеры:
- `DatasetReplayAdapter` (`adapter: dataset_replay`) — replay видеофайла покадрово с реальным FPS (только видео);
- `MediaFileAdapter` (`adapter: media_file`) — видео + (опц.) аудиодорожка из отдельного wav, нарезаемая
  на окна `win_ms` с шагом `hop_ms` → **синхронные видео+аудио потоки** для проверки fusion. «Тепличный»
  сценарий пилота: файлы из `sandboxDataForSimulator/` (монтируется в контейнер как `/data/sandbox`);
- `MmaudReplayAdapter` (`adapter: mmaud_replay`) — replay секвенции из датасета **MMAUD** (NTU): реально
  синхронные видео+аудио («полевой» сценарий). Видео — папка кадров (`image/`, имя=ROS-таймстемп) или mp4,
  аудио — wav (берётся один канал из многоканального массива); аннотации MMAUD (`ground_truth/`) — для
  офлайн-оценки решений fusion (`evaluate_fusion_jsonl`). Раскладку под конкретное зеркало MMAUD парсер
  ещё уточняется (см. `adapters/mmaud_replay.py`);
- `VideoFileAdapter`, `SyntheticAdapter` — заготовки (плейлист видео / синтетический поток).

По умолчанию в `configs/pilot.yaml` — `adapter: media_file` на «тепличных» файлах из `sandboxDataForSimulator/`.
Для пилотного fusion на этих синхронных данных в конфиге fusion имеет смысл `fusion.mode: late` (или `hybrid`).

Видео- и аудиопотоки идут параллельно (аудио — в отдельном потоке), если адаптер отдаёт обе модальности.

## Место в пайплайне
```
source-simulator  --gRPC StreamVideo / StreamAudio-->  ingest-gateway  -->  Kafka video.raw / audio.raw
```

## Конфиг (`configs/pilot.yaml`)
```yaml
source:
  source_id: cam-01
  adapter: media_file            # dataset_replay (видео) | media_file (видео+аудио) | mmaud_replay | video_file | synthetic
  enable_audio: true             # если адаптер отдаёт аудио — стримить и его
  dataset_replay:
    video_path: /data/sample/sample.mp4
    loop: true
    fps: 25
    jpeg_quality: 75
  media_file:                    # «тепличный» сценарий: видео+wav дрона из sandboxDataForSimulator/ (-> /data/sandbox)
    video_path: /data/sandbox/sandbox-video-for-simulator.mp4
    audio_path: /data/sandbox/sandbox-audio-for-simulator.wav   # null -> только видео
    loop: true
    fps: 25
    jpeg_quality: 75
  mmaud_replay:                  # «полевой» сценарий: срез MMAUD (см. train/data/mmaud -> /data/mmaud)
    video_path: /data/mmaud/V1/Mavic2/image
    audio_path: null
    annotations_path: null
    loop: true
    fps: 30
    audio_channel: 0
  audio:
    sample_rate: 16000
    win_ms: 1000                 # длина аудио-окна
    hop_ms: 500                  # шаг окна (= период отправки)
  degradation:                   # на пилоте enabled: false; для стресс-прогонов:
    enabled: false
    seed: 1337
    dt_shift_ms: 0               # сдвиг ts модальности (рассинхрон Δt); >0 — отстаёт
    dt_shift_target: audio
    frame_drop_prob: 0.0         # дроп видеокадров
    window_drop_prob: 0.0        # дроп аудио-окон
    audio_noise_snr_db: null     # добавить шум в аудио до этого SNR
    modality_dropout: []         # интервалы [[start_s, end_s], ...] полного отказа модальности
    modality_dropout_target: audio
    confidence_jitter_prob: 0.0  # кратковременная потеря резкости кадра
    confidence_jitter_quality: 25
ingest_grpc:
  host: ingest-gateway
  port: 50051
  max_message_mb: 16
  connect_timeout_s: 30
```

## Запуск
```bash
make run-pipeline      # в составе MVP-пайплайна (Docker)
make run-simulator     # отдельно (Docker)

# локально (нужен поднятый ingest-gateway и сгенерированные gRPC-стабы):
pip install -e libs/common -e libs/proto -e services/source-simulator
make proto-gen
python -m source_simulator --config configs/pilot.yaml
```

## Структура
| Модуль | Назначение |
|---|---|
| `adapters/base.py` | контракт `DataSourceAdapter`, типы событий `FrameItem`/`AudioItem` |
| `adapters/dataset_replay.py` | `DatasetReplayAdapter` — replay видеофайла покадрово (только видео) |
| `adapters/media_file.py` | `MediaFileAdapter` — видео + аудио из wav (sliding window); синхронные модальности |
| `adapters/mmaud_replay.py` | `MmaudReplayAdapter` — replay секвенции MMAUD (видео-кадры + многоканальное аудио); парсер аннотаций уточняется |
| `adapters/video_file.py`, `adapters/synthetic.py` | заготовки адаптеров |
| `degradation/channel.py` | `DegradationChannel` — декоратор адаптера (цепочка стратегий) |
| `degradation/strategies.py` | стратегии: `PassThrough`, `FrameDrop`, `WindowDrop`, `DtShift`, `AudioNoise`, `ModalityDropout`, `ConfidenceJitter` (все реализованы, детерминированы при `seed`) |
| `factory.py` | `AdapterFactory` — выбор адаптера + сборка цепочки деградации по конфигу |
| `grpc_client.py` | `GrpcStreamClient` — gRPC-клиент `SourceStream` (`StreamVideo`/`StreamAudio`) |
| `controller.py` | `SimulatorController` — read → выдержка темпа (FPS / hop) → стрим; видео и аудио параллельно |
| `__main__.py` | точка входа (`python -m source_simulator`) |

## Данные для имитатора
- **«Тепличный» (по умолчанию)** — `adapter: media_file`, файлы из `sandboxDataForSimulator/` (видео + wav дрона; монтируется в контейнер как `/data/sandbox`). См. `sandboxDataForSimulator/README.md`.
- **Видео-только** — `adapter: dataset_replay`, `data/sample/sample.mp4` (монтируется как `/data/sample`).
- **«Полевой»** — `adapter: mmaud_replay`, срез датасета MMAUD в `train/data/mmaud/` (раскомментировать монтирование `/data/mmaud` в `infra/docker-compose.app.yml`).

Чтобы аудиопоток реально обрабатывался — поднять `acoustic-detector` (профиль `audio`):
`docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml --profile audio up acoustic-detector`.
Для пилотного fusion на синхронных видео+аудио в конфиге поставить `fusion.mode: late` (или `hybrid`).
