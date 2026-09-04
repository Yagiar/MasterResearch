# sandboxDataForSimulator/ — медиа-сэмплы для имитатора («тепличный» сценарий)

Сюда кладутся **видео + аудиодорожка** для запуска пайплайна в «тепличных условиях» (дрон в помещении,
без посторонних шумов) через `source-simulator` с адаптером `media_file`. Сами файлы в git **не коммитятся**
(`.gitignore` — `sandboxDataForSimulator/*`, кроме `.gitkeep` и этого README) — они большие; в репо хранится
только структура папки.

## Что положить
- `sandbox-video-for-simulator.mp4` — видеоклип с дроном (любой формат, который читает OpenCV; напр. H.264 .mp4);
- `sandbox-audio-for-simulator.wav` — аудиодорожка того же события (WAV; любой sample rate / число каналов —
  адаптер ресемплит к целевому SR из `configs/pilot.yaml -> source.audio.sample_rate` и берёт первый канал).

Видео и аудио должны быть **одной длительности и синхронны** (одно событие) — fusion-движок выравнивает
потоки по ±ε, но опора на общий старт.

## Как подключается
`configs/pilot.yaml`:
```yaml
source:
  adapter: media_file
  enable_audio: true
  media_file:
    video_path: /data/sandbox/sandbox-video-for-simulator.mp4
    audio_path: /data/sandbox/sandbox-audio-for-simulator.wav
    loop: true
    fps: 25                # темп воспроизведения (имитация реального FPS)
  audio:
    sample_rate: 16000
    win_ms: 1000           # длина аудио-окна
    hop_ms: 500            # шаг окна (= период отправки)
```
В контейнере `source-simulator` эта папка смонтирована как `/data/sandbox` (см. `infra/docker-compose.app.yml`).

## Запуск пилота на этих данных
```bash
make infra-up && make topics-create && make db-migrate
make run-pipeline        # source-simulator → ingest-gateway → visual-detector → fusion → sink
# + акустический детектор (профиль audio):
docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml --profile audio up acoustic-detector
# + дашборд:
docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml --profile dashboard up gateway   # http://localhost:8080
```
`source-simulator` гонит видео покадрово (25 fps) и параллельно аудио-окна по gRPC в `ingest-gateway`;
дальше — `video.raw`/`audio.raw` → `visual-detector`/`acoustic-detector` → `inference` → `fusion` (режим
`fusion.mode` из конфига: для пилотного fusion поставь `late` или `hybrid`) → `decisions` → `sink` (лог +
`data/decisions/decisions.jsonl` + Prometheus + PostgreSQL `uavdet.decisions`/`uavdet.inference`) + дашборд.

Для стресс-сценариев — включить `source.degradation.enabled: true` в конфиге (Δt-рассинхрон, шум в аудио,
дроп кадров/окон, отказ канала, дрожание резкости).
