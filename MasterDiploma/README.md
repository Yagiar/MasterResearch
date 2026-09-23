# MasterDiploma — мультимодальная система обнаружения БПЛА (НИР-2)

Магистерская НИР (ИТМО, ОП «Системное и прикладное ПО», 09.04.04). Тема НИР текущего семестра: **«Методы мультимодального анализа визуальных и акустических данных для автоматизации обнаружения БПЛА»**.

Это монорепо с **кодом продукта** (микросервисный потоковый пайплайн на Apache Kafka) и **train-приложением** (обучение/оценка моделей). Архитектура спроектирована и задокументирована в Obsidian-вольте: `/Users/otrix/MasterDiplomaVaultObsidian/00 — Карта/03 — НИР-2 (текущий семестр)/Архитектура приложения/` (микросервисы, контракты, паттерны GoF/SOLID, high-load, развёртывание, C4-диаграммы). Черновик отчёта по НИР — `reports/НИР-2/otchet.md`.

> Текущий статус: сквозной путь (`source-simulator → ingest-gateway → visual-detector → fusion → sink`) + акустическая ветка (`acoustic-detector` + обучение CNN в `train/`) + режимы fusion `video-only`/`audio-only`/`late`/`hybrid` с adaptive gating + PostgreSQL для результатов + dashboard-`gateway` (REST/WS + веб-дашборд) + источник с аудио (`MediaFileAdapter`) и реальный «канал деградации» (на пилоте выключен). Парсеры в `prepare_visual` терпимы к раскладке архивов, но проверены не на всех релизах. **Живые Docker/GPU-прогоны пайплайна выполнялись многократно** (ablation v2–v5, it-30/42…46; последние — живой A/B ансамбля it-81 и фоновый FP-прогон it-82, REPRODUCE §6o/§6p в `../research/`).

## Структура

```
libs/common/           — uavdet-common: интерфейсы, pydantic-схемы сообщений, Kafka-обёртка, базовый consumer-сервис, конфиг, метрики
libs/proto/            — uavdet-proto: gRPC-контракт source-simulator ↔ ingest-gateway (ingest.proto + сгенерированные стабы)
services/
  source-simulator/    — ИМИТАТОР источников (gRPC-клиент): replay видео (dataset_replay) / видео+аудио из wav (media_file) / синтетика; канал деградации (Δt, дроп, шум, отказ канала)
  ingest-gateway/      — приёмник (gRPC-сервер → Kafka): нормализует поток, публикует в video.raw / audio.raw
  visual-detector/     — консьюмер video.raw → YOLOv8 + ByteTrack → inference (modality=video)
  acoustic-detector/   — консьюмер audio.raw → MFCC/спектрограмма + lightweight CNN → inference (modality=audio); без весов — режим энергетического порога
  fusion/              — консьюмер inference → временное выравнивание + video-only/audio-only/late/hybrid + adaptive gating → decisions
  sink/                — консьюмер inference + decisions → лог + Prometheus-метрики + jsonl + PostgreSQL
  gateway/             — dashboard-bff: REST (/decisions,/stats) + WS-стрим /ws/decisions + веб-дашборд /  (профиль `dashboard`)
train/                 — обучение: notebooks/pipeline.ipynb + пакет src/uavtrain (download → prepare → train → eval → export) + CLI `python -m uavtrain.cli`
infra/                 — docker-compose: Kafka (KRaft, single-broker) + PostgreSQL + Liquibase + опц. Prometheus/Grafana; postgres/changelog — миграции схемы uavdet. Профили: `observability` (Prometheus/Grafana), `migrate` (Liquibase), `audio` (acoustic-detector), `dashboard` (gateway)
configs/               — YAML-профили (pilot.yaml) + example.env
models/                — веса моделей (в git не коммитятся; пишутся train-приложением при export; туда же кладутся веса YOLO из ВКР)
sandboxDataForSimulator/ — медиа-сэмплы имитатора (видео+wav дрона, «тепличный» сценарий; файлы в git не коммитятся — см. README в папке)
reports/НИР-2/         — черновик отчёта по НИР (markdown → .docx через pandoc)
```

Топики Kafka: `video.raw`, `audio.raw`, `inference`, `decisions` (key = `source_id`, JSON, `schema_ver: 1`). Результаты пайплайна (`inference`, `decisions`) дополнительно пишутся в PostgreSQL (схема `uavdet`, таблицы `inference`/`decisions`; миграции — Liquibase).

## Runbook — этапы (обучение → запуск → проверка)

### 0. Окружение
```bash
git clone https://github.com/Yagiar/MasterDiploma.git && cd MasterDiploma
python -m venv .venv && source .venv/bin/activate
pip install -e 'libs/proto[dev]' -e libs/common         # libs/proto[dev] тянет grpcio-tools
make proto-gen                                          # сгенерировать gRPC-стабы из libs/proto/.../ingest.proto
make install                                            # установить остальные сервисы (-e)
cp configs/example.env .env                             # при необходимости отредактировать
```

### 1. Обучение моделей (отдельное приложение `train/`)
Запускается на машине с GPU (для НИР — удалённый сервер с RTX 3090). Notebook-ориентированный пайплайн: `train/notebooks/pipeline.ipynb` (секциями) либо CLI `python -m uavtrain.cli <команда>`; логика — в пакете `train/src/uavtrain/`. Подробности — `train/README.md`.

1. **Скачать датасеты** — секция 1 ноутбука (`uavtrain.datasets.download(...)`): DUT Anti-UAV + Drone-vs-Bird (визуал), Multiclass Acoustic + Al-Emadi/ESC-50 (аудио). Скачивается в `train/data/`, проверяются хеши.
2. **Подготовить данные** — секция 2 (`prepare_visual` / `prepare_audio`): конвертация в YOLO-формат / спектрограммы, train/val/test split.
3. **Обучить** — секции 3–4 (`train_visual` — Ultralytics YOLOv8 fine-tune; `train_acoustic` — lightweight CNN, PyTorch).
4. **Eval** — секция 5 (`evaluate`): метрики на test (mAP@IoU, F1, precision, recall, confusion matrix), отчёт сохраняется (json + png).
5. **Export весов** — секция 6 (`export`): лучшие веса → `models/visual/`, `models/acoustic/`; обновляется `models/README.md` (версия, метрики, дата).

*На MVP, если своих весов нет, `visual-detector` работает на предобученных весах YOLOv8n (`models/visual/yolov8n.pt`) — положить вручную или скачать через Ultralytics.*

### 2. Поднять инфру
```bash
make infra-up        # Kafka (KRaft, single-broker) + PostgreSQL
make topics-create   # создать video.raw / audio.raw / inference / decisions
make db-migrate      # применить миграции Liquibase (схема uavdet: таблицы inference, decisions)
# опц. наблюдаемость: docker compose -f infra/docker-compose.yml --profile observability up -d
```

### 3. Запустить пайплайн
```bash
make run-pipeline       # MVP-путь (видео), фоном (-d): source-simulator → ingest-gateway → visual-detector → fusion → sink
make run-pipeline-mm    # полный мультимодальный путь, фоном (+ acoustic-detector; источник с аудио — adapter: media_file)
make run-pipeline-gpu   # то же на GPU, фоном: visual-detector на CUDA + проброс NVIDIA GPU; FPS источника 25 (реалтайм).
                        #   нужен nvidia-container-toolkit на хосте; device=cuda задаётся env-override (pilot.yaml не трогается)
make run-dashboard      # gateway, фоном: REST/WS + веб-дашборд http://localhost:8080
make pipeline-logs      # логи всех сервисов пайплайна (Ctrl+C — выйти; контейнеры продолжат работать; для GPU — pipeline-logs-gpu)
make pipeline-down      # остановить/удалить контейнеры приложения (инфра остаётся; для неё make infra-down)
```
Все `run-*` запускаются в фоне (`-d`) — терминал не блокируется; смотреть логи — `make pipeline-logs`.
`source-simulator` проигрывает видео покадрово (FPS из `configs/pilot.yaml` — для CPU 5, для GPU-прогона 25) и параллельно стримит аудио-окна из wav по gRPC в `ingest-gateway`; gateway публикует в `video.raw`/`audio.raw`; `visual-detector` (YOLO) и `acoustic-detector` (MFCC+CNN) → `inference`; `fusion` (режим из конфига; на пилоте `late`) → `decisions`; `sink` логирует, пишет `data/decisions/decisions.jsonl`, экспонирует Prometheus-метрики и складывает `inference`/`decisions` в PostgreSQL; `gateway` (если поднят) показывает поток решений на `http://localhost:8080`.

> На CPU `visual-detector` (YOLO + ByteTrack) не держит реальный темп видео — лаг копится; `run-pipeline-gpu` решает это. См. `reports/pilot_results.md`.

### 4. Проверить
```bash
# решения в топике decisions
docker compose -f infra/docker-compose.yml exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic decisions --from-beginning --max-messages 5
# метрики sink
curl localhost:9108/metrics
# лог решений
tail -f data/decisions/decisions.jsonl
# результаты в PostgreSQL
docker compose -f infra/docker-compose.yml exec postgres psql -U uavdet -d uavdet -c "SELECT count(*) FROM uavdet.decisions; SELECT count(*) FROM uavdet.inference;"
# дашборд: http://localhost:8080  (или REST: curl localhost:8080/stats ; curl 'localhost:8080/decisions?limit=10')
# Grafana (если поднята): http://localhost:3000
```

### 5. Остановить
```bash
make infra-down
```

## Прочие команды
- `make test` — pytest по монорепо.
- `make lint` — ruff check.
- `make run-simulator` / `run-gateway` / `run-visual` / `run-fusion` / `run-sink` — запустить отдельный сервис.
- `make proto-gen` — пересгенерировать gRPC-стабы (после правки `ingest.proto`).

## Git
Удалённый репозиторий: https://github.com/Yagiar/MasterDiploma. Коммиты — фактические (что было сделано), без сторонних соавторов.
