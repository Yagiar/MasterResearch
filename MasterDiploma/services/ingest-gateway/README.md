# ingest-gateway — приёмник потоков (gRPC-сервер → Kafka)

Тонкий приёмник: **gRPC-сервер** `SourceStream` (контракт — `libs/proto/ingest.proto`),
принимает стримы `Frame`/`AudioWindow` от `source-simulator` (или, в будущем, от
`sensor-driver` — взаимозаменяемых клиентов), **нормализует** (`schema_ver`/`msg_id`/`ts`/`seq`,
base64-упаковка payload) и **публикует** `VideoRawMsg`/`AudioRawMsg` в Kafka-топики
`video.raw` / `audio.raw` (ключ — `source_id`).

## Место в пайплайне
```
source-simulator  --gRPC-->  [ingest-gateway]  --Kafka-->  video.raw / audio.raw  -->  детекторы
```

## Конфиг (`configs/pilot.yaml`)
```yaml
kafka:
  bootstrap_servers: kafka:9092
ingest_grpc:
  bind_host: "0.0.0.0"
  port: 50051
  max_message_mb: 16
  max_workers: 8
  metrics_port: 0          # >0 — поднять /metrics на этом порту
```

## Запуск
```bash
make run-pipeline       # в составе MVP-пайплайна (Docker)
make run-gateway        # отдельно (Docker)

# локально:
pip install -e libs/common -e libs/proto -e services/ingest-gateway
make proto-gen
python -m ingest_gateway --config configs/pilot.yaml
```

## Структура
| Модуль | Назначение |
|---|---|
| `grpc_server.py` | `SourceStreamServicer` + `serve()` — gRPC-сервер |
| `normalizer.py` | protobuf `Frame`/`AudioWindow` → `VideoRawMsg`/`AudioRawMsg` |
| `publisher.py` | `Publisher` — публикация в `video.raw`/`audio.raw` (key=`source_id`) |
| `__main__.py` | точка входа (`python -m ingest_gateway`) |

## Проверка
```bash
docker compose -f infra/docker-compose.yml exec kafka \
  /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic video.raw --from-beginning --max-messages 3
```
