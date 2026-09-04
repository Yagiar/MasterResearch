# sink — терминальный потребитель результатов

Consumer-сервис на **два топика**:
- `decisions` → структурный лог + запись в **JSON Lines** (`data/decisions/decisions.jsonl`) +
  агрегаты в **Prometheus** (`/metrics` :9108) + (опц.) **PostgreSQL** (`uavdet.decisions`);
- `inference` → (опц.) **PostgreSQL** (`uavdet.inference`) + счётчик.

Каждый consumer крутится в своём потоке (`KafkaConsumerService` = один топик); PG-хранилище
(`PgResultStore`) общее, запись идемпотентна (`ON CONFLICT (msg_id) DO NOTHING`). Схему БД
создаёт Liquibase (`make db-migrate`, см. `infra/postgres/changelog/`). На пилоте это «конец»
пайплайна; в целевой конфигурации рядом встаёт `dashboard-bff` (сервис `gateway`).

## Место в пайплайне
```
inference --\
             >--> [sink] --> stdout-лог + decisions.jsonl + /metrics + PostgreSQL (uavdet.inference / uavdet.decisions)
decisions --/
```

## Конфиг (`configs/pilot.yaml`)
```yaml
kafka:
  bootstrap_servers: kafka:9092
postgres:
  enabled: true                # false -> не писать в БД (только jsonl/метрики)
  host: postgres
  port: 5432
  dbname: uavdet
  user: uavdet
  password: uavdet             # пилот; в проде — UAVDET_POSTGRES__PASSWORD / секреты
  # dsn: "host=... port=... dbname=... user=... password=..."   # альтернатива
sink:
  group_id: sink
  inference_group_id: sink-inference
  metrics_port: 9108
  decisions_log: /data/decisions/decisions.jsonl   # пусто -> не писать jsonl
  also_consume_inference: true                     # потреблять inference и писать в uavdet.inference
  auto_offset_reset: latest
```

## Запуск
```bash
make run-pipeline     # в составе MVP-пайплайна (Docker)
make run-sink         # отдельно (Docker)

# локально (нужен поднятый Kafka):
pip install -e libs/common -e services/sink
python -m sink --config configs/pilot.yaml
```

## Структура
| Модуль | Назначение |
|---|---|
| `consumer.py` | `DecisionsConsumer` (decisions → лог/jsonl/метрики/PG), `InferenceRecorder` (inference → PG) |
| `store.py` | `JsonlStore` — append-only запись решений в .jsonl |
| `metrics_collector.py` | `MetricsCollector` — обновление Prometheus-метрик + краткий лог |
| `__main__.py` | точка входа (`python -m sink`) — поднимает оба consumer'а в потоках |

PG-хранилище (`PgResultStore`) и схема — в `libs/common` (`uavdet_common.db`) и `infra/postgres/`.

## Проверка
```bash
# метрики
curl -s http://localhost:9108/metrics | grep uavdet_
# решения в jsonl
tail -n 5 data/decisions/decisions.jsonl
# результаты в PostgreSQL
docker compose -f infra/docker-compose.yml exec postgres psql -U uavdet -d uavdet -c \
  "SELECT mode, count(*), avg(p_fused)::numeric(4,3) FROM uavdet.decisions GROUP BY mode;"
docker compose -f infra/docker-compose.yml exec postgres psql -U uavdet -d uavdet -c \
  "SELECT modality, label, count(*) FROM uavdet.inference GROUP BY modality, label;"
```
