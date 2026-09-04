# uavdet-common

Общая библиотека для всех сервисов системы обнаружения БПЛА (НИР-2). Реализует точки вариации из C4-модели (см. вольт: `00 — Карта/03 — НИР-2 (текущий семестр)/Архитектура приложения/`).

## Модули
| Модуль | Что |
|---|---|
| `abstractions` | интерфейсы (Protocol): `DataSource`, `DegradationStrategy`, `Detector`, `FusionStrategy`, `GatingPolicy`, `MessageBus`, `Serializer`, `MetricsSink` |
| `messages` | pydantic-схемы сообщений Kafka-топиков: `VideoRawMsg`, `AudioRawMsg`, `InferenceMsg`, `DecisionMsg`, `ConfigCommandMsg`; класс `Topics` с именами топиков; `TOPIC_MODELS` (карта топик→модель); `schema_ver = 1` |
| `serialization` | `JsonSerializer` (pydantic ↔ bytes); хелперы base64 для бинарных payload |
| `bus` | `KafkaMessageBus` (confluent-kafka, at-least-once, ручной commit) и `InMemoryBus` (для тестов / самого раннего пилота) |
| `consumer_service` | `KafkaConsumerService` — базовый каркас consumer-сервиса (Template Method: poll → deserialize → process → produce → commit; graceful shutdown по SIGINT/SIGTERM; идемпотентность по `msg_id` через LRU). Подклассы задают `in_topic`/`group_id`/`in_model` и реализуют `process()` |
| `config` | `load_config()` — YAML-профиль (`configs/pilot.yaml` или `$UAVDET_CONFIG`) + переопределение env-переменными `UAVDET_<path__через__split>`; `Settings` (pydantic-settings) |
| `metrics` | Prometheus-метрики (`uavdet_messages_total`, `uavdet_e2e_latency_seconds`, `uavdet_consumer_lag`, `uavdet_delta_t_ms`, ...) + `start_metrics_server(port)` |
| `logging` | настройка `structlog` (`configure_logging`, `get_logger`) |
| `db` | `PgResultStore` — запись `InferenceMsg`/`DecisionMsg` в PostgreSQL (psycopg3, идемпотентно через `ON CONFLICT (msg_id)`); `make_dsn(...)`. Схему БД создаёт Liquibase (`infra/postgres/changelog/`) |

## Установка
```bash
pip install -e libs/common          # из корня репо
pip install -e 'libs/common[dev]'   # + pytest
```

## Тесты
```bash
cd libs/common && pytest
```
Покрывают: round-trip сериализацию схем, `InMemoryBus`, `KafkaConsumerService` (обработка + дедупликация по `msg_id`) — без реального Kafka.

## Зависимость
Каждый сервис в `services/*` объявляет `uavdet-common` в своих зависимостях. `uavdet-common` сам по себе лёгкий (без numpy / torch / ultralytics — массивы кадров типизированы как `Any`).
