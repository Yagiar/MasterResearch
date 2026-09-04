# gateway — dashboard backend-for-frontend

FastAPI-сервис: фоновый Kafka-потребитель топика `decisions` → кольцевой буфер последних N
решений + агрегаты; REST + WebSocket-стрим + минимальный веб-дашборд (одна страница). На пилоте
может быть слит с `sink`; вынесен в отдельный сервис для будущего фронтенда. В docker-compose —
под профилем `dashboard` (запускается явно).

## Эндпоинты
| Метод | Путь | Что |
|---|---|---|
| GET | `/healthz` | health-проба |
| GET | `/status` | состояние (жив ли consumer, размер буфера, топик) |
| GET | `/stats` | агрегаты: `total`, `drone_ratio`, `avg_p_fused`, `avg_e2e_latency_ms`, `by_mode` |
| GET | `/decisions?limit=N` | последние N решений (новые сверху; dict-представление `DecisionMsg`) |
| WS | `/ws/decisions` | стрим новых решений в реальном времени (при подключении присылает последние 20) |
| GET | `/` | веб-дашборд (`static/index.html`) — карточки агрегатов + живая таблица решений |

## Место в пайплайне
```
decisions  -->  [gateway]  -->  REST /decisions,/stats  ·  WS /ws/decisions  ·  веб-дашборд /
```
(потребляет тот же топик `decisions`, что и `sink`, в собственной consumer-группе — не мешает ему)

## Конфиг (`configs/pilot.yaml`)
```yaml
kafka:
  bootstrap_servers: kafka:9092
gateway:
  bind_host: "0.0.0.0"
  port: 8080
  group_id: gateway-dashboard      # отдельная consumer-группа (не пересекается с sink)
  buffer_size: 500                 # размер кольцевого буфера решений
  auto_offset_reset: latest
```

## Запуск
```bash
# Docker (профиль `dashboard`):
docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml --profile dashboard up gateway
#   -> http://localhost:8080

# локально (нужен поднятый Kafka):
pip install -e libs/common -e services/gateway
python -m gateway --config configs/pilot.yaml
```

## Структура
| Модуль | Назначение |
|---|---|
| `app.py` | FastAPI-приложение (lifespan: поднимает store + Kafka-поток + WS-hub); REST + WS-эндпоинты |
| `consumer.py` | `DecisionsFeedConsumer(KafkaConsumerService)` — `decisions` → `DecisionStore` + callback для WS |
| `state.py` | `DecisionStore` — потокобезопасный кольцевой буфер решений + агрегаты |
| `static/index.html` | веб-дашборд (vanilla JS: карточки `/stats`, таблица `/decisions`, живой WS) |
| `__main__.py` | точка входа (`python -m gateway`) — uvicorn |
