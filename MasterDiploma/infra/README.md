# infra/ — инфраструктура (Kafka + опц. Prometheus/Grafana)

Для НИР-2 (пилот) — минимальная конфигурация: **Apache Kafka в KRaft-режиме** (single-broker, без ZooKeeper), `replication.factor=1`, по 1 партишну на топик; **PostgreSQL 16** для результатов (таблицы `uavdet.inference`, `uavdet.decisions`), схема — миграциями **Liquibase**. Целевая конфигурация (кластер брокеров RF≥3, claim-check через MinIO, k8s-развёртывание) — см. вольт `00 — Карта/03 — НИР-2 (текущий семестр)/Архитектура приложения/Развёртывание (пилот и целевая конфигурация).md`.

## Файлы
- `docker-compose.yml` — `kafka` + `postgres` (всегда) + `liquibase` (профиль `migrate`, one-shot) + `prometheus`/`grafana` (профиль `observability`).
- `docker-compose.app.yml` — сервисы приложения (`source-simulator`, `ingest-gateway`, `visual-detector`, `acoustic-detector`[профиль `audio`], `fusion`, `sink`, `gateway`[профиль `dashboard`]); накладывается поверх `docker-compose.yml`.
- `postgres/changelog/` — Liquibase: `db.changelog-master.xml`, `liquibase.properties`, `changes/*.sql` (схема `uavdet`, таблицы `inference`/`decisions`).
- `prometheus/prometheus.yml` — scrape-конфиг (`sink:9108`).
- `grafana/provisioning/` — datasource (Prometheus) + provider дашбордов (минимально).

## Команды (из корня репо)
```bash
make infra-up        # поднять Kafka + PostgreSQL
make topics-create   # создать топики video.raw / audio.raw / inference / decisions
make db-migrate      # применить миграции Liquibase (схема uavdet)
make run-pipeline    # MVP-пайплайн: source-simulator → ingest-gateway → visual-detector → fusion → sink
make infra-down      # остановить инфру

# с наблюдаемостью (Prometheus + Grafana):
docker compose -f infra/docker-compose.yml --profile observability up -d
```

## Порты
| Сервис | Порт | Назначение |
|---|---|---|
| kafka | 9092 | PLAINTEXT для клиентов снаружи docker-сети (`localhost:9092`) |
| postgres | 5432 | PostgreSQL (`uavdet`/`uavdet`/`uavdet` — пилотные креды) |
| sink | 9108 | Prometheus `/metrics` |
| prometheus | 9090 | UI Prometheus (профиль `observability`) |
| grafana | 3000 | UI Grafana, анонимный доступ (профиль `observability`) |
| gateway | 8080 | REST/WS dashboard-bff (профиль `dashboard`, заготовка) |

Внутри docker-сети `uavdet-net` сервисы обращаются к брокеру по `kafka:9092`, к gateway — по `ingest-gateway:50051` (gRPC), к БД — по `postgres:5432`.

## Проверка
```bash
# топики
docker compose -f infra/docker-compose.yml exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
# содержимое decisions
docker compose -f infra/docker-compose.yml exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic decisions --from-beginning --max-messages 5
# таблицы и данные в PostgreSQL
docker compose -f infra/docker-compose.yml exec postgres psql -U uavdet -d uavdet -c "\dt uavdet.*"
docker compose -f infra/docker-compose.yml exec postgres psql -U uavdet -d uavdet -c "SELECT count(*) FROM uavdet.decisions;"
```
