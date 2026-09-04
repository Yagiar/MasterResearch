# Makefile — удобные команды монорепо НИР-2.
# Архитектура и runbook — см. README.md и Obsidian-вольт (00 — Карта/03 — НИР-2/Архитектура приложения/).

COMPOSE_INFRA = docker compose -f infra/docker-compose.yml
COMPOSE_APP   = docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml
COMPOSE_GPU   = docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml -f infra/docker-compose.gpu.yml
PROTO_SRC     = libs/proto/src/uavdet_proto

.PHONY: help install train-install proto-gen infra-up infra-down topics-create db-migrate \
        run-pipeline run-pipeline-mm run-pipeline-gpu run-dashboard run-simulator run-gateway run-visual run-acoustic run-fusion run-sink \
        pipeline-logs pipeline-logs-gpu pipeline-down test lint clean

help:
	@echo "Цели:"
	@echo "  install         — pip install -e всех пакетов (libs/common, libs/proto, services/*)"
	@echo "  train-install   — pip install -e train (обучающее приложение, тяжёлые зависимости)"
	@echo "  proto-gen       — сгенерировать gRPC-стабы из libs/proto/src/uavdet_proto/ingest.proto"
	@echo "  infra-up        — поднять инфру (Kafka KRaft + PostgreSQL + опц. Prometheus/Grafana)"
	@echo "  infra-down      — остановить инфру"
	@echo "  topics-create   — создать топики video.raw / audio.raw / inference / decisions"
	@echo "  db-migrate      — применить миграции Liquibase (схема uavdet: таблицы inference, decisions)"
	@echo "  run-pipeline    — MVP-путь (видео), фоном (-d): simulator → ingest-gateway → visual-detector → fusion → sink"
	@echo "  run-pipeline-mm — полный мультимодальный путь, фоном (+ acoustic-detector; нужен источник с аудио)"
	@echo "  run-pipeline-gpu— то же на GPU, фоном (visual-detector на CUDA; нужен nvidia-container-toolkit; FPS 25)"
	@echo "  run-dashboard   — поднять gateway, фоном (веб-дашборд решений, http://localhost:8080)"
	@echo "  run-simulator|run-gateway|run-visual|run-acoustic|run-fusion|run-sink — запустить отдельный сервис фоном (Docker)"
	@echo "  pipeline-logs   — логи сервисов пайплайна (Ctrl+C — выйти, сервисы продолжат работать)"
	@echo "  pipeline-down   — остановить и удалить контейнеры пайплайна (инфра остаётся)"
	@echo "  test            — pytest по монорепо"
	@echo "  lint            — ruff check"
	@echo "  clean           — удалить __pycache__, .pytest_cache, сгенерированные стабы"

install:
	pip install -e libs/common -e libs/proto
	pip install -e services/source-simulator -e services/ingest-gateway -e services/visual-detector -e services/acoustic-detector -e services/fusion -e services/sink -e services/gateway

# обучающее приложение ставится отдельно (тяжёлые зависимости: torch/ultralytics/librosa)
train-install:
	pip install -e 'train[notebook]'

# grpcio-tools должен быть установлен (входит в libs/proto extras: pip install -e 'libs/proto[dev]')
proto-gen:
	python -m grpc_tools.protoc -I$(PROTO_SRC) --python_out=$(PROTO_SRC) --grpc_python_out=$(PROTO_SRC) $(PROTO_SRC)/ingest.proto
	@# фикс импорта в *_pb2_grpc.py на относительный (pb2 лежит рядом)
	@sed -i.bak 's/^import ingest_pb2/from . import ingest_pb2/' $(PROTO_SRC)/ingest_pb2_grpc.py && rm -f $(PROTO_SRC)/ingest_pb2_grpc.py.bak
	@echo "gRPC-стабы сгенерированы в $(PROTO_SRC)/"

infra-up:
	$(COMPOSE_INFRA) up -d kafka postgres

infra-down:
	$(COMPOSE_INFRA) down

topics-create:
	$(COMPOSE_INFRA) exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic video.raw  --partitions 1 --replication-factor 1
	$(COMPOSE_INFRA) exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic audio.raw  --partitions 1 --replication-factor 1
	$(COMPOSE_INFRA) exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic inference  --partitions 1 --replication-factor 1
	$(COMPOSE_INFRA) exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic decisions  --partitions 1 --replication-factor 1
	$(COMPOSE_INFRA) exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list

# применяет changelog (infra/postgres/changelog/db.changelog-master.xml) и завершается
db-migrate:
	$(COMPOSE_INFRA) run --rm liquibase

# Все run-* запускаются в фоне (-d). Логи: make pipeline-logs. Остановить: make pipeline-down.

# MVP-путь (видео): source-simulator -> ingest-gateway -> visual-detector -> fusion -> sink
run-pipeline:
	$(COMPOSE_APP) up -d --build source-simulator ingest-gateway visual-detector fusion sink

# полный мультимодальный путь (+ acoustic-detector; источник должен отдавать аудио — adapter: media_file/mmaud_replay)
run-pipeline-mm:
	$(COMPOSE_APP) up -d --build source-simulator ingest-gateway visual-detector acoustic-detector fusion sink

# то же на GPU: visual-detector — CUDA-образ + проброс NVIDIA GPU; FPS источника поднят до 25 (реалтайм).
# Требования: NVIDIA-драйвер + nvidia-container-toolkit на хосте. (visual_detector.device=cuda через env-override)
run-pipeline-gpu:
	$(COMPOSE_GPU) up -d --build source-simulator ingest-gateway visual-detector acoustic-detector fusion sink

# + дашборд (gateway, http://localhost:8080)
run-dashboard:
	$(COMPOSE_APP) up -d --build gateway

run-simulator:
	$(COMPOSE_APP) up -d --build source-simulator

run-gateway:
	$(COMPOSE_APP) up -d --build ingest-gateway

run-visual:
	$(COMPOSE_APP) up -d --build visual-detector

run-acoustic:
	$(COMPOSE_APP) up -d --build acoustic-detector

run-fusion:
	$(COMPOSE_APP) up -d --build fusion

run-sink:
	$(COMPOSE_APP) up -d --build sink

# логи всех сервисов пайплайна (Ctrl+C — выйти, контейнеры продолжат работать)
pipeline-logs:
	$(COMPOSE_APP) logs -f --tail=100 source-simulator ingest-gateway visual-detector acoustic-detector fusion sink gateway

pipeline-logs-gpu:
	$(COMPOSE_GPU) logs -f --tail=100 source-simulator ingest-gateway visual-detector acoustic-detector fusion sink gateway

# остановить и удалить контейнеры приложения (инфра — Kafka/PostgreSQL — остаётся; для неё make infra-down)
pipeline-down:
	$(COMPOSE_APP) rm -sf source-simulator ingest-gateway visual-detector acoustic-detector fusion sink gateway

test:
	pytest

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
	rm -f $(PROTO_SRC)/ingest_pb2.py $(PROTO_SRC)/ingest_pb2_grpc.py $(PROTO_SRC)/ingest_pb2.pyi
