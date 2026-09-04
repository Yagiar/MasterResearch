# source-simulator — имитатор источников данных (gRPC-клиент ingest-gateway).
# Контекст сборки — корень монорепо (нужны libs/common, libs/proto).
FROM python:3.11-slim

# OpenCV (headless) требует libGL/glib; soundfile — libsndfile (чтение wav)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# устойчивость pip к медленной/нестабильной сети при сборке
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

# Сначала контракты/общий код (кешируемые слои), затем сам сервис
COPY libs/proto /app/libs/proto
COPY libs/common /app/libs/common
COPY services/source-simulator /app/services/source-simulator

# Генерация gRPC-стабов внутри образа (в репозитории они не хранятся)
RUN pip install --no-cache-dir grpcio-tools \
    && python -m grpc_tools.protoc \
        -I libs/proto/src/uavdet_proto \
        --python_out=libs/proto/src/uavdet_proto \
        --grpc_python_out=libs/proto/src/uavdet_proto \
        libs/proto/src/uavdet_proto/ingest.proto \
    && sed -i 's/^import ingest_pb2/from . import ingest_pb2/' \
        libs/proto/src/uavdet_proto/ingest_pb2_grpc.py

RUN pip install --no-cache-dir ./libs/common ./libs/proto ./services/source-simulator

ENV UAVDET_CONFIG=/app/configs/pilot.yaml
ENTRYPOINT ["python", "-m", "source_simulator"]
