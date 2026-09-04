# fusion — движок мультимодального слияния (inference -> decisions).
# Контекст сборки — корень монорепо (нужен libs/common).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        librdkafka1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# устойчивость pip к медленной/нестабильной сети при сборке
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

COPY libs/common /app/libs/common
COPY services/fusion /app/services/fusion

RUN pip install --no-cache-dir ./libs/common ./services/fusion

ENV UAVDET_CONFIG=/app/configs/pilot.yaml
ENTRYPOINT ["python", "-m", "fusion"]
