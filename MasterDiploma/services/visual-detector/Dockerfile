# visual-detector — YOLOv8 + ByteTrack (CPU-сборка для пилота).
# Контекст сборки — корень монорепо (нужен libs/common).
# GPU-вариант: заменить базу на nvidia/cuda + установить torch с CUDA, в configs.pilot.yaml — device: "cuda".
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 librdkafka1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# устойчивость pip к медленной/нестабильной сети при сборке
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

# Тяжёлые зависимости (torch/ultralytics) ставим отдельным слоем для кеша
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics supervision opencv-python-headless

COPY libs/common /app/libs/common
COPY services/visual-detector /app/services/visual-detector

RUN pip install --no-cache-dir ./libs/common ./services/visual-detector

# Папка для весов (монтируется как volume ../models -> /models)
RUN mkdir -p /models/visual

ENV UAVDET_CONFIG=/app/configs/pilot.yaml
ENTRYPOINT ["python", "-m", "visual_detector"]
