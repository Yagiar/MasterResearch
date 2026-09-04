# acoustic-detector — MFCC/мел-спектрограммы + lightweight CNN (CPU-сборка для пилота).
# Контекст сборки — корень монорепо (нужен libs/common).
# GPU-вариант: заменить базу на nvidia/cuda + torch с CUDA, в configs.pilot.yaml — device: "cuda".
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 librdkafka1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# устойчивость pip к медленной/нестабильной сети при сборке
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

# torch + torchvision (CPU) отдельным слоем для кеша (torchvision — ResNet-18 для arch=resnet18)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir librosa numpy \
    && pip install --no-cache-dir 'transformers>=4.40' 'safetensors>=0.4'

COPY libs/common /app/libs/common
COPY services/acoustic-detector /app/services/acoustic-detector

RUN pip install --no-cache-dir ./libs/common ./services/acoustic-detector

RUN mkdir -p /models/acoustic

ENV UAVDET_CONFIG=/app/configs/pilot.yaml
ENTRYPOINT ["python", "-m", "acoustic_detector"]
