# train — обучающий пайплайн (GPU-сборка для сервера с RTX 3090).
# Контекст сборки — каталог train/.
# Базовый образ — официальный PyTorch с CUDA; на CPU-машине заменить на python:3.11-slim
# (Ultralytics/torch встанут в CPU-варианте — пригодно только для подготовки/eval, не для обучения).
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# устойчивость pip к медленной/нестабильной сети при сборке
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN pip install --no-cache-dir -e '.[notebook]'

# каталоги данных/выводов — монтируются как volume при запуске
RUN mkdir -p /app/data /app/runs

EXPOSE 8888
# по умолчанию — Jupyter Lab/Notebook с ноутбуком пайплайна;
# для CLI: docker run ... uavtrain <команда>
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]
