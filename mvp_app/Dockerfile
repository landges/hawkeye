# ---- 1-й этап: runtime ----
FROM python:3.9-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    # где будут храниться веса DeepFace / RetinaFace
    DEEPFACE_HOME=/root/.deepface \
    # предотвратить предупреждение OpenBLAS о многопоточности
    OPENBLAS_NUM_THREADS=1

# ——— системные зависимости (OpenCV, FAISS, Torch) ———
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake \
        ffmpeg libsm6 libxext6 libxrender-dev \
        libgl1 libglib2.0-0 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ——— Python-зависимости ———
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- 2-й этап: preload models ----
FROM runtime AS preload

# Небольшой скрипт, который скачает Facenet512 и RetinaFace веса
RUN python - <<'PY'
from deepface import DeepFace
from retinaface import RetinaFace
DeepFace.build_model("Facenet512")
RetinaFace.build_model()
print("[*] Models ready.")
PY

# ---- 3-й этап: финальный образ ----
FROM runtime

# копируем уже скачанные веса из слоя preload
COPY --from=preload /root/.deepface /root/.deepface

# копируем исходники приложения
WORKDIR /app
COPY ./app /app/app

# убедимся, что DATA_DIR (по умолчанию "data") существует и записываемый
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
