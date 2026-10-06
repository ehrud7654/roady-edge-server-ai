FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    ROADY_AI_MODEL_PATH=/app/ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.pt \
    ROADY_AI_MODEL_SHA256_PATH=/app/ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.sha256 \
    ROADY_AI_DEVICE=0 \
    ROADY_AI_IMGSZ=768 \
    ROADY_AI_INFERENCE_CONCURRENCY=1 \
    ROADY_AI_VERIFY_MODEL_HASH=true \
    ROADY_AI_WARMUP_ENABLED=true

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ROADY_AI/requirements.txt ROADY_AI/requirements-server.txt /tmp/requirements/

RUN python -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir \
        -r /tmp/requirements/requirements-server.txt

# Keep model/runtime settings after dependency installation so model swaps do
# not invalidate the expensive PyTorch installation layer.
ENV ROADY_AI_MODEL_PATH=/app/ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.pt \
    ROADY_AI_MODEL_SHA256_PATH=/app/ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.sha256 \
    ROADY_AI_DEVICE=0 \
    ROADY_AI_IMGSZ=768 \
    ROADY_AI_INFERENCE_CONCURRENCY=1 \
    ROADY_AI_VERIFY_MODEL_HASH=true \
    ROADY_AI_WARMUP_ENABLED=true

COPY ROADY_AI /app/ROADY_AI

RUN useradd --create-home --uid 10001 roady \
    && mkdir -p /tmp/ultralytics \
    && chown -R roady:roady /tmp/ultralytics /app

USER roady

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"]

CMD ["uvicorn", "ROADY_AI.server_damage_analysis.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
