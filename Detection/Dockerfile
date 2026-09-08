FROM python:3.12-slim-bookworm AS builder

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY Detection/requirements-inference.txt Detection/requirements-service.txt /tmp/detection/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url ${PYTORCH_INDEX_URL} && \
    pip install --no-cache-dir -r /tmp/detection/requirements-service.txt

FROM python:3.12-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DETECTION_ALLOWED_MEDIA_ROOT=/data/shared_media \
    DETECTION_OUTPUT_DIR=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app/Detection
COPY --chown=appuser:appuser Detection/scripts ./scripts
COPY --chown=appuser:appuser Detection/service ./service

RUN mkdir -p /app/Detection/checkpoints /app/Detection/models /data/shared_media /tmp/pramaan-x-detection \
    && chown -R appuser:appuser /data/shared_media /tmp/pramaan-x-detection

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]
