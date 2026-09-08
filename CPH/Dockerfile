FROM python:3.11-slim-bookworm AS builder

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="Chandigarh Police Hackathon Team"
LABEL description="Pramaan-X source attribution and unified gateway"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    exiftool \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
RUN playwright install --with-deps chromium

WORKDIR /app
COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p data/cache/investigation data/cache/astar data/graphs data/investigations /data/shared_media \
    && chown -R appuser:appuser /app /data/shared_media

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
