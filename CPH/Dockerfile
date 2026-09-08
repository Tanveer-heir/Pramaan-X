# ==============================================================================
# Chandigarh Police Digital Forensics Unit (CPH Track §2)
# Autonomous Multi-Platform Origin Attribution & Forensic Intelligence Agent
# ==============================================================================

FROM python:3.11-slim-bookworm

LABEL maintainer="Chandigarh Police Hackathon Team"
LABEL description="Autonomous Media Origin Tracing and Forensic Attribution Engine"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# 1. Install system utilities, ExifTool, and OpenCV headless runtime dependencies
RUN sed -i 's|URIs: http://deb.debian.org/debian$|URIs: http://mirror.kakao.com/debian|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    exiftool \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Layer-cached dependency installation
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 3. Install Playwright Chromium with system shared libraries
RUN playwright install --with-deps chromium

# 4. Copy project code (Bind mount in docker-compose will override for live development)
COPY . .

# Ensure cache and output directories exist with full permissions
RUN mkdir -p data/cache/investigation data/cache/astar data/graphs data/sample_media && \
    chmod -R 777 data

# 5. Default port for API / Dashboard
EXPOSE 8000

# Default entrypoint runs the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
