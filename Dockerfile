FROM python:3.11-slim-bookworm

LABEL maintainer="Chandigarh Police Hackathon Team"
LABEL description="CPH Forensics - Node 1: Pramaan-X Multimodal Manipulation Detection"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8001 \
    PRAMAAN_DEVICE=cpu

# 1. Install system utilities and FFmpeg (required for video/audio extraction)
RUN sed -i 's|URIs: http://deb.debian.org/debian$|URIs: http://mirror.kakao.com/debian|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install PyTorch CPU pair
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

# 3. Install project dependencies
COPY requirements-inference.txt .
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt && \
    pip install --no-cache-dir onnxruntime==1.23.2 && \
    pip install --no-cache-dir -r requirements-api.txt

# 4. Copy project code
COPY . .

# Ensure required directories exist
RUN mkdir -p checkpoints/fusion checkpoints/visual_temporal checkpoints/audio_classifier checkpoints/av_sync_dense \
             models/mediapipe models/xlsr_sls predictions/temp_uploads

EXPOSE 8001

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
