FROM python:3.11-slim-bookworm

LABEL maintainer="Chandigarh Police Hackathon Team"
LABEL description="CPH Forensics - Node 2: Device Attribution & PRNU Fingerprint Service"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8002 \
    HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure dataset, models, and runs directories exist
RUN mkdir -p dataset models runs/uploads

EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["python", "-m", "prnu_attribution.api", "--host", "0.0.0.0", "--port", "8002", "--allow-demo-model"]
