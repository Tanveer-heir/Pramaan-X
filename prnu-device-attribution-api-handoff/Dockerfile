FROM python:3.11-slim-bookworm AS builder

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="Chandigarh Police Hackathon Team"
LABEL description="Pramaan-X device attribution and PRNU service"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8002 \
    HOST=0.0.0.0

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY prnu_attribution ./prnu_attribution
COPY web ./web

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p dataset models runs/uploads /data/shared_media \
    && chown -R appuser:appuser /app /data/shared_media

USER appuser
EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["python", "-m", "prnu_attribution.api", "--host", "0.0.0.0", "--port", "8002"]
