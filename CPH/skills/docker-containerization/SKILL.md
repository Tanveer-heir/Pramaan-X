---
name: docker-containerization
description: Comprehensive Docker and container orchestration guide. Covers multi-stage builds, layer caching optimization, docker-compose orchestration, GPU/CUDA acceleration, healthchecks, and container debugging.
---

# Docker & Containerization Engineering

This skill governs container lifecycle, Dockerfile authoring, multi-service composition, and containerized debugging for production and local environments.

---

## 1. Production Dockerfile Best Practices

### A. Multi-Stage Builds
Always separate the build environment (compilers, dev headers, SDKs) from the runtime environment (minimal footprint):

```dockerfile
# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime-only OS dependencies (e.g., ffmpeg, libgl1 for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Non-root user security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### B. Layer Caching Rules
- **Order from least frequent to most frequent change**:
  1. Base OS dependencies (`apt-get`)
  2. Dependency manifests (`requirements.txt`, `package.json`, `Cargo.toml`)
  3. Package installations (`pip install`, `npm install`)
  4. Application source code (`COPY . .`)
- **Clean package caches in the same RUN step**:
  `RUN apt-get update && apt-get install -y ... && rm -rf /var/lib/apt/lists/*`

---

## 2. Docker Compose for Local & Staging Stacks

Organize multi-container architectures (API, database, Redis cache/queue, vector store) cleanly:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:secret@postgres:5432/cph_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: cph_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d cph_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

---

## 3. AI / ML & GPU Acceleration in Containers

When running deep learning or vision pipelines (e.g. PyTorch, ONNX, C2PA tools):
* **NVIDIA GPU Passthrough**:
  Add `runtime: nvidia` or deploy options in compose:
  ```yaml
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  ```
* **Shared Memory (`shm_size`)**: PyTorch dataloaders often crash with `Bus error` in Docker without adequate shared memory:
  ```yaml
  shm_size: '2gb' # or --shm-size=2g in docker run
  ```

---

## 4. Diagnostics & Troubleshooting Playbook

| Issue | Diagnosis Command | Root Cause & Resolution |
| :--- | :--- | :--- |
| **Container exits immediately** | `docker logs <container_id> --tail 50` | Missing env var or broken CMD. Run with `docker run -it --entrypoint sh <image>` to inspect. |
| **Port already allocated** | `netstat -ano \| findstr :8000` (Windows) | Another local service or stale container bound to the port. Kill process or remap `8001:8000`. |
| **Volume permission denied** | `docker exec -it <id> ls -la /app/data` | Container UID (non-root) does not have write access to host-mounted volume. Run `chown` or adjust UID. |
| **Healthcheck failing** | `docker inspect --format='{{json .State.Health}}' <id>` | Service taking too long to boot or endpoint returning 500. Increase `start-period` or debug endpoint. |
| **Out of disk space** | `docker system df` | Stale layers, dangling volumes, and stopped containers. Clean safely: `docker system prune -f`. |

---

## 5. Non-Interactive CLI Guardrails

When using Docker from the terminal inside agent sessions:
* Always use `-d` (detached) when launching long-running containers.
* Use `--rm` for one-off commands (e.g. `docker run --rm <image> pytest`).
* Never run interactive shells (`-it`) without detached mode or input redirect.
* To execute a command inside a running container: `docker exec <container_name> python -m pytest`.
