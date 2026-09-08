---
name: backend-management
description: High-reliability backend engineering and lifecycle management. Covers API architecture (FastAPI/REST), database migrations (Alembic), async task queues (Redis/Celery), structured telemetry, RBAC, and error envelopes.
---

# Backend Engineering & Lifecycle Management

This skill provides architectural patterns, API design standards, database lifecycle management, and service orchestration for production backend systems.

---

## 1. API Design & Contract Standards

### A. Strict Schema Enforcement (FastAPI + Pydantic)
* Use typed Pydantic models for all request bodies, query parameters, and response payloads.
* Enable strict validation: reject undeclared fields (`extra = 'forbid'`) on sensitive endpoints.
* Provide automated OpenAPI documentation (`/docs`, `/redoc`).

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from enum import Enum

class VerdictEnum(str, Enum):
    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"

class ProvenanceCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(..., description="Unique media file identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    c2pa_valid: bool
    exif_consistent: bool
    prnu_match: bool
    verdict: VerdictEnum
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)
```

### B. Standardized API Response Envelopes
Never return raw strings or inconsistent dict structures across endpoints:

```json
// Success Response (200 OK)
{
  "status": "success",
  "data": { ... },
  "metadata": {
    "request_id": "req_01h8a9bc...",
    "latency_ms": 42.1
  }
}

// Error Response (4xx / 5xx)
{
  "status": "error",
  "error": {
    "code": "INVALID_EVIDENCE_PAYLOAD",
    "message": "Field 'file_id' must be a valid UUID string.",
    "details": [...]
  },
  "metadata": {
    "request_id": "req_01h8a9bc..."
  }
}
```

---

## 2. Database Lifecycle & Migrations

### A. Migration Hygiene (Alembic)
* **Never alter production schemas manually** via raw SQL commands.
* Generate migrations incrementally:
  ```bash
  alembic revision --autogenerate -m "add_provenance_verdict_index"
  ```
* Review every generated migration file before applying: verify that downgrades (`downgrade()`) are fully implemented.
* Apply migrations non-interactively:
  ```bash
  alembic upgrade head
  ```

### B. Connection Pooling & Transactions
* Use connection pools (`pool_size=10, max_overflow=20`) to prevent database connection exhaustion.
* Wrap multi-step mutations (e.g. creating evidence record + updating audit log) in explicit database transactions (`with session.begin(): ...`).

---

## 3. Asynchronous Task Processing & Queues

For compute-intensive operations (media hashing, reverse image search, PRNU computation):
* **Never block the HTTP request loop** with heavy CPU/IO processing (>200ms).
* Offload to background worker queues (Celery, RQ, or FastAPI `BackgroundTasks` for light tasks):

```python
from fastapi import APIRouter, BackgroundTasks, status

router = APIRouter(prefix="/api/v1/jobs")

@router.post("/process-media", status_code=status.HTTP_202_ACCEPTED)
async def submit_media_analysis(payload: MediaRequest, background_tasks: BackgroundTasks):
    job_id = generate_job_id()
    # Enqueue background task or submit to Redis queue
    background_tasks.add_task(run_provenance_pipeline, job_id, payload)
    return {
        "status": "accepted",
        "job_id": job_id,
        "poll_url": f"/api/v1/jobs/{job_id}"
    }
```

---

## 4. Security, Authentication & RBAC

* **Authentication**: Use JWT tokens with short lifetimes (15-60 min) and secure refresh tokens.
* **Role-Based Access Control (RBAC)**:
  - `Admin`: Full access, user management, configuration changes.
  - `Investigator`: Run provenance checks, origin tracing, generate evidence reports.
  - `Viewer`: Read-only access to final reports and dashboard metrics.
* **Rate Limiting**: Protect endpoints against brute-force or denial-of-service using Redis-backed rate limiters (e.g. `slowapi` or nginx).

---

## 5. Observability & Graceful Shutdown

* **Structured Logging**: Output logs in JSON format with `timestamp`, `level`, `request_id`, `service`, and `msg`.
* **Health Probes**:
  - `/health/live`: Checks if process is running.
  - `/health/ready`: Verifies database, Redis, and dependent services are reachable.
* **Graceful Shutdown**: Intercept `SIGTERM` / `SIGINT` to allow in-flight requests and background jobs to complete cleanly before termination.
