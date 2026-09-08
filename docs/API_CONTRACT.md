# Pramaan-X API Contract

This document is the frontend handoff for `pramaan_x_investigation_v1`.

## Service topology and ports

| Service | Port | Role |
|---|---:|---|
| CPH unified gateway | 8000 | Browser upload, integrity hash, routing, aggregation, custody, artifacts |
| Detection | 8001 | Accepted image and video Detection CLIs behind HTTP |
| PRNU device attribution | 8002 | Still-image device evidence against metadata or known references |

The browser communicates only with port 8000. The gateway sends the same immutable evidence path to child services through the shared media volume.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DETECTION_SERVICE_URL` | `http://ai-detection:8001` | Detection child-service base URL |
| `PRNU_SERVICE_URL` | `http://prnu-forensics:8002` | PRNU child-service base URL |
| `SHARED_MEDIA_DIR` | `data/shared_media` | Single-write evidence root |
| `INVESTIGATION_OUTPUT_DIR` | `data/investigations` | Investigation-scoped artifact root |
| `DETECTION_TIMEOUT_SEC` | `180` | Gateway Detection timeout |
| `PRNU_TIMEOUT_SEC` | `60` | Gateway PRNU timeout |
| `SOURCE_ATTRIBUTION_TIMEOUT_SEC` | `180` | Gateway source-attribution timeout |
| `SERVICE_HEALTH_TIMEOUT_SEC` | `3` | Capability probe timeout |
| `MAX_UPLOAD_BYTES` | `524288000` | Maximum browser upload size |
| `CORS_ALLOWED_ORIGINS` | localhost ports 3000 and 5173 | Comma-separated frontend origins |
| `KIMI_API_KEY` | none | Detection image-analysis credential |
| `PRAMAAN_DEVICE` | `auto` locally, `cpu` in Compose | Video Detection device selection |
| `GEMINI_API_KEY` | none | CPH source-attribution credential where used |
| `APIFY_API_TOKEN` | none | Optional CPH search connector credential |
| `ALLOW_DEMO_MODEL` | `false` | Explicit PRNU synthetic-model opt-in for smoke tests only |

Use the root `.env.example`. Never commit `.env`.

## Unified upload

`POST /api/v1/investigate/upload`

Content type: `multipart/form-data`

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | JPG, JPEG, PNG, WEBP, MP4, MOV, M4V, AVI, MKV, or WEBM evidence |
| `case_id` | no | Investigator case identifier. A value is generated when omitted. |

The gateway sanitizes the client filename, verifies the content signature, writes the evidence once under a generated investigation identifier, computes SHA-256 during ingestion, and verifies the hash again after module execution. Unsupported content returns HTTP 415. Oversized content returns HTTP 413.

Compatibility alias: `POST /investigate/upload`.

The JSON path endpoint `POST /api/v1/investigate` exists only for trusted local development and is disabled unless `ALLOW_LOCAL_PATH_ENDPOINTS=true`. A frontend must not use it.

## Exact unified response schema

```json
{
  "schema_version": "pramaan_x_investigation_v1",
  "investigation_id": "INV-0123456789ABCDEF",
  "case_id": "CASE-DEMO-001",
  "status": "COMPLETED",
  "media": {
    "filename": "evidence.mp4",
    "media_type": "video",
    "sha256": "64 lowercase hexadecimal characters",
    "size_bytes": 123456
  },
  "modules": {
    "detection": {
      "status": "COMPLETED",
      "reason": null,
      "message": null,
      "duration_sec": 12.3,
      "result": {}
    },
    "device_attribution": {
      "status": "NOT_APPLICABLE",
      "reason": "device attribution currently supports still images only",
      "message": null,
      "duration_sec": 0.0,
      "result": null
    },
    "source_attribution": {
      "status": "COMPLETED",
      "reason": null,
      "message": null,
      "duration_sec": 18.1,
      "result": {}
    }
  },
  "artifacts": {
    "lineage": {
      "url": "/api/v1/investigations/INV-0123456789ABCDEF/artifacts/lineage",
      "media_type": "text/html",
      "sha256": "64 lowercase hexadecimal characters"
    }
  },
  "summary": {
    "detection_label": "AUDIO_MANIPULATION",
    "detection_score_semantics": "uncalibrated_softmax",
    "attributed_device": null,
    "device_method": null,
    "patient_zero_candidate_domain": "example.com"
  },
  "custody": {
    "hash_algorithm": "sha256",
    "entries": [
      {
        "sequence": 1,
        "event": "EVIDENCE_INGESTED",
        "timestamp": "2026-09-08T12:00:00+00:00",
        "input_sha256": "64 lowercase hexadecimal characters",
        "module_statuses": {},
        "artifact_ids": [],
        "previous_entry_hash": null,
        "entry_hash": "64 lowercase hexadecimal characters"
      }
    ]
  },
  "warnings": [],
  "execution_time_sec": 18.4
}
```

`modules.<module>.result` retains the subsystem's native JSON, except server-local path values are replaced with an evidence hash reference, an allowlisted artifact URL, or `REDACTED_INTERNAL_PATH`.

## Module and investigation status

| Module status | Meaning |
|---|---|
| `COMPLETED` | The module returned valid JSON. |
| `NOT_APPLICABLE` | The media type is outside the module's supported scope. |
| `FAILED` | The module failed or returned malformed output. |
| `TIMEOUT` | The module exceeded its independent timeout. |

| Investigation status | Meaning |
|---|---|
| `COMPLETED` | Every applicable module completed. |
| `PARTIAL` | At least one module completed and at least one applicable module failed or timed out. |
| `FAILED` | No module completed, or the post-analysis evidence hash changed. |

A child-service failure is represented inside the HTTP 200 investigation record so successful independent evidence is not discarded.

## Image routing example

Images run Detection image triage, PRNU/device attribution, and CPH A-star visual source attribution concurrently. The Detection result preserves its categorical `assessment`, `visual_findings`, `supporting_signals`, `input`, and `limitations`. It is not converted into a numeric fake probability.

```bash
curl -X POST http://localhost:8000/api/v1/investigate/upload \
  -F "file=@/absolute/path/to/evidence.jpg" \
  -F "case_id=CASE-IMAGE-001"
```

An unknown device remains inconclusive when no matching real reference fingerprint or usable metadata exists. The checked-in synthetic PRNU model is not enabled for external evidence by default.

## Video routing example

Videos run four-class Detection and CPH source attribution concurrently. Device attribution is explicitly `NOT_APPLICABLE`. CPH source attribution first requires a real decoded keyframe. It no longer fabricates a placeholder frame when decoding fails.

```bash
curl -X POST http://localhost:8000/api/v1/investigate/upload \
  -F "file=@/absolute/path/to/evidence.mp4" \
  -F "case_id=CASE-VIDEO-001"
```

The native Detection result retains `selected_class`, `classification`, `class_logits`, `softmax_probabilities`, `branch_evidence`, `counterfactual_modality_analysis`, `temporal_evidence`, `checkpoint_provenance`, and `raw_video` when produced.

## Partial failure example

```json
{
  "status": "PARTIAL",
  "modules": {
    "detection": {
      "status": "COMPLETED",
      "duration_sec": 8.2,
      "result": {"schema_version": "pramaan_x_image_analysis_v1"}
    },
    "device_attribution": {
      "status": "FAILED",
      "reason": "PRNU_ANALYSIS_FAILED",
      "message": "Device-attribution service did not produce a usable result.",
      "duration_sec": 0.4,
      "result": null
    },
    "source_attribution": {
      "status": "TIMEOUT",
      "reason": "SOURCE_ATTRIBUTION_TIMEOUT",
      "message": "Forensic module exceeded its configured timeout.",
      "duration_sec": 180.0,
      "result": null
    }
  }
}
```

The omitted top-level fields remain identical to the exact schema above.

## Score semantics warning

- Video `softmax_probabilities` are retained under their native name for compatibility, but they are uncalibrated class scores.
- Image confidence is `LOW`, `MEDIUM`, or `HIGH` categorical LLM triage.
- PRNU and metadata values are independent device evidence. They do not alter Detection.
- The current A-star `probability` and `confidence` fields are heuristic origin-ranking values from the native CPH implementation. They are not calibrated Patient Zero probabilities.
- No overall forensic confidence, combined manipulation probability, or combined authenticity score exists.

## Artifact endpoints

`GET /api/v1/investigations/{investigation_id}/artifacts/{artifact_key}`

Current keys can include `lineage`, `social_graph`, and `dissemination_graph`. The response schema contains URLs, not filesystem paths. Unknown artifacts, cross-investigation lookups, and traversal attempts return 404.

`GET /api/v1/investigations/{investigation_id}` returns a completed in-memory result during the current process lifetime.

## Health and capabilities

| Endpoint | Purpose |
|---|---|
| `GET http://localhost:8000/health` | Gateway liveness without loading models |
| `GET http://localhost:8000/api/v1/capabilities` | Live frontend capability view based on child health and local runtime |
| `GET http://localhost:8001/health` | Detection assets, runtime provider state, image/video readiness |
| `GET http://localhost:8002/health` | PRNU process, real reference availability, image-only capability |

## Local commands

Start each process in its own terminal.

```bash
cd Detection
uvicorn service.main:app --env-file ../.env --host 0.0.0.0 --port 8001
```

```bash
cd prnu-device-attribution-api-handoff
python -m prnu_attribution.api --host 0.0.0.0 --port 8002 --dataset dataset --model-dir models
```

```bash
cd CPH
uvicorn src.api.main:app --env-file ../.env --host 0.0.0.0 --port 8000
```

For local processes, set `SHARED_MEDIA_DIR` to a common absolute directory and set `DETECTION_ALLOWED_MEDIA_ROOT` and `SHARED_MEDIA_ROOT` to the same path.

## Docker Compose

From the repository root:

```bash
docker compose --env-file .env -f CPH/docker-compose.yml up --build
```

The gateway has read/write access to `shared_media`. Detection and PRNU mount the same evidence volume read-only. Checkpoints, model assets, datasets, secrets, and generated investigations are not baked into the images.

## CLI

With child services running:

```bash
python CPH/scripts/investigate.py \
  --media "/absolute/path/to/evidence.mp4" \
  --case-id "CASE-DEMO-001" \
  --output "outputs/CASE-DEMO-001.json"
```

## Notes for the future UI developer

1. Upload one file only to `/api/v1/investigate/upload`.
2. Render each module status independently.
3. Treat `FAILED`, `TIMEOUT`, and `NOT_APPLICABLE` as distinct states.
4. Do not derive authenticity from a missing modality or successful AV synchronization.
5. Do not combine Detection, device, and source scores.
6. Display Detection's native image and video taxonomies separately.
7. Use only returned artifact URLs. Never construct a server path.
8. Check `/api/v1/capabilities` before enabling media-specific UI actions.
9. Swagger and OpenAPI remain available at `/docs` and `/openapi.json`.

## Validation scope

Focused tests mock model subprocesses and external forensic services. They validate upload safety, deterministic SHA-256, case and investigation IDs, concurrent fan-out, partial failures, timeouts, native result retention, video PRNU applicability, malformed JSON handling, artifact isolation, safe subprocess argv usage, and OpenAPI generation. Real model, GPU, paid API, and live reverse-search execution require separate laptop validation.
