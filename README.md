# Pramaan-X

Pramaan-X is a three-service digital-forensics prototype for manipulation detection, device attribution, and source tracing. The CPH service is the only UI-facing gateway. It accepts one browser upload and returns one versioned investigation record while keeping each forensic subsystem independent.

## Architecture

```text
Future UI
   |
   | POST /api/v1/investigate/upload
   v
CPH unified gateway :8000
   | concurrent fan-out over one immutable evidence object
   +-- Detection :8001
   +-- PRNU device attribution :8002, images only
   +-- CPH source attribution, local to gateway
   |
   v
pramaan_x_investigation_v1 JSON
```

| Component | Responsibility | Score semantics |
|---|---|---|
| `Detection/` | Image triage and four-class video manipulation detection | Image confidence is categorical. Video softmax values are uncalibrated model scores. |
| `prnu-device-attribution-api-handoff/` | Known-reference PRNU matching and metadata fallback for still images | Independent ranked device evidence. Unknown devices can remain inconclusive. |
| `CPH/` | Source tracing, Patient Zero candidate discovery, lineage artifacts, custody, and unified orchestration | Source ranking values are heuristic unless explicitly documented otherwise. |

The gateway never changes the Detection verdict, never combines these channels into an overall confidence, and never treats a missing module as negative evidence.

## Primary API

```bash
curl -X POST http://localhost:8000/api/v1/investigate/upload \
  -F "file=@/absolute/path/to/evidence.jpg" \
  -F "case_id=CASE-DEMO-001"
```

Swagger is available at `http://localhost:8000/docs`. The exact schema, status behavior, examples, artifact routes, and frontend handoff notes are in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

## Docker quick start

The default Detection image is CPU-based. Real GPU inference is normally run from the validated local Detection environment.

```bash
cp .env.example .env
docker compose --env-file .env -f CPH/docker-compose.yml up --build
```

Service health:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/capabilities
curl http://localhost:8001/health
curl http://localhost:8002/health
```

The repository does not bake datasets, credentials, local model assets, or investigation uploads into container images. See [`Detection/README.md`](Detection/README.md) for required video assets and the accepted checkpoint contract.

## CLI smoke path

From the repository root, with the CPH environment active and child services running:

```bash
python CPH/scripts/investigate.py \
  --media "/absolute/path/to/evidence.mp4" \
  --case-id "CASE-DEMO-001" \
  --output "outputs/CASE-DEMO-001.json"
```

The CLI calls the same orchestration class as the HTTP endpoint. It does not implement a second forensic pipeline.

## Validation boundary

Gateway and service-contract tests use mocks and synthetic media fixtures. They prove routing, validation, concurrency, status aggregation, path safety, and schema preservation. They do not prove real ML accuracy or external source-search performance. Real media validation requires the Detection assets, credentials, PRNU reference set, and source-attribution runtime described in the subsystem documentation.
