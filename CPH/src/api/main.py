"""
FastAPI Server & Investigator Dashboard Backend (Features 6–10).
"""

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone
import os
import asyncio
import httpx
import uuid
import shutil
import hashlib
import time
import urllib.request

from src.common.schemas import CombinedEvidenceReport, OriginTracingEvidence
from src.pipeline.orchestrator import Section2Orchestrator
from src.source_attribution.pipeline import SourceAttributionPipeline
from src.research.astar_attribution.astar_engine import AStarVisualAttributionEngine
from src.common.logger import logger
from src.common.config import settings

app = FastAPI(
    title="Chandigarh Police Hackathon — Section 2 Forensic Gateway",
    description="Orchestrator & Gateway for AI Content Detection, PRNU Forensics, and A* Visual Attribution",
    version="2.0.0"
)

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-detection:8001")
PRNU_SERVICE_URL = os.getenv("PRNU_SERVICE_URL", "http://prnu-forensics:8002")
SHARED_MEDIA_DIR = Path(os.getenv("SHARED_MEDIA_DIR", "data/shared_media"))
SHARED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

orchestrator = Section2Orchestrator()
source_attribution_pipeline = SourceAttributionPipeline()

# In-memory case cache for demonstration
cases_db: Dict[str, Any] = {}

class AnalysisRequest(BaseModel):
    media_path: str = Field(..., description="Local path or URL to submitted media")
    case_id: Optional[str] = Field(None, description="Optional investigator case ID")
    section1_evidence: Optional[Dict[str, Any]] = Field(None, description="Section 1 detection payload")


class AttributionRequest(BaseModel):
    media_path: str = Field(..., description="Local path or URL to submitted media")


class InvestigationRequest(BaseModel):
    media_path: Optional[str] = Field(None, description="Local path or remote URL to submitted media")
    file_path: Optional[str] = Field(None, description="Shared volume path to submitted media")
    case_id: Optional[str] = Field(None, description="Optional investigator case ID")


def _prepare_shared_media(target_input: str) -> tuple[Path, str, str]:
    """
    Ensures media is available in the shared volume.
    Returns: (local_path, shared_container_path, sha256_hash)
    """
    target_str = str(target_input)
    if target_str.startswith(("http://", "https://")):
        ext = Path(target_str.split("?")[0]).suffix or ".jpg"
        unique_name = f"download_{uuid.uuid4().hex[:12]}{ext}"
        local_dest = SHARED_MEDIA_DIR / unique_name
        req = urllib.request.Request(target_str, headers={"User-Agent": "CPH-Gateway/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(local_dest, "wb") as f:
            f.write(resp.read())
    else:
        src = Path(target_str)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"Local media file not found: {target_str}")
        if SHARED_MEDIA_DIR.resolve() not in src.resolve().parents and src.resolve().parent != SHARED_MEDIA_DIR.resolve():
            unique_name = f"{uuid.uuid4().hex[:8]}_{src.name}"
            local_dest = SHARED_MEDIA_DIR / unique_name
            shutil.copy2(src, local_dest)
        else:
            local_dest = src
            unique_name = src.name

    hasher = hashlib.sha256()
    with open(local_dest, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    sha256_hex = hasher.hexdigest()

    shared_container_path = f"/data/shared_media/{unique_name}"
    return local_dest, shared_container_path, sha256_hex


async def execute_gateway_fanout(target_input: str, case_id: Optional[str] = None) -> Dict[str, Any]:
    local_path, shared_container_path, sha256_hex = _prepare_shared_media(target_input)
    effective_case_id = case_id or f"CASE-{int(time.time())}-{sha256_hex[:8]}"

    # In Docker container, shared media is mounted at /data/shared_media
    path_for_nodes = shared_container_path if os.path.exists("/data/shared_media") else str(local_path.resolve())

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Node 1: AI Content Detection (Pramaan-X)
        ai_task = client.post(
            f"{AI_SERVICE_URL}/detect",
            json={"file_path": path_for_nodes, "media_path": path_for_nodes}
        )

        # 2. Node 2: PRNU / Device Attribution
        prnu_task = client.post(
            f"{PRNU_SERVICE_URL}/analyse",
            json={"file_path": path_for_nodes, "media_path": path_for_nodes}
        )

        # 3. Node 3: A* Visual Source Attribution
        engine = AStarVisualAttributionEngine(
            output_graph_path="data/graphs/astar_lineage_tree.html"
        )
        source_task = engine.trace_origin_async(str(local_path.resolve()))

        # Trigger all 3 concurrently with exception resilience
        ai_res, prnu_res, source_res = await asyncio.gather(
            ai_task, prnu_task, source_task, return_exceptions=True
        )

    # Process AI Detection response
    if isinstance(ai_res, Exception):
        logger.warning("gateway.ai_detection_offline", error=str(ai_res))
        ai_payload = {"status": "offline", "error": str(ai_res), "service": AI_SERVICE_URL}
    elif ai_res.status_code != 200:
        ai_payload = {"status": "error", "code": ai_res.status_code, "detail": ai_res.text}
    else:
        try:
            ai_payload = ai_res.json()
        except Exception:
            ai_payload = {"status": "error", "raw": ai_res.text}

    # Process PRNU Forensics response
    if isinstance(prnu_res, Exception):
        logger.warning("gateway.prnu_offline", error=str(prnu_res))
        prnu_payload = {"status": "offline", "error": str(prnu_res), "service": PRNU_SERVICE_URL}
    elif prnu_res.status_code != 200:
        prnu_payload = {"status": "error", "code": prnu_res.status_code, "detail": prnu_res.text}
    else:
        try:
            prnu_payload = prnu_res.json()
        except Exception:
            prnu_payload = {"status": "error", "raw": prnu_res.text}

    # Process Source Attribution response
    if isinstance(source_res, Exception):
        logger.error("gateway.source_attribution_failed", error=str(source_res))
        source_payload = {"status": "error", "error": str(source_res)}
    else:
        source_payload = source_res

    # Build consolidated unified report
    report = {
        "case_id": effective_case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_media": {
            "filename": local_path.name,
            "sha256": sha256_hex,
            "local_path": str(local_path),
            "shared_path": shared_container_path,
        },
        "ai_content_detection": ai_payload,
        "prnu_sensor_forensics": prnu_payload,
        "source_attribution": {
            "target": source_payload.get("target_media") or source_payload.get("target") if isinstance(source_payload, dict) else None,
            "patient_zero": source_payload.get("patient_zero") if isinstance(source_payload, dict) else None,
            "candidate_sources": source_payload.get("candidate_sources", []) if isinstance(source_payload, dict) else [],
            "graph_html_path": source_payload.get("graph_html_path", "data/graphs/astar_lineage_tree.html") if isinstance(source_payload, dict) else None,
            "iterations": source_payload.get("iterations", 0) if isinstance(source_payload, dict) else 0,
            "execution_time_sec": source_payload.get("execution_time_sec", 0.0) if isinstance(source_payload, dict) else 0.0,
        },
        "status": "completed",
    }
    cases_db[effective_case_id] = report
    return report


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "cph-section2-gateway", "environment": settings.ENVIRONMENT}


# ==============================================================================
# Unified Gateway Fan-Out Endpoints (§2.4 Multi-Service Orchestrator)
# ==============================================================================

@app.post("/api/v1/investigate", tags=["Gateway Orchestrator"])
@app.post("/investigate", tags=["Gateway Orchestrator"])
async def run_investigation_json(request: InvestigationRequest):
    """
    Gateway Fan-Out (Concurrent Execution across 3 nodes).
    Triggers Node 1 (AI Detection), Node 2 (PRNU Forensics), and Node 3 (A* Attribution)
    in parallel using asyncio.gather.
    """
    target = request.media_path or request.file_path
    if not target:
        raise HTTPException(status_code=400, detail="Must provide 'media_path' or 'file_path'")
    return await execute_gateway_fanout(target, request.case_id)


@app.post("/api/v1/investigate/upload", tags=["Gateway Orchestrator"])
@app.post("/investigate/upload", tags=["Gateway Orchestrator"])
async def run_investigation_upload(file: UploadFile = File(...), case_id: Optional[str] = None):
    """Uploads media file once to shared volume and triggers full 3-node investigation concurrently."""
    ext = Path(file.filename or "media.jpg").suffix or ".jpg"
    unique_name = f"upload_{uuid.uuid4().hex[:12]}{ext}"
    local_dest = SHARED_MEDIA_DIR / unique_name
    with open(local_dest, "wb") as f:
        f.write(await file.read())
    return await execute_gateway_fanout(str(local_dest), case_id)


# ==============================================================================
# Full Pipeline Endpoints
# ==============================================================================

@app.post("/api/v1/cases/analyze", response_model=CombinedEvidenceReport, status_code=status.HTTP_200_OK, tags=["Investigation"])
async def run_analysis(request: AnalysisRequest):
    """Executes full Section 2 pipeline: Provenance Check + Origin Tracing + Chain of Custody."""
    try:
        report = await orchestrator.analyze(
            media_path=request.media_path,
            case_id=request.case_id,
            section1_evidence=request.section1_evidence
        )
        cases_db[report.case_id] = report
        return report
    except Exception as e:
        logger.error("api.analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}"
        )


@app.get("/api/v1/cases/{case_id}/report", response_model=CombinedEvidenceReport, tags=["Investigation"])
async def get_case_report(case_id: str):
    """Retrieves an existing analysis report without re-running computation (Feature 6)."""
    if case_id not in cases_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return cases_db[case_id]


# ==============================================================================
# Source Attribution Dedicated Endpoints
# ==============================================================================

@app.post("/api/v1/source-attribution/analyze", response_model=OriginTracingEvidence, tags=["Source Attribution"])
async def run_source_attribution(request: AttributionRequest):
    """Directly executes Source Attribution & Origin Tracing (§2.4)."""
    try:
        evidence = await source_attribution_pipeline.execute(media_path=request.media_path)
        return evidence
    except Exception as e:
        logger.error("api.attribution_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Source attribution error: {str(e)}"
        )


@app.post("/api/v1/source-attribution", tags=["A* Visual Attribution"])
@app.post("/source-attribution", tags=["A* Visual Attribution"])
async def run_astar_attribution(request: AttributionRequest):
    """
    Executes A* Heuristic Traversal Engine (§2.4b).
    Isolates Patient Zero and returns ranked source candidates with match probabilities.
    """
    try:
        engine = AStarVisualAttributionEngine(
            output_graph_path="data/graphs/astar_lineage_tree.html"
        )
        results = await asyncio.to_thread(engine.trace_origin, request.media_path)
        return results
    except Exception as e:
        logger.error("api.astar_attribution_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A* source attribution error: {str(e)}"
        )


@app.get("/api/v1/source-attribution/graphs/astar", response_class=HTMLResponse, tags=["A* Visual Attribution"])
@app.get("/graphs/astar", response_class=HTMLResponse, tags=["A* Visual Attribution"])
async def get_astar_lineage_graph_html():
    """Serves the interactive PyVis A* Lineage Tree in browser."""
    path = "data/graphs/astar_lineage_tree.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="A* graph not generated yet. Run attribution first.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/v1/source-attribution/graphs/social", response_class=HTMLResponse, tags=["Source Attribution"])
async def get_social_graph_html():
    """Serves the interactive PyVis Account-Level Social Amplification Graph in browser."""
    path = "data/graphs/social_graph.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Graph not generated yet. Run an analysis first.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/v1/source-attribution/graphs/dissemination", response_class=HTMLResponse, tags=["Source Attribution"])
async def get_dissemination_graph_html():
    """Serves the interactive PyVis Content Dissemination Timeline Tree in browser."""
    path = "data/graphs/dissemination_graph.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Graph not generated yet. Run an analysis first.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
