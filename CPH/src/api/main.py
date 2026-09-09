"""
FastAPI Server & Investigator Dashboard Backend (Features 6–10).
"""

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import asyncio

from src.common.schemas import CombinedEvidenceReport, OriginTracingEvidence
from src.common.logger import logger
from src.common.config import settings
from src.gateway.artifacts import ArtifactRegistry
from src.gateway.config import GatewayConfig
from src.gateway.media import MediaValidationError, store_path_once, store_stream_once
from src.gateway.models import CapabilitiesResponse, InvestigationResponse
from src.gateway.orchestrator import UnifiedInvestigationOrchestrator, new_investigation_id
from src.api.standalone_routes import router as standalone_router

app = FastAPI(
    title="Chandigarh Police Hackathon Section 2 Forensic Gateway",
    description="Orchestrator & Gateway for AI Content Detection, PRNU Forensics, and A* Visual Attribution",
    version="3.0.0"
)

gateway_config = GatewayConfig.from_env()
gateway_config.shared_media_dir.mkdir(parents=True, exist_ok=True)
artifact_registry = ArtifactRegistry(gateway_config.investigation_output_dir)
gateway_orchestrator = UnifiedInvestigationOrchestrator(
    config=gateway_config,
    artifact_registry=artifact_registry,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(gateway_config.cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(standalone_router)

# In-memory case cache for demonstration
cases_db: Dict[str, Any] = {}


def _new_astar_engine(output_graph_path: str):
    from src.research.astar_attribution.astar_engine import AStarVisualAttributionEngine

    return AStarVisualAttributionEngine(output_graph_path=output_graph_path)

class AnalysisRequest(BaseModel):
    media_path: str = Field(..., description="Local path or URL to submitted media")
    case_id: Optional[str] = Field(None, description="Optional investigator case ID")
    section1_evidence: Optional[Dict[str, Any]] = Field(None, description="Section 1 detection payload")


class AttributionRequest(BaseModel):
    media_path: str = Field(..., description="Local path or URL to submitted media")


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    media_path: Optional[str] = Field(None, description="Developer-only local media path")
    file_path: Optional[str] = Field(None, description="Developer-only local media path alias")
    case_id: Optional[str] = Field(None, description="Optional investigator case ID")


@app.get("/health", tags=["System"])
async def health_check():
    """Cheap liveness probe that does not load forensic models."""
    return {
        "status": "ok",
        "process_alive": True,
        "service": "cph-unified-gateway",
        "environment": settings.ENVIRONMENT,
    }


@app.get("/api/v1/capabilities", response_model=CapabilitiesResponse, tags=["System"])
async def capabilities():
    """Report live child-service capabilities for frontend routing."""
    return await gateway_orchestrator.capabilities()


# ==============================================================================
# Unified Gateway Fan-Out Endpoints (§2.4 Multi-Service Orchestrator)
# ==============================================================================

@app.post("/api/v1/investigate", response_model=InvestigationResponse, tags=["Gateway Orchestrator"])
@app.post("/investigate", response_model=InvestigationResponse, tags=["Gateway Orchestrator"])
async def run_investigation_json(request: InvestigationRequest):
    """Disabled-by-default developer endpoint for trusted local paths."""
    if os.getenv("ALLOW_LOCAL_PATH_ENDPOINTS", "false").lower() not in {"1", "true", "yes"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "LOCAL_PATH_ENDPOINT_DISABLED", "message": "Use the upload endpoint."},
        )
    target = request.media_path or request.file_path
    if not target:
        raise HTTPException(
            status_code=400,
            detail={"code": "MEDIA_PATH_REQUIRED", "message": "Provide media_path or file_path."},
        )
    if target.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail={"code": "REMOTE_PATH_REJECTED", "message": "Remote URLs are not accepted here."},
        )
    investigation_id = new_investigation_id()
    try:
        stored = await asyncio.to_thread(
            store_path_once,
            Path(target),
            investigation_id,
            gateway_config.shared_media_dir,
            gateway_config.max_upload_bytes,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"code": "MEDIA_NOT_FOUND", "message": "Media file not found."})
    except MediaValidationError as exc:
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA", "message": str(exc)})
    report = await gateway_orchestrator.investigate(stored, request.case_id)
    cases_db[report.case_id] = report
    cases_db[report.investigation_id] = report
    return report


@app.post("/api/v1/investigate/upload", response_model=InvestigationResponse, tags=["Gateway Orchestrator"])
@app.post("/investigate/upload", response_model=InvestigationResponse, tags=["Gateway Orchestrator"])
async def run_investigation_upload(
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(default=None),
):
    """Save one browser upload once, then fan out over the immutable evidence."""
    investigation_id = new_investigation_id()
    try:
        stored = await asyncio.to_thread(
            store_stream_once,
            file.file,
            file.filename,
            investigation_id,
            gateway_config.shared_media_dir,
            gateway_config.max_upload_bytes,
        )
    except MediaValidationError as exc:
        status_code = 413 if "maximum size" in str(exc) else 415
        code = "UPLOAD_TOO_LARGE" if status_code == 413 else "UNSUPPORTED_MEDIA"
        raise HTTPException(status_code=status_code, detail={"code": code, "message": str(exc)})
    finally:
        await file.close()
    report = await gateway_orchestrator.investigate(stored, case_id)
    cases_db[report.case_id] = report
    cases_db[report.investigation_id] = report
    return report


@app.get(
    "/api/v1/investigations/{investigation_id}",
    response_model=InvestigationResponse,
    tags=["Gateway Orchestrator"],
)
async def get_investigation(investigation_id: str):
    report = cases_db.get(investigation_id)
    if not isinstance(report, InvestigationResponse):
        raise HTTPException(status_code=404, detail={"code": "INVESTIGATION_NOT_FOUND", "message": "Investigation not found."})
    return report


@app.get("/api/v1/investigations/{investigation_id}/artifacts/{artifact_key}", tags=["Artifacts"])
async def get_investigation_artifact(investigation_id: str, artifact_key: str):
    entry = artifact_registry.resolve(investigation_id, artifact_key)
    if entry is None:
        raise HTTPException(status_code=404, detail={"code": "ARTIFACT_NOT_FOUND", "message": "Artifact not found."})
    return FileResponse(entry.path, media_type=entry.reference.media_type)


# ==============================================================================
# Full Pipeline Endpoints
# ==============================================================================

@app.post("/api/v1/cases/analyze", response_model=CombinedEvidenceReport, status_code=status.HTTP_200_OK, tags=["Investigation"])
async def run_analysis(request: AnalysisRequest):
    """Executes full Section 2 pipeline: Provenance Check + Origin Tracing + Chain of Custody."""
    try:
        from src.pipeline.orchestrator import Section2Orchestrator

        report = await Section2Orchestrator().analyze(
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
        from src.source_attribution.pipeline import SourceAttributionPipeline

        evidence = await SourceAttributionPipeline().execute(media_path=request.media_path)
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
    Isolates a Patient Zero candidate and returns native heuristic source rankings.
    """
    try:
        engine = _new_astar_engine("data/graphs/astar_lineage_tree.html")
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
