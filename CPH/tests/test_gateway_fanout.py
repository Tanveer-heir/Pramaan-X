"""Contract tests for the unified upload and aggregation boundary.

All forensic modules are mocked. These tests do not execute ML or external APIs.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import src.api.main as api
from src.gateway.artifacts import ArtifactRegistry
from src.gateway.config import GatewayConfig
from src.gateway.orchestrator import (
    ExecutionOutput,
    ModuleExecutionError,
    UnifiedInvestigationOrchestrator,
)


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 4), color=(20, 30, 40)).save(stream, format="PNG")
    return stream.getvalue()


def mp4_bytes() -> bytes:
    return b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"0" * 32


def detection_result(media_type: str = "image") -> dict:
    if media_type == "video":
        return {
            "schema_version": "pramaan_x_prediction_v2",
            "selected_class": "AUDIO_MANIPULATION",
            "softmax_probabilities": {"AUDIO_MANIPULATION": 0.8},
            "branch_evidence": {"visual_available": True, "audio_available": True},
            "counterfactual_modality_analysis": {"method": "leave_one_available_modality_out"},
            "temporal_evidence": {"dense_av": {"segments": []}},
            "raw_video": {"path": "/data/shared_media/private/video.mp4"},
        }
    return {
        "schema_version": "pramaan_x_image_analysis_v1",
        "media_type": "image",
        "assessment": {"label": "SUSPICIOUS", "confidence_level": "MEDIUM"},
        "visual_findings": [],
        "supporting_signals": {},
        "input": {"path": "/data/shared_media/private/image.png"},
        "limitations": ["Not a calibrated forensic probability."],
    }


def configure_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, timeout: float = 1.0, runners=None):
    config = GatewayConfig(
        detection_service_url="http://detection.test",
        prnu_service_url="http://prnu.test",
        shared_media_dir=tmp_path / "shared",
        investigation_output_dir=tmp_path / "artifacts",
        detection_timeout_sec=timeout,
        prnu_timeout_sec=timeout,
        source_attribution_timeout_sec=timeout,
        health_timeout_sec=timeout,
        max_upload_bytes=1024 * 1024,
        cors_allowed_origins=("http://localhost:3000",),
    )
    registry = ArtifactRegistry(config.investigation_output_dir)

    async def detection(media, _):
        return ExecutionOutput(detection_result(media.public.media_type.value))

    async def prnu(_media, _artifact_dir):
        return ExecutionOutput({
            "ok": True,
            "result": {
                "device_attribution": {
                    "prediction": "OnePlus 12R",
                    "display_name": "OnePlus 12R",
                    "primary_method": "prnu",
                    "confidence": 0.81,
                }
            },
        })

    async def source(_media, artifact_dir):
        graph = artifact_dir / "lineage.html"
        graph.write_text("<html>lineage</html>", encoding="utf-8")
        return ExecutionOutput(
            {"patient_zero": {"domain": "example.com"}, "graph_html_path": str(graph)},
            {"lineage": graph},
        )

    selected = runners or (detection, prnu, source)
    orchestrator = UnifiedInvestigationOrchestrator(
        config,
        registry,
        detection_runner=selected[0],
        prnu_runner=selected[1],
        source_runner=selected[2],
    )
    monkeypatch.setattr(api, "gateway_config", config)
    monkeypatch.setattr(api, "artifact_registry", registry)
    monkeypatch.setattr(api, "gateway_orchestrator", orchestrator)
    api.cases_db.clear()
    return TestClient(api.app), orchestrator


def test_valid_image_upload_hash_case_id_native_results_and_artifact(tmp_path, monkeypatch):
    client, _ = configure_gateway(tmp_path, monkeypatch)
    response = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("../../unsafe name.png", png_bytes(), "image/png")},
        data={"case_id": "CASE-IMAGE-001"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "pramaan_x_investigation_v1"
    assert data["case_id"] == "CASE-IMAGE-001"
    assert data["investigation_id"].startswith("INV-")
    assert data["media"]["filename"] == "unsafe_name.png"
    assert len(data["media"]["sha256"]) == 64
    assert data["status"] == "COMPLETED"
    assert data["modules"]["detection"]["result"]["assessment"]["label"] == "SUSPICIOUS"
    assert "overall_forensic_confidence" not in str(data)
    assert data["artifacts"]["lineage"]["url"].endswith("/artifacts/lineage")
    assert "/data/shared_media" not in str(data)
    assert len(list((tmp_path / "shared").rglob("evidence.png"))) == 1

    artifact = client.get(data["artifacts"]["lineage"]["url"])
    assert artifact.status_code == 200
    assert b"lineage" in artifact.content
    assert client.get("/api/v1/investigations/INV-OTHER/artifacts/lineage").status_code == 404
    assert client.get(f"/api/v1/investigations/{data['investigation_id']}/artifacts/unknown").status_code == 404
    assert client.get(f"/api/v1/investigations/{data['investigation_id']}/artifacts/%2e%2e").status_code == 404


def test_video_routes_detection_and_source_but_not_prnu(tmp_path, monkeypatch):
    called = {"prnu": 0}

    async def detection(_media, _artifact_dir):
        return ExecutionOutput(detection_result("video"))

    async def prnu(_media, _artifact_dir):
        called["prnu"] += 1
        return ExecutionOutput({"ok": True, "result": {}})

    async def source(_media, _artifact_dir):
        return ExecutionOutput({"earliest_candidate": {"domain": "example.org"}})

    client, _ = configure_gateway(tmp_path, monkeypatch, runners=(detection, prnu, source))
    response = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("clip.mp4", mp4_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert called["prnu"] == 0
    assert data["media"]["media_type"] == "video"
    assert data["modules"]["device_attribution"]["status"] == "NOT_APPLICABLE"
    assert data["modules"]["detection"]["result"]["selected_class"] == "AUDIO_MANIPULATION"
    assert "counterfactual_modality_analysis" in data["modules"]["detection"]["result"]
    assert "temporal_evidence" in data["modules"]["detection"]["result"]


def test_modules_start_concurrently(tmp_path, monkeypatch):
    starts = []

    def runner(result):
        async def run(_media, _artifact_dir):
            starts.append(perf_counter())
            await asyncio.sleep(0.04)
            return ExecutionOutput(result)
        return run

    runners = (
        runner(detection_result()),
        runner({"ok": True, "result": {"device_attribution": {"primary_method": "inconclusive"}}}),
        runner({"candidate_sources": []}),
    )
    client, _ = configure_gateway(tmp_path, monkeypatch, runners=runners)
    assert client.post(
        "/api/v1/investigate/upload",
        files={"file": ("image.png", png_bytes(), "image/png")},
    ).status_code == 200
    assert len(starts) == 3
    assert max(starts) - min(starts) < 0.03


@pytest.mark.parametrize("failure_count, expected", [(1, "PARTIAL"), (3, "FAILED")])
def test_module_failure_status_and_no_stack_trace(tmp_path, monkeypatch, failure_count, expected):
    async def success_detection(_media, _artifact_dir):
        return ExecutionOutput(detection_result())

    async def success_prnu(_media, _artifact_dir):
        return ExecutionOutput({"ok": True, "result": {"device_attribution": {"primary_method": "inconclusive"}}})

    async def success_source(_media, _artifact_dir):
        return ExecutionOutput({"candidate_sources": []})

    async def fail(_media, _artifact_dir):
        raise RuntimeError("SECRET_TRACEBACK_TOKEN")

    runners = [success_detection, success_prnu, success_source]
    for index in range(failure_count):
        runners[index] = fail
    client, _ = configure_gateway(tmp_path, monkeypatch, runners=tuple(runners))
    response = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("image.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == expected
    assert "SECRET_TRACEBACK_TOKEN" not in response.text


def test_module_timeout_and_malformed_result_are_isolated(tmp_path, monkeypatch):
    async def slow(_media, _artifact_dir):
        await asyncio.sleep(0.1)
        return ExecutionOutput({})

    async def malformed(_media, _artifact_dir):
        return ExecutionOutput("not-an-object")  # type: ignore[arg-type]

    async def source(_media, _artifact_dir):
        return ExecutionOutput({"candidate_sources": []})

    client, _ = configure_gateway(tmp_path, monkeypatch, timeout=0.01, runners=(slow, malformed, source))
    response = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("image.png", png_bytes(), "image/png")},
    )
    data = response.json()
    assert data["status"] == "PARTIAL"
    assert data["modules"]["detection"]["status"] == "TIMEOUT"
    assert data["modules"]["device_attribution"]["status"] == "FAILED"
    assert data["modules"]["source_attribution"]["status"] == "COMPLETED"


def test_unsupported_media_returns_415_without_creating_investigation(tmp_path, monkeypatch):
    client, _ = configure_gateway(tmp_path, monkeypatch)
    response = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("payload.txt", b"not media", "text/plain")},
    )
    assert response.status_code == 415
    assert "Traceback" not in response.text


def test_generated_case_id_and_deterministic_sha256(tmp_path, monkeypatch):
    client, _ = configure_gateway(tmp_path, monkeypatch)
    content = png_bytes()
    first = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("one.png", content, "image/png")},
    ).json()
    second = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("two.png", content, "image/png")},
    ).json()
    assert first["case_id"].startswith("CASE-")
    assert first["media"]["sha256"] == second["media"]["sha256"]
    assert first["investigation_id"] != second["investigation_id"]


def test_explicit_module_error_uses_structured_reason(tmp_path, monkeypatch):
    async def rejected(_media, _artifact_dir):
        raise ModuleExecutionError("PRNU_MODEL_UNAVAILABLE", "Reference set is unavailable.")

    async def detection(_media, _artifact_dir):
        return ExecutionOutput(detection_result())

    async def source(_media, _artifact_dir):
        return ExecutionOutput({})

    client, _ = configure_gateway(tmp_path, monkeypatch, runners=(detection, rejected, source))
    data = client.post(
        "/api/v1/investigate/upload",
        files={"file": ("image.png", png_bytes(), "image/png")},
    ).json()
    module_result = data["modules"]["device_attribution"]
    assert module_result["status"] == "FAILED"
    assert module_result["reason"] == "PRNU_MODEL_UNAVAILABLE"
    assert module_result["result"] is None


def test_openapi_exposes_versioned_upload_contract(tmp_path, monkeypatch):
    client, _ = configure_gateway(tmp_path, monkeypatch)
    schema = client.get("/openapi.json").json()
    assert "/api/v1/investigate/upload" in schema["paths"]
    operation = schema["paths"]["/api/v1/investigate/upload"]["post"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/InvestigationResponse")
