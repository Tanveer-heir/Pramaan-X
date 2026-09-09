"""Mocked contract tests for independent multipart subsystem routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.api import standalone_routes
from src.common.schemas import OriginTracingEvidence


def test_detection_upload_returns_native_payload_and_cleans_temp_file(monkeypatch):
    native = {"schema_version": "native_detection", "assessment": {"label": "SUSPICIOUS"}}
    seen = []

    def fake_detection(media_path, _output_path):
        seen.append(media_path)
        return native

    monkeypatch.setattr(standalone_routes, "run_detection", fake_detection)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/detection/analyze",
            files={"file": ("evidence.jpg", b"image-bytes", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json() == native
    assert seen and not seen[0].exists()


def test_prnu_rejects_video_without_running_inference(monkeypatch):
    called = False

    def fake_prnu(_media_path):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(standalone_routes, "run_prnu", fake_prnu)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/prnu/analyze",
            files={"file": ("evidence.mp4", b"video-bytes", "video/mp4")},
        )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "NOT_APPLICABLE"
    assert called is False


def test_prnu_upload_returns_native_payload_and_cleans_temp_file(monkeypatch):
    native = {"device_attribution": {"prediction": "device_A", "confidence": 0.82}}
    seen = []

    def fake_prnu(media_path):
        seen.append(media_path)
        return native

    monkeypatch.setattr(standalone_routes, "run_prnu", fake_prnu)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/prnu/analyze",
            files={"file": ("evidence.png", b"image-bytes", "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == native
    assert seen and not seen[0].exists()


def test_cph_upload_returns_native_evidence_and_cleans_temp_file(monkeypatch):
    evidence = OriginTracingEvidence(circulation_summary={"source": "native-cph"})
    seen = []

    async def fake_cph(media_path):
        seen.append(media_path)
        return evidence

    monkeypatch.setattr(standalone_routes, "run_cph", fake_cph)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cph/trace",
            files={"file": ("evidence.webm", b"video-bytes", "video/webm")},
        )

    assert response.status_code == 200
    assert response.json() == evidence.model_dump(mode="json")
    assert seen and not seen[0].exists()
