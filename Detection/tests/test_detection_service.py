"""Mocked HTTP and subprocess tests for the Detection service boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import pytest

try:
    from Detection.service import main
except ModuleNotFoundError:
    from service import main


def valid_image() -> dict:
    return {
        "schema_version": "pramaan_x_image_analysis_v1",
        "media_type": "image",
        "assessment": {"label": "SUSPICIOUS", "confidence_level": "LOW"},
        "visual_findings": [],
        "supporting_signals": {},
    }


def valid_video() -> dict:
    return {
        "schema_version": "pramaan_x_prediction_v2",
        "selected_class": "REAL",
        "branch_evidence": {},
        "softmax_probabilities": {"REAL": 0.7},
    }


@pytest.fixture
def service(tmp_path, monkeypatch):
    allowed = tmp_path / "shared"
    allowed.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    settings = replace(
        main.SETTINGS,
        allowed_media_root=allowed,
        output_root=output,
        inference_timeout_sec=0.05,
    )
    monkeypatch.setattr(main, "SETTINGS", settings)
    return TestClient(main.app), allowed


def test_image_and_video_route_to_accepted_commands(service, monkeypatch):
    client, allowed = service
    image = allowed / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    video = allowed / "video.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 32)
    calls = []

    async def fake(command, _output):
        calls.append(command)
        return valid_image() if "analyse_pramaan_x_image.py" in " ".join(command) else valid_video()

    monkeypatch.setattr(main, "run_inference", fake)
    image_response = client.post("/api/v1/detect", json={"media_path": str(image)})
    video_response = client.post("/detect", json={"media_path": str(video)})
    assert image_response.status_code == 200
    assert video_response.status_code == 200
    assert "analyse_pramaan_x_image.py" in " ".join(calls[0])
    assert "run_pramaan_x.py" in " ".join(calls[1])
    assert image_response.json() == valid_image()
    assert video_response.json() == valid_video()


def test_unsupported_and_outside_root_are_rejected(service):
    client, allowed = service
    unsupported = allowed / "payload.jpg"
    unsupported.write_bytes(b"not an image")
    assert client.post("/detect", json={"media_path": str(unsupported)}).status_code == 415
    assert client.post("/detect", json={"media_path": "/etc/passwd"}).status_code == 403


def test_invalid_detection_schema_is_502(service, monkeypatch):
    client, allowed = service
    image = allowed / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    monkeypatch.setattr(main, "run_inference", AsyncMock(return_value={"assessment": {"label": "MADE_UP"}}))
    response = client.post("/detect", json={"media_path": str(image)})
    assert response.status_code == 502
    assert "Traceback" not in response.text


class FakeProcess:
    def __init__(self, returncode=0, output_path: Path | None = None, body: str | None = None, delay=0):
        self.returncode = returncode
        self.output_path = output_path
        self.body = body
        self.delay = delay
        self.killed = False

    async def communicate(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.output_path is not None and self.body is not None:
            self.output_path.write_text(self.body, encoding="utf-8")
        return b"stdout", b"private stderr detail"

    def kill(self):
        self.killed = True
        self.delay = 0


@pytest.mark.asyncio
async def test_subprocess_invocation_is_argv_only_and_valid_json_passthrough(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    process = FakeProcess(output_path=output, body=json.dumps(valid_image()))
    captured = {}

    async def create(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", create)
    result = await main.run_inference(["python", "script.py", "unsafe;touch"], output)
    assert result == valid_image()
    assert captured["args"] == ("python", "script.py", "unsafe;touch")
    assert "shell" not in captured["kwargs"]


@pytest.mark.asyncio
async def test_subprocess_failure_invalid_json_and_timeout(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(main, "SETTINGS", replace(main.SETTINGS, inference_timeout_sec=0.01))

    async def failed(*_args, **_kwargs):
        return FakeProcess(returncode=3)

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", failed)
    with pytest.raises(Exception) as failed_exc:
        await main.run_inference(["python", "script.py"], output)
    assert getattr(failed_exc.value, "status_code", None) == 502

    async def invalid(*_args, **_kwargs):
        return FakeProcess(output_path=output, body="not json")

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", invalid)
    with pytest.raises(Exception) as invalid_exc:
        await main.run_inference(["python", "script.py"], output)
    assert getattr(invalid_exc.value, "status_code", None) == 502

    process = FakeProcess(delay=0.2)

    async def slow(*_args, **_kwargs):
        return process

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", slow)
    with pytest.raises(Exception) as timeout_exc:
        await main.run_inference(["python", "script.py"], output)
    assert getattr(timeout_exc.value, "status_code", None) == 504
    assert process.killed is True
