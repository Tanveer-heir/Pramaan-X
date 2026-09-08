"""Mocked contract checks for the PRNU child-service adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gateway.config import GatewayConfig
from src.gateway.media import StoredMedia
from src.gateway.models import MediaRecord, MediaType
from src.gateway import orchestrator as module


def config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        detection_service_url="http://detection",
        prnu_service_url="http://prnu",
        shared_media_dir=tmp_path,
        investigation_output_dir=tmp_path / "artifacts",
        detection_timeout_sec=1,
        prnu_timeout_sec=1,
        source_attribution_timeout_sec=1,
        health_timeout_sec=1,
        max_upload_bytes=1000,
        cors_allowed_origins=(),
    )


def media(tmp_path: Path) -> StoredMedia:
    path = tmp_path / "evidence.png"
    path.write_bytes(b"x")
    return StoredMedia(
        investigation_id="INV-TEST",
        local_path=path,
        public=MediaRecord(filename="evidence.png", media_type=MediaType.IMAGE, sha256="0" * 64, size_bytes=1),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["prnu", "inconclusive"])
async def test_prnu_success_and_inconclusive_are_preserved(tmp_path, monkeypatch, method):
    payload = {
        "ok": True,
        "result": {"device_attribution": {"prediction": None, "primary_method": method, "evidence": ["native"]}},
    }

    async def post(_url, _payload):
        return payload

    monkeypatch.setattr(module, "_post_json", post)
    result = await module.build_default_prnu_runner(config(tmp_path))(media(tmp_path), tmp_path)
    assert result.result == payload


@pytest.mark.asyncio
async def test_prnu_unavailable_model_or_service_failure_is_explicit(tmp_path, monkeypatch):
    async def unavailable(_url, _payload):
        return {"ok": False, "error": {"code": "MODEL_UNAVAILABLE"}}

    monkeypatch.setattr(module, "_post_json", unavailable)
    with pytest.raises(module.ModuleExecutionError) as exc:
        await module.build_default_prnu_runner(config(tmp_path))(media(tmp_path), tmp_path)
    assert exc.value.reason == "PRNU_ANALYSIS_FAILED"

    async def service_failure(_url, _payload):
        raise module.ModuleExecutionError("UPSTREAM_HTTP_ERROR", "Service failed.")

    monkeypatch.setattr(module, "_post_json", service_failure)
    with pytest.raises(module.ModuleExecutionError) as service_exc:
        await module.build_default_prnu_runner(config(tmp_path))(media(tmp_path), tmp_path)
    assert service_exc.value.reason == "UPSTREAM_HTTP_ERROR"
