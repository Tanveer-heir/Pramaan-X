"""Focused tests for the standalone CPH source-attribution CLI boundary."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

from src.common.schemas import OriginTracingEvidence


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_source_attribution.py"
SPEC = importlib.util.spec_from_file_location("run_source_attribution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_wrapper_serializes_native_origin_tracing_evidence(tmp_path, monkeypatch):
    expected = OriginTracingEvidence(circulation_summary={"status": "native"})

    class FakePipeline:
        async def execute(self, *, media_path):
            assert media_path.endswith("evidence.png")
            return expected

    monkeypatch.setattr(MODULE, "SourceAttributionPipeline", FakePipeline)

    payload = asyncio.run(MODULE.run(tmp_path / "evidence.png"))
    output = tmp_path / "origin.json"
    MODULE.atomic_json(output, payload)

    assert payload == expected.model_dump(mode="json")
    assert output.read_text(encoding="utf-8").endswith("\n")
