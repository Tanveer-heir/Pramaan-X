"""Tests for Master Section 2 Orchestrator."""

import pytest
from src.pipeline.orchestrator import Section2Orchestrator

@pytest.mark.asyncio
async def test_orchestrator_execution():
    orchestrator = Section2Orchestrator()
    report = await orchestrator.analyze(
        media_path="data/sample_media/test_image.jpg",
        case_id="case_demo_001",
        section1_evidence={"status": "complete", "deepfake_detected": False}
    )
    assert report.case_id == "case_demo_001"
    assert report.detection["deepfake_detected"] is False
    assert report.provenance.metadata.consistent is True
    assert len(report.chain_of_custody.audit_log) >= 3
