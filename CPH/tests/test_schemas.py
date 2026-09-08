"""Tests for Canonical Section 2 Pydantic Schemas."""

import pytest
from src.common.schemas import (
    ProvenanceEvidence,
    OriginTracingEvidence,
    CombinedEvidenceReport,
    ChainOfCustody,
    AuditLogEntry
)

def test_provenance_schema_defaults():
    prov = ProvenanceEvidence()
    assert prov.c2pa.present is False
    assert prov.synthid.status == "not_checked"
    assert prov.prnu.status == "NOT_APPLICABLE"
    assert prov.metadata.consistent is True

def test_origin_tracing_schema():
    orig = OriginTracingEvidence()
    assert orig.internal_matches == []
    assert orig.external_matches == []
    assert orig.account_attribution.sources_not_searched == ["instagram", "facebook"]

def test_combined_report_serialization():
    custody = ChainOfCustody(
        content_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        audit_log=[
            AuditLogEntry(
                stage="ingested",
                input_hash="hash1",
                output_hash="hash2",
                prev_entry_hash="0000"
            )
        ]
    )
    report = CombinedEvidenceReport(
        case_id="case_test_001",
        chain_of_custody=custody
    )
    json_data = report.model_dump()
    assert json_data["case_id"] == "case_test_001"
    assert "provenance" in json_data
    assert "origin_tracing" in json_data
    assert "chain_of_custody" in json_data
