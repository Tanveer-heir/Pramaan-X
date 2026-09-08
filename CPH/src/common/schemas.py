"""
Canonical Pydantic Schemas for Section 2: Provenance, Origin Tracing & Evidence Reporting.
Strictly adheres to Section 2 specification (§2.3, §2.4, §2.5).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ==============================================================================
# 1. Provenance Check Schemas (§2.3)
# ==============================================================================

class C2PAResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    present: bool = Field(default=False, description="Whether C2PA credentials are found")
    valid: Optional[bool] = Field(default=None, description="Cryptographic validity if present")
    issuer: Optional[str] = Field(default=None, description="Certifying entity or tool")


class SynthIDResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str = Field(default="not_checked", description="Verification status (roadmap slot)")


class MetadataResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    consistent: bool = Field(default=True, description="Whether metadata aligns with image specs")
    camera_model: Optional[str] = None
    software_tag: Optional[str] = None
    created_timestamp: Optional[str] = None
    gps_coordinates: Optional[Dict[str, float]] = None
    notes: List[str] = Field(default_factory=list, description="Inconsistency / tamper notes")


class PRNUResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str = Field(default="NOT_APPLICABLE", description="Match status or NOT_APPLICABLE")
    reason: Optional[str] = Field(default="no_reference_fingerprint", description="Reason if N/A")
    matched_device_id: Optional[str] = None
    pce_score: Optional[float] = None
    confidence: Optional[float] = None


class PlatformFingerprintResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    predicted_platform: Optional[str] = Field(default=None, description="WhatsApp, Instagram, etc.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quantization_quality: Optional[int] = None


class ProvenanceEvidence(BaseModel):
    """Output of Provenance Check stage (§2.3)."""
    model_config = ConfigDict(extra="ignore")
    c2pa: C2PAResult = Field(default_factory=C2PAResult)
    synthid: SynthIDResult = Field(default_factory=SynthIDResult)
    metadata: MetadataResult = Field(default_factory=MetadataResult)
    prnu: PRNUResult = Field(default_factory=PRNUResult)
    platform_fingerprint: PlatformFingerprintResult = Field(default_factory=PlatformFingerprintResult)


# ==============================================================================
# 2. Origin Tracing & Source Attribution Schemas (§2.4)
# ==============================================================================

class InternalMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instance_id: str
    platform: str
    timestamp: str
    hamming_distance: int
    match_confidence: float


class ExternalMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str
    source: str = Field(..., description="e.g. google_vision_web_detection, yandex_reverse_image")
    date_found: Optional[str] = None
    page_title: Optional[str] = None


class EarliestCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instance_id: Optional[str] = None
    url: Optional[str] = None
    account: Optional[str] = None
    platform: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: str = Field(default="moderate", description="high, moderate, or low")
    title: Optional[str] = None


class PublicationRecord(BaseModel):
    """An individual published occurrence in the 50-place chronological propagation chain."""
    model_config = ConfigDict(extra="ignore")
    rank: int = Field(..., description="Chronological publication sequence (#1 = earliest origin)")
    role: str = Field(default="propagation", description="primary_origin, early_reporting, viral_spread, current_circulation")
    platform: str = Field(..., description="open_web, x, reddit, youtube, telegram, reverse_visual_search")
    account: str = Field(..., description="Username, channel, or publishing domain")
    timestamp: str = Field(..., description="UTC publication timestamp")
    elapsed_time: str = Field(default="+0m", description="Elapsed time relative to primary origin")
    post_url: Optional[str] = None
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    title: Optional[str] = None
    snippet: Optional[str] = None


class ShortlistCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    platform: str
    account: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    post_url: Optional[str] = None
    timestamp: Optional[str] = None
    post_text: Optional[str] = None
    has_media: bool = Field(default=False, description="True if post contains the actual image/video")


class TemporalWeighting(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content_age: str = Field(default="new", description="'new' or 'older'")
    platform_search_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    reverse_search_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class AccountAttribution(BaseModel):
    model_config = ConfigDict(extra="ignore")
    media_description: str = Field(default="", description="LLM-generated description")
    search_queries: Dict[str, List[str]] = Field(default_factory=dict)
    shortlist: List[ShortlistCandidate] = Field(default_factory=list)
    sources_not_searched: List[str] = Field(
        default_factory=lambda: ["instagram", "facebook"],
        description="Platforms with restricted open search APIs"
    )
    google_trends_corroboration: Optional[Dict[str, Any]] = None
    temporal_weighting: TemporalWeighting = Field(default_factory=TemporalWeighting)
    social_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Account-level repost/interaction graph JSON or ref"
    )
    identified_event: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Discovered real-world incident context, entities, and canonical date"
    )


class OriginTracingEvidence(BaseModel):
    """Output of Origin Tracing stage (§2.4)."""
    model_config = ConfigDict(extra="ignore")
    internal_matches: List[InternalMatch] = Field(default_factory=list)
    external_matches: List[ExternalMatch] = Field(default_factory=list)
    earliest_candidate: Optional[EarliestCandidate] = None
    dissemination_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Content instance nearest-predecessor propagation graph"
    )
    account_attribution: AccountAttribution = Field(default_factory=AccountAttribution)
    first_50_publications: List[PublicationRecord] = Field(
        default_factory=list,
        description="First 50 chronological publication locations and active circulation venues"
    )
    circulation_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Summary of current live circulation, spread velocity, and platform breakdown"
    )


# ==============================================================================
# 3. Chain of Custody & Combined Evidence Report (§2.5)
# ==============================================================================

class AuditLogEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stage: str = Field(
        ...,
        description="ingested, detection_complete, provenance_complete, origin_tracing_complete, report_generated, investigator_confirmed"
    )
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_hash: str
    output_hash: str
    prev_entry_hash: Optional[str] = Field(
        default=None,
        description="Hash of previous entry for tamper-evident ledger integrity"
    )
    model_version: Optional[str] = None
    investigator_id: Optional[str] = None
    written_back_to: Optional[List[str]] = None


class ChainOfCustody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content_hash_sha256: str
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_version: str = "section1-v3 + section2-v1"
    audit_log: List[AuditLogEntry] = Field(default_factory=list)


class CombinedEvidenceReport(BaseModel):
    """
    Combined Forensic Evidence Object (§2.5).
    Feeds the Investigator Dashboard (Feature 6) & Evidence-Grade Export (Feature 8).
    """
    model_config = ConfigDict(extra="ignore")
    case_id: str
    detection: Dict[str, Any] = Field(
        default_factory=dict,
        description="Section 1 detection evidence object (unmodified)"
    )
    provenance: ProvenanceEvidence = Field(default_factory=ProvenanceEvidence)
    origin_tracing: OriginTracingEvidence = Field(default_factory=OriginTracingEvidence)
    chain_of_custody: ChainOfCustody
