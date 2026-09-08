"""Strict public contracts for the versioned unified investigation API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class ModuleStatus(str, Enum):
    COMPLETED = "COMPLETED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class InvestigationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class MediaRecord(StrictModel):
    filename: str
    media_type: MediaType
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)


class ModuleResult(StrictModel):
    status: ModuleStatus
    reason: str | None = None
    message: str | None = None
    duration_sec: float = Field(ge=0.0)
    result: dict[str, Any] | None = None


class InvestigationModules(StrictModel):
    detection: ModuleResult
    device_attribution: ModuleResult
    source_attribution: ModuleResult


class ArtifactReference(StrictModel):
    url: str
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CustodyEntry(StrictModel):
    sequence: int = Field(ge=1)
    event: str
    timestamp: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    module_statuses: dict[str, ModuleStatus] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
    previous_entry_hash: str | None = None
    entry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CustodyRecord(StrictModel):
    hash_algorithm: Literal["sha256"] = "sha256"
    entries: list[CustodyEntry]


class InvestigationSummary(StrictModel):
    detection_label: str | None = None
    detection_score_semantics: str | None = None
    attributed_device: str | None = None
    device_method: str | None = None
    patient_zero_candidate_domain: str | None = None


class InvestigationResponse(StrictModel):
    schema_version: Literal["pramaan_x_investigation_v1"] = "pramaan_x_investigation_v1"
    investigation_id: str
    case_id: str
    status: InvestigationStatus
    media: MediaRecord
    modules: InvestigationModules
    artifacts: dict[str, ArtifactReference] = Field(default_factory=dict)
    summary: InvestigationSummary | None = None
    custody: CustodyRecord
    warnings: list[str] = Field(default_factory=list)
    execution_time_sec: float = Field(ge=0.0)


class CapabilityStatus(StrictModel):
    available: bool
    healthy: bool
    detail: str | None = None


class CapabilitiesResponse(StrictModel):
    schema_version: Literal["pramaan_x_capabilities_v1"] = "pramaan_x_capabilities_v1"
    image_detection: CapabilityStatus
    video_detection: CapabilityStatus
    prnu_image_attribution: CapabilityStatus
    source_attribution_image: CapabilityStatus
    source_attribution_video: CapabilityStatus
