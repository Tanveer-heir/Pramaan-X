"""Environment-backed gateway configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class GatewayConfig:
    detection_service_url: str
    prnu_service_url: str
    shared_media_dir: Path
    investigation_output_dir: Path
    detection_timeout_sec: float
    prnu_timeout_sec: float
    source_attribution_timeout_sec: float
    health_timeout_sec: float
    max_upload_bytes: int
    cors_allowed_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        origins = tuple(
            value.strip()
            for value in os.getenv(
                "CORS_ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:5173",
            ).split(",")
            if value.strip()
        )
        return cls(
            detection_service_url=os.getenv(
                "DETECTION_SERVICE_URL",
                os.getenv("AI_SERVICE_URL", "http://ai-detection:8001"),
            ).rstrip("/"),
            prnu_service_url=os.getenv("PRNU_SERVICE_URL", "http://prnu-forensics:8002").rstrip("/"),
            shared_media_dir=Path(os.getenv("SHARED_MEDIA_DIR", "data/shared_media")),
            investigation_output_dir=Path(os.getenv("INVESTIGATION_OUTPUT_DIR", "data/investigations")),
            detection_timeout_sec=_positive_float("DETECTION_TIMEOUT_SEC", 180.0),
            prnu_timeout_sec=_positive_float("PRNU_TIMEOUT_SEC", 60.0),
            source_attribution_timeout_sec=_positive_float("SOURCE_ATTRIBUTION_TIMEOUT_SEC", 180.0),
            health_timeout_sec=_positive_float("SERVICE_HEALTH_TIMEOUT_SEC", 3.0),
            max_upload_bytes=_positive_int("MAX_UPLOAD_BYTES", 500 * 1024 * 1024),
            cors_allowed_origins=origins,
        )
