"""Evidence provenance and conservative normalization for source attribution."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from pydantic import ConfigDict

from src.common.schemas import ExternalMatch, OriginTracingEvidence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".flv"}
VALID_TIMESTAMP_TYPES = {"published", "platform_decoded", "archive_observed", "retrieved_at", "unknown"}
VALID_MEDIA_VERIFICATIONS = {"exact", "near_duplicate", "unverified_text", "not_downloaded", "failed"}
VALID_EVIDENCE_STATUSES = {"verified", "unverified", "fixture", "connector_error", "unavailable"}
GENERATED_STEM_RE = re.compile(
    r"^(?:upload|uploaded|file|image|photo|picture|video|media|evidence|input|output|tmp|temp)"
    r"(?:[_-]?(?:\d+|[0-9a-f]{8,}|[0-9a-f]{8}-[0-9a-f-]{27,}))?$",
    re.IGNORECASE,
)
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)


class SourceAttributionEvidence(OriginTracingEvidence):
    """Backward-compatible evidence object with explicit provenance extensions.

    ``OriginTracingEvidence`` lives in the shared package and intentionally remains
    unchanged.  This subclass allows the source-attribution boundary to expose the
    additional fields without changing the shared Section 2 schema.
    """

    model_config = ConfigDict(extra="allow")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def public_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    # These domains are reserved for documentation and must never be evidence.
    if (
        host == "example.com"
        or host.endswith((".example.com", ".example.org", ".example.net"))
        or host in {"example.org", "example.net", "localhost", "127.0.0.1"}
        or host.endswith((".invalid", ".test"))
    ):
        return False
    return True


def meaningful_filename(filename: str | Path | None) -> bool:
    if filename is None:
        return False
    stem = Path(str(filename)).stem.strip().replace(" ", "_")
    if not stem:
        return False
    normalized = stem.strip("._-").lower()
    if not normalized or normalized in {
        "upload", "uploaded", "file", "image", "photo", "picture", "video", "media",
        "evidence", "input", "output", "tmp", "temp", "test", "sample",
    }:
        return False
    compact = re.sub(r"[^0-9a-f]", "", normalized)
    if UUID_RE.fullmatch(normalized) or (len(compact) >= 16 and compact == normalized.replace("_", "")):
        return False
    if GENERATED_STEM_RE.fullmatch(normalized):
        return False
    if re.fullmatch(
        r"(?:upload(?:ed)?|file|image|photo|picture|video|media|evidence|input|output|tmp|temp)[_-][0-9a-f]{6,}",
        normalized,
    ):
        return False
    if re.fullmatch(r"(?:test|sample|fixture|dummy|example)(?:[_-].*)?", normalized):
        return False
    return len(re.sub(r"[_-]+", "", normalized)) >= 3


def local_context(media_path: str | Path, original_filename: str | None = None) -> list[str]:
    """Collect only facts observed locally, without guessing scene semantics."""
    path = Path(media_path)
    context: list[str] = []
    filename = original_filename or path.name
    if meaningful_filename(filename):
        context.append(f"User-provided filename: {Path(filename).name}")

    try:
        from PIL import Image, ExifTags

        with Image.open(path) as image:
            exif = image.getexif()
            tags = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
            for label in ("Make", "Model", "Software", "DateTime", "DateTimeOriginal"):
                value = tags.get(label)
                if value not in (None, ""):
                    context.append(f"EXIF {label}: {str(value)[:200]}")
            # OCR is optional local evidence.  It is never replaced with a
            # guessed description when the OCR dependency is absent or fails.
            try:
                import pytesseract

                ocr_text = " ".join(pytesseract.image_to_string(image).split())[:500]
                if ocr_text:
                    context.append(f"OCR text: {ocr_text}")
            except Exception:
                pass
    except Exception:
        pass
    return context


def normalize_candidate(
    value: Any,
    connector: str,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Normalize a connector record and attach an explicit evidence provenance."""
    source = as_mapping(value)
    url = str(source.get("post_url") or source.get("url") or "").strip()
    raw_synthetic = source.get("is_synthetic", False)
    if isinstance(raw_synthetic, str):
        is_synthetic = raw_synthetic.strip().lower() in {"1", "true", "yes"}
    else:
        is_synthetic = bool(raw_synthetic)
    status = str(
        source.get("evidence_status")
        or (source.get("status") if source.get("status") in VALID_EVIDENCE_STATUSES else None)
        or ("fixture" if is_synthetic else "unverified")
    )
    if status not in VALID_EVIDENCE_STATUSES:
        status = "unverified"

    published_at = source.get("published_at")
    timestamp_type = str(source.get("timestamp_type") or ("published" if published_at else "unknown"))
    if timestamp_type not in VALID_TIMESTAMP_TYPES:
        timestamp_type = "unknown"
    if not published_at or timestamp_type in {"retrieved_at", "unknown"}:
        published_at = None
    elif isinstance(published_at, str):
        try:
            datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError:
            published_at = None
            timestamp_type = "unknown"

    media_verification = str(source.get("media_verification") or ("not_downloaded" if source.get("has_media") else "unverified_text"))
    if media_verification not in VALID_MEDIA_VERIFICATIONS:
        media_verification = "unverified_text"

    normalized = dict(source)
    normalized.update(
        {
            "source_connector": str(source.get("source_connector") or connector),
            "retrieved_at": str(source.get("retrieved_at") or retrieved_at or utc_now()),
            "published_at": published_at,
            "timestamp_type": timestamp_type,
            "media_verification": media_verification,
            "evidence_status": status,
            "is_synthetic": is_synthetic,
            "post_url": url,
            "failure_reason": source.get("failure_reason"),
        }
    )
    return normalized


def candidate_is_verified(candidate: Mapping[str, Any]) -> bool:
    return (
        public_url(candidate.get("post_url") or candidate.get("url"))
        and candidate.get("is_synthetic") is False
        and candidate.get("evidence_status") == "verified"
        and candidate.get("media_verification") in {"exact", "near_duplicate"}
    )


def candidate_has_usable_timestamp(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("published_at")
        and candidate.get("timestamp_type") in {"published", "platform_decoded", "archive_observed"}
    )


def candidate_to_external_match(candidate: Mapping[str, Any]) -> ExternalMatch:
    return ExternalMatch(
        url=str(candidate.get("post_url") or candidate.get("url")),
        source=str(candidate.get("source_connector") or candidate.get("platform") or "unknown"),
        date_found=candidate.get("published_at"),
        page_title=candidate.get("title") or None,
    )


def connector_status(
    connector: str,
    status: str,
    *,
    result_count: int = 0,
    reason: str | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source_connector": connector,
        "status": status,
        "result_count": result_count,
        "retrieved_at": retrieved_at or utc_now(),
    }
    if reason:
        value["failure_reason"] = reason
    return value


def redact_secrets(value: Any) -> Any:
    """Remove configured credentials from payload-shaped values before serialization."""
    secret_names = (
        "GEMINI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "YOUTUBE_API_KEY",
        "TWITTER_BEARER_TOKEN", "APIFY_API_TOKEN", "YANDEX_API_KEY",
    )
    secrets = [os.getenv(name, "") for name in secret_names]
    # pydantic-settings can load values from .env without placing them in the
    # process environment.  Include those values when available as well.
    try:
        from src.common.config import settings

        secrets.extend(str(getattr(settings, name, "") or "") for name in secret_names)
    except Exception:
        pass
    secrets = [secret for secret in secrets if secret]
    if isinstance(value, dict):
        return {key: redact_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secrets:
            result = result.replace(secret, "[REDACTED]")
        return result
    return value
