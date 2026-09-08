"""Single-write media ingestion and content-based routing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import BinaryIO

from .models import MediaRecord, MediaType


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


class MediaValidationError(ValueError):
    """Raised when an upload cannot be accepted as supported media."""


@dataclass(frozen=True)
class StoredMedia:
    investigation_id: str
    local_path: Path
    public: MediaRecord


def sanitize_filename(filename: str | None) -> str:
    basename = Path(filename or "evidence").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return safe[:180] or "evidence"


def detect_media_type(header: bytes, suffix: str) -> MediaType:
    suffix = suffix.lower()
    if header.startswith(b"\xff\xd8\xff") or header.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = MediaType.IMAGE
    elif len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        detected = MediaType.IMAGE
    elif len(header) >= 12 and header[4:8] == b"ftyp":
        detected = MediaType.VIDEO
    elif header.startswith(b"\x1aE\xdf\xa3"):
        detected = MediaType.VIDEO
    elif len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        detected = MediaType.VIDEO
    else:
        raise MediaValidationError("File content is not a supported image or video format")

    allowed = IMAGE_SUFFIXES if detected is MediaType.IMAGE else VIDEO_SUFFIXES
    if suffix not in allowed:
        raise MediaValidationError("Filename extension does not match a supported media type")
    return detected


def _read_header(path: Path, size: int = 32) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def store_stream_once(
    stream: BinaryIO,
    original_filename: str | None,
    investigation_id: str,
    shared_media_dir: Path,
    max_upload_bytes: int,
) -> StoredMedia:
    safe_name = sanitize_filename(original_filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise MediaValidationError("Unsupported filename extension")

    shared_media_dir = shared_media_dir.resolve()
    shared_media_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = shared_media_dir / investigation_id / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    destination = evidence_dir / f"evidence{suffix}"
    temporary = evidence_dir / ".uploading"
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("xb") as output:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_upload_bytes:
                    raise MediaValidationError("Upload exceeds configured maximum size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total == 0:
            raise MediaValidationError("Uploaded file is empty")
        media_type = detect_media_type(_read_header(temporary), suffix)
        os.replace(temporary, destination)
        destination.chmod(0o444)
    except Exception:
        temporary.unlink(missing_ok=True)
        try:
            evidence_dir.rmdir()
            evidence_dir.parent.rmdir()
        except OSError:
            pass
        raise

    return StoredMedia(
        investigation_id=investigation_id,
        local_path=destination,
        public=MediaRecord(
            filename=safe_name,
            media_type=media_type,
            sha256=digest.hexdigest(),
            size_bytes=total,
        ),
    )


def store_path_once(
    source: Path,
    investigation_id: str,
    shared_media_dir: Path,
    max_upload_bytes: int,
) -> StoredMedia:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")
    with source.open("rb") as stream:
        return store_stream_once(
            stream,
            source.name,
            investigation_id,
            shared_media_dir,
            max_upload_bytes,
        )
