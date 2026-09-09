"""Independent upload boundaries for Detection, PRNU, and CPH tracing."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.source_attribution.pipeline import SourceAttributionPipeline


router = APIRouter(tags=["Standalone Subsystems"])

CPH_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = CPH_ROOT.parent
DETECTION_RUNNER = REPOSITORY_ROOT / "Detection" / "scripts" / "inference" / "run_pramaan_x.py"
PRNU_ROOT = REPOSITORY_ROOT / "prnu-device-attribution-api-handoff"
PRNU_MODEL_DIR = Path(os.getenv("PRNU_MODEL_DIR", str(PRNU_ROOT / "models")))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def _suffix(filename: str | None) -> str:
    return Path(filename or "upload").suffix.lower()


def _require_detection_media(filename: str | None) -> None:
    suffix = _suffix(filename)
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={"code": "UNSUPPORTED_MEDIA", "message": "Detection accepts supported image or video files only."},
        )


def _require_image(filename: str | None) -> None:
    if _suffix(filename) not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={"code": "NOT_APPLICABLE", "message": "PRNU device attribution is image-only."},
        )


async def _save_upload(file: UploadFile, directory: Path) -> Path:
    suffix = _suffix(file.filename)
    destination = directory / f"upload{suffix}"
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)
    if destination.stat().st_size == 0:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_UPLOAD", "message": "Uploaded file is empty."})
    return destination


def run_detection(media_path: Path, output_path: Path) -> dict[str, Any]:
    if not DETECTION_RUNNER.is_file():
        raise RuntimeError(f"Detection boundary is missing: {DETECTION_RUNNER}")
    completed = subprocess.run(
        [sys.executable, str(DETECTION_RUNNER), str(media_path), "--output", str(output_path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")[-600:]
        raise RuntimeError(f"Detection boundary failed: {detail or 'nonzero exit'}")
    if not output_path.is_file():
        raise RuntimeError("Detection boundary exited without producing JSON output")
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_prnu(image_path: Path) -> dict[str, Any]:
    if str(PRNU_ROOT) not in sys.path:
        sys.path.insert(0, str(PRNU_ROOT))
    from prnu_attribution.model import predict

    return predict(image_path, PRNU_MODEL_DIR, mode="device")


async def run_cph(media_path: Path):
    return await SourceAttributionPipeline().execute(media_path=str(media_path))


@router.post("/api/v1/detection/analyze")
async def analyze_detection(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return the native Detection payload for one uploaded image or video."""
    try:
        _require_detection_media(file.filename)
        with tempfile.TemporaryDirectory(prefix="pramaan_detection_") as temporary:
            directory = Path(temporary)
            media_path = await _save_upload(file, directory)
            return await asyncio.to_thread(run_detection, media_path, directory / "detection.json")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "DETECTION_FAILED", "message": str(exc)}) from exc
    finally:
        await file.close()


@router.post("/api/v1/prnu/analyze")
async def analyze_prnu(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return the native image-only PRNU device-attribution payload."""
    try:
        _require_image(file.filename)
        with tempfile.TemporaryDirectory(prefix="pramaan_prnu_") as temporary:
            media_path = await _save_upload(file, Path(temporary))
            return await asyncio.to_thread(run_prnu, media_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "PRNU_FAILED", "message": str(exc)}) from exc
    finally:
        await file.close()


@router.post("/api/v1/cph/trace")
async def trace_cph(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return native OriginTracingEvidence for one uploaded image or video."""
    try:
        _require_detection_media(file.filename)
        with tempfile.TemporaryDirectory(prefix="pramaan_cph_") as temporary:
            media_path = await _save_upload(file, Path(temporary))
            evidence = await run_cph(media_path)
            return evidence.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "CPH_TRACE_FAILED", "message": str(exc)}) from exc
    finally:
        await file.close()
