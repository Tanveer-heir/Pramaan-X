"""Thin FastAPI service around the accepted image and video inference CLIs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SCRIPT = ROOT / "scripts" / "inference" / "analyse_pramaan_x_image.py"
VIDEO_SCRIPT = ROOT / "scripts" / "inference" / "run_pramaan_x.py"
VIDEO_CLASSES = {
    "REAL",
    "VISUAL_MANIPULATION",
    "AUDIO_MANIPULATION",
    "AUDIO_VISUAL_MANIPULATION",
}
IMAGE_LABELS = {"LIKELY_AUTHENTIC", "SUSPICIOUS", "LIKELY_MANIPULATED"}
logger = logging.getLogger("pramaan_x.detection_service")

load_dotenv(ROOT.parent / ".env", override=False)
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class ServiceSettings:
    allowed_media_root: Path
    output_root: Path
    device: str
    inference_timeout_sec: float
    image_timeout_sec: float
    max_input_bytes: int
    fusion_checkpoint: Path
    face_landmarker: Path
    xlsr_model: Path

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            allowed_media_root=Path(os.getenv("DETECTION_ALLOWED_MEDIA_ROOT", "/data/shared_media")),
            output_root=Path(os.getenv("DETECTION_OUTPUT_DIR", tempfile.gettempdir())) / "pramaan-x-detection",
            device=os.getenv("PRAMAAN_DEVICE", "auto"),
            inference_timeout_sec=float(os.getenv("DETECTION_PROCESS_TIMEOUT_SEC", "900")),
            image_timeout_sec=float(os.getenv("DETECTION_IMAGE_TIMEOUT_SEC", "90")),
            max_input_bytes=int(os.getenv("DETECTION_MAX_INPUT_BYTES", str(500 * 1024 * 1024))),
            fusion_checkpoint=Path(
                os.getenv(
                    "PRAMAAN_FUSION_CHECKPOINT",
                    str(ROOT / "checkpoints" / "fusion" / "pramaan_x_hackathon_final.pth"),
                )
            ),
            face_landmarker=Path(
                os.getenv(
                    "PRAMAAN_FACE_LANDMARKER",
                    str(ROOT / "models" / "mediapipe" / "face_landmarker.task"),
                )
            ),
            xlsr_model=Path(
                os.getenv(
                    "PRAMAAN_XLSR_MODEL",
                    str(ROOT / "models" / "xlsr_sls" / "xlsr-sls.onnx"),
                )
            ),
        )


SETTINGS = ServiceSettings.from_env()
SETTINGS.output_root.mkdir(parents=True, exist_ok=True)


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    media_path: str = Field(min_length=1)


app = FastAPI(
    title="Pramaan-X Detection Service",
    version="1.0.0",
    description="HTTP adapter for the accepted Pramaan-X image and video inference commands.",
)


def _resolve_media(value: str) -> Path:
    if value.startswith(("http://", "https://", "file://")):
        raise HTTPException(status_code=400, detail={"code": "REMOTE_MEDIA_REJECTED", "message": "Use a shared local evidence path."})
    path = Path(value).resolve()
    root = SETTINGS.allowed_media_root.resolve()
    if root != path and root not in path.parents:
        raise HTTPException(status_code=403, detail={"code": "MEDIA_PATH_FORBIDDEN", "message": "Media path is outside the shared evidence root."})
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "MEDIA_NOT_FOUND", "message": "Evidence file was not found."})
    if path.stat().st_size > SETTINGS.max_input_bytes:
        raise HTTPException(status_code=413, detail={"code": "MEDIA_TOO_LARGE", "message": "Evidence exceeds the configured size limit."})
    return path


def _media_type(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(32)
    suffix = path.suffix.lower()
    is_image = (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or (len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP")
    )
    is_video = (
        (len(header) >= 12 and header[4:8] == b"ftyp")
        or header.startswith(b"\x1aE\xdf\xa3")
        or (len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ")
    )
    if is_image and suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "image"
    if is_video and suffix in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
        return "video"
    raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA", "message": "Unsupported or mismatched media content."})


def build_command(media_path: Path, media_type: str, output_path: Path) -> list[str]:
    if media_type == "image":
        return [
            sys.executable,
            str(IMAGE_SCRIPT),
            str(media_path),
            "--output",
            str(output_path),
            "--timeout",
            str(SETTINGS.image_timeout_sec),
        ]
    return [
        sys.executable,
        str(VIDEO_SCRIPT),
        str(media_path),
        "--output",
        str(output_path),
        "--device",
        SETTINGS.device,
        "--fusion-checkpoint",
        str(SETTINGS.fusion_checkpoint),
        "--face-landmarker",
        str(SETTINGS.face_landmarker),
    ]


def validate_result(payload: Any, media_type: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Detection output is not a JSON object")
    if media_type == "image":
        assessment = payload.get("assessment")
        if payload.get("schema_version") != "pramaan_x_image_analysis_v1" or not isinstance(assessment, dict):
            raise ValueError("Image output has an invalid schema")
        if assessment.get("label") not in IMAGE_LABELS:
            raise ValueError("Image output has an invalid assessment label")
    else:
        if payload.get("schema_version") != "pramaan_x_prediction_v2":
            raise ValueError("Video output has an invalid schema")
        if payload.get("selected_class") not in VIDEO_CLASSES:
            raise ValueError("Video output has an invalid selected class")
        if not isinstance(payload.get("branch_evidence"), dict):
            raise ValueError("Video output is missing branch evidence")
    return payload


async def run_inference(command: list[str], output_path: Path) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=SETTINGS.inference_timeout_sec,
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise HTTPException(status_code=504, detail={"code": "DETECTION_TIMEOUT", "message": "Detection exceeded its configured timeout."})
    if process.returncode != 0:
        logger.error(
            "Detection process failed with code %s; stderr tail=%s",
            process.returncode,
            stderr.decode("utf-8", errors="replace")[-800:],
        )
        raise HTTPException(status_code=502, detail={"code": "DETECTION_PROCESS_FAILED", "message": "Detection engine exited without a valid result."})
    if not output_path.is_file():
        logger.error("Detection process exited successfully without output; stdout tail=%s", stdout.decode("utf-8", errors="replace")[-800:])
        raise HTTPException(status_code=502, detail={"code": "DETECTION_OUTPUT_MISSING", "message": "Detection engine did not produce a result."})
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail={"code": "DETECTION_OUTPUT_INVALID", "message": "Detection engine produced invalid JSON."}) from exc


@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    image_assets = {"script": IMAGE_SCRIPT.is_file(), "credential": bool(os.getenv("KIMI_API_KEY"))}
    video_assets = {
        "script": VIDEO_SCRIPT.is_file(),
        "fusion_checkpoint": SETTINGS.fusion_checkpoint.is_file(),
        "face_landmarker": SETTINGS.face_landmarker.is_file(),
        "xlsr_model": SETTINGS.xlsr_model.is_file(),
        "visual_checkpoint": (ROOT / "checkpoints" / "visual_temporal" / "visual_temporal_20260904T145626Z.pt").is_file(),
        "audio_checkpoint": (ROOT / "checkpoints" / "audio_classifier" / "audio_classifier_20260904T181732Z.pt").is_file(),
        "av_checkpoint": (ROOT / "checkpoints" / "av_sync_dense" / "av_sync_dense_20260905T055537Z.pt").is_file(),
    }
    runtime = _runtime_status()
    image_available = all(image_assets.values())
    video_available = all(video_assets.values()) and runtime["torch_available"] and runtime["onnxruntime_available"]
    if SETTINGS.device.startswith("cuda"):
        video_available = video_available and runtime["cuda_available"] and runtime["onnx_cuda_provider"]
    return {
        "status": "ready" if image_available or video_available else "degraded",
        "process_alive": True,
        "service": "pramaan-x-detection",
        "device_requested": SETTINGS.device,
        "runtime": runtime,
        "capabilities": {
            "image": {"available": image_available, "assets": image_assets},
            "video": {"available": video_available, "assets": video_assets},
        },
    }


def _runtime_status() -> dict[str, bool]:
    result = {
        "torch_available": importlib.util.find_spec("torch") is not None,
        "onnxruntime_available": importlib.util.find_spec("onnxruntime") is not None,
        "cuda_available": False,
        "onnx_cuda_provider": False,
    }
    if result["torch_available"]:
        try:
            import torch

            result["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            pass
    if result["onnxruntime_available"]:
        try:
            import onnxruntime

            result["onnx_cuda_provider"] = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        except Exception:
            pass
    return result


@app.post("/api/v1/detect", tags=["Detection"])
@app.post("/detect", tags=["Detection"])
async def detect(request: DetectionRequest) -> dict[str, Any]:
    media_path = _resolve_media(request.media_path)
    media_type = _media_type(media_path)
    with tempfile.TemporaryDirectory(prefix="request_", dir=SETTINGS.output_root) as temporary:
        output_path = Path(temporary) / "detection.json"
        command = build_command(media_path, media_type, output_path)
        payload = await run_inference(command, output_path)
    try:
        return validate_result(payload, media_type)
    except ValueError as exc:
        logger.error("Detection output schema validation failed: %s", exc)
        raise HTTPException(status_code=502, detail={"code": "DETECTION_SCHEMA_INVALID", "message": "Detection output failed schema validation."}) from exc
