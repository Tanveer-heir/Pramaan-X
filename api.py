"""
Pramaan-X Microservice API Node (CPH Track 2 - Node 1: Detect)
Exposes POST /detect returning manipulation probabilities for images and videos.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
import subprocess
import urllib.request
import json
import uuid
import sys
import os

app = FastAPI(
    title="Pramaan-X Manipulation Detection API",
    description="Multimodal video fusion & image visual authenticity triage",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT / "predictions" / "temp_uploads"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

SCRIPTS_DIR = (ROOT / "Detection" / "scripts") if (ROOT / "Detection" / "scripts").exists() else (ROOT / "scripts")
DEFAULT_FUSION = (
    str(ROOT / "Detection" / "checkpoints" / "fusion" / "pramaan_x_hackathon_final.pth")
    if (ROOT / "Detection" / "checkpoints").exists()
    else str(ROOT / "checkpoints" / "fusion" / "pramaan_x_hackathon_final.pth")
)
FUSION_CHECKPOINT = os.getenv("FUSION_CHECKPOINT", DEFAULT_FUSION)
DEVICE = os.getenv("PRAMAAN_DEVICE", "cpu")


class DetectRequest(BaseModel):
    media_path: Optional[str] = Field(None, description="Local path or remote URL to image/video")
    file_path: Optional[str] = Field(None, description="Shared volume path to image/video")
    image: Optional[str] = None
    video: Optional[str] = None


@app.get("/health", tags=["System"])
def health():
    fusion_exists = Path(FUSION_CHECKPOINT).is_file()
    return {
        "status": "ok",
        "service": "cph-detect-pramaan-x",
        "device": DEVICE,
        "fusion_checkpoint_loaded": fusion_exists,
    }


def download_or_resolve(media_path: str) -> Path:
    if media_path.startswith(("http://", "https://")):
        ext = Path(media_path.split("?")[0]).suffix or ".mp4"
        dest = WORKSPACE_DIR / f"{uuid.uuid4().hex}{ext}"
        req = urllib.request.Request(
            media_path, headers={"User-Agent": "Pramaan-X-Detector/1.0"}
        )
        with urllib.request.urlopen(req, timeout=45) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return dest
    else:
        p = Path(media_path)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {media_path}")
        return p


def run_video_inference(video_path: Path) -> Dict[str, Any]:
    out_json = WORKSPACE_DIR / f"vid_{uuid.uuid4().hex}.json"
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "inference" / "predict_pramaan_x_video.py"),
        "--video", str(video_path.resolve()),
        "--fusion-checkpoint", FUSION_CHECKPOINT,
        "--device", DEVICE,
        "--output", str(out_json.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPTS_DIR.parent))
    if not out_json.exists():
        raise RuntimeError(f"Video inference failed: {res.stderr or res.stdout}")

    with open(out_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out_json.unlink(missing_ok=True)

    softmax = raw.get("softmax_probabilities", [0.25, 0.25, 0.25, 0.25])
    selected = raw.get("selected_class", "UNKNOWN")
    branches = raw.get("branch_evidence", {})

    return {
        "media_type": "video",
        "prediction": selected,
        "confidence": round(max(softmax), 4),
        "probabilities": {
            "real": round(softmax[0], 4) if len(softmax) > 0 else 0.0,
            "visual_manipulation": round(softmax[1], 4) if len(softmax) > 1 else 0.0,
            "audio_manipulation": round(softmax[2], 4) if len(softmax) > 2 else 0.0,
            "audio_visual_manipulation": round(softmax[3], 4) if len(softmax) > 3 else 0.0,
        },
        "branch_status": {
            "visual_available": branches.get("visual_available", False),
            "audio_available": branches.get("audio_available", False),
            "av_available": branches.get("av_available", False),
        }
    }


def run_image_inference(image_path: Path) -> Dict[str, Any]:
    out_json = WORKSPACE_DIR / f"img_{uuid.uuid4().hex}.json"
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "inference" / "analyse_pramaan_x_image.py"),
        str(image_path.resolve()),
        "--output", str(out_json.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPTS_DIR.parent))
    if not out_json.exists():
        # Fallback if KIMI_API_KEY is not set or network failed
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                exif = img.getexif()
                has_exif = bool(exif)
            return {
                "media_type": "image",
                "prediction": "LIKELY_AUTHENTIC" if has_exif else "SUSPICIOUS",
                "confidence": "MEDIUM",
                "probability": 0.20 if has_exif else 0.60,
                "summary": "Local image triage completed. Exif metadata present." if has_exif else "Missing EXIF metadata; flagged as suspicious.",
                "findings": ["exif_present"] if has_exif else ["exif_missing"],
                "note": "For deep VLM analysis, set KIMI_API_KEY in .env"
            }
        except Exception:
            raise RuntimeError(f"Image analysis failed: {res.stderr or res.stdout}")

    with open(out_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out_json.unlink(missing_ok=True)

    assessment = raw.get("assessment", {})
    label = assessment.get("label", "SUSPICIOUS")
    prob_map = {"LIKELY_AUTHENTIC": 0.15, "SUSPICIOUS": 0.55, "LIKELY_MANIPULATED": 0.90}
    findings_list = [f.get("category", "") for f in raw.get("visual_findings", [])]

    return {
        "media_type": "image",
        "prediction": label,
        "confidence": assessment.get("confidence_level", "MEDIUM"),
        "probability": prob_map.get(label, 0.50),
        "summary": assessment.get("summary", ""),
        "findings": [f for f in findings_list if f]
    }


@app.post("/detect", tags=["Detection"])
@app.post("/api/detect", tags=["Detection"])
async def detect(payload: DetectRequest):
    target = payload.media_path or payload.file_path or payload.image or payload.video
    if not target:
        raise HTTPException(
            status_code=400,
            detail="Request body must provide 'media_path', 'file_path', 'image', or 'video'",
        )
    file_path = download_or_resolve(target)
    suffix = file_path.suffix.lower()

    if suffix in IMAGE_EXTS:
        return run_image_inference(file_path)
    elif suffix in VIDEO_EXTS:
        return run_video_inference(file_path)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported formats: {IMAGE_EXTS | VIDEO_EXTS}"
        )


@app.post("/detect/upload", tags=["Detection"])
@app.post("/api/detect/upload", tags=["Detection"])
async def detect_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "file.mp4").suffix.lower()
    temp_path = WORKSPACE_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        if suffix in IMAGE_EXTS:
            return run_image_inference(temp_path)
        elif suffix in VIDEO_EXTS:
            return run_video_inference(temp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{suffix}'")
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
