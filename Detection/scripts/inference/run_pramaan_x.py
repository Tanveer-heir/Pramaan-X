"""Analyse one image or video with the accepted Pramaan-X boundaries.

This is the stable command-line boundary for a later API.  It deliberately
delegates raw video extraction and masked fusion to
``predict_pramaan_x_video.py``. Still images deliberately use the separate
categorical Kimi/VLM implementation rather than a fabricated fusion score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEO_PREDICTOR = ROOT / "scripts" / "inference" / "predict_pramaan_x_video.py"
IMAGE_PREDICTOR = ROOT / "scripts" / "inference" / "analyse_pramaan_x_image.py"
DEFAULT_FUSION = ROOT / "checkpoints" / "fusion" / "pramaan_x_hackathon_final.pth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# These are the formats accepted by the existing raw-video service boundary.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("media", type=Path, help="Path to an input image or video file.")
    result.add_argument("--output", type=Path, help="Output JSON path. Defaults to predictions/<media-stem>.json.")
    result.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    result.add_argument("--fusion-checkpoint", type=Path, default=DEFAULT_FUSION)
    result.add_argument("--keep-temporary-features", action="store_true")
    result.add_argument("--face-landmarker", type=Path)
    return result


def default_output(media: Path) -> Path:
    return ROOT / "predictions" / f"{media.stem}_pramaan_x.json"


def media_type(media: Path) -> str:
    suffix = media.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
    raise ValueError(
        f"Unsupported media type {suffix or '<none>'}. Supported extensions: {supported}"
    )


def command_for_media(args: argparse.Namespace, media: Path, output: Path, kind: str) -> list[str]:
    if kind == "image":
        if not IMAGE_PREDICTOR.is_file():
            raise RuntimeError(f"Required inference script is missing: {IMAGE_PREDICTOR}")
        return [
            sys.executable,
            str(IMAGE_PREDICTOR),
            str(media),
            "--output",
            str(output),
        ]

    if not RAW_VIDEO_PREDICTOR.is_file():
        raise RuntimeError(f"Required inference script is missing: {RAW_VIDEO_PREDICTOR}")
    if not args.fusion_checkpoint.is_file():
        raise FileNotFoundError(f"Fusion checkpoint not found: {args.fusion_checkpoint}")
    command = [
        sys.executable, str(RAW_VIDEO_PREDICTOR),
        "--video", str(media),
        "--fusion-checkpoint", str(args.fusion_checkpoint),
        "--output", str(output),
        "--device", args.device,
    ]
    if args.keep_temporary_features:
        command.append("--keep-temporary-features")
    if args.face_landmarker is not None:
        command.extend(("--face-landmarker", str(args.face_landmarker.expanduser().resolve())))
    return command


def concise_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": payload.get("raw_video", {}).get("path"),
        "selected_class": payload.get("selected_class"),
        "softmax_probabilities": payload.get("softmax_probabilities"),
        "branch_availability": {
            name: payload.get("branch_evidence", {}).get(f"{name}_available")
            for name in ("visual", "audio", "av")
        },
        "note": payload.get("probability_note"),
    }


def main() -> None:
    args = parser().parse_args()
    media = args.media.expanduser().resolve()
    output = (args.output or default_output(media)).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"Input media not found: {media}")
    kind = media_type(media)
    command = command_for_media(args, media, output, kind)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not output.is_file():
        raise RuntimeError(f"Inference exited successfully but did not produce: {output}")
    with output.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    print("\nPRAMAAN-X RESULT")
    if kind == "video":
        print(json.dumps(concise_summary(payload), indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nFull forensic JSON: {output}")


if __name__ == "__main__":
    main()
