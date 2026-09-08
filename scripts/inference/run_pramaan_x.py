"""Analyse one video with the frozen Pramaan-X hackathon model.

This is the stable command-line boundary for a later API.  It deliberately
delegates extraction and masked fusion to ``predict_pramaan_x_video.py`` so
there is only one raw-video inference implementation to maintain.
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
DEFAULT_FUSION = ROOT / "checkpoints" / "fusion" / "pramaan_x_hackathon_final.pth"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("video", type=Path, help="Path to the input video file.")
    result.add_argument("--output", type=Path, help="Output JSON path. Defaults to predictions/<video-stem>.json.")
    result.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    result.add_argument("--fusion-checkpoint", type=Path, default=DEFAULT_FUSION)
    result.add_argument("--keep-temporary-features", action="store_true")
    result.add_argument("--face-landmarker", type=Path)
    return result


def default_output(video: Path) -> Path:
    return ROOT / "predictions" / f"{video.stem}_pramaan_x.json"


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
    video = args.video.expanduser().resolve()
    output = (args.output or default_output(video)).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    if not RAW_VIDEO_PREDICTOR.is_file():
        raise RuntimeError(f"Required inference script is missing: {RAW_VIDEO_PREDICTOR}")
    if not args.fusion_checkpoint.is_file():
        raise FileNotFoundError(f"Fusion checkpoint not found: {args.fusion_checkpoint}")
    command = [
        sys.executable, str(RAW_VIDEO_PREDICTOR),
        "--video", str(video),
        "--fusion-checkpoint", str(args.fusion_checkpoint),
        "--output", str(output),
        "--device", args.device,
    ]
    if args.keep_temporary_features:
        command.append("--keep-temporary-features")
    if args.face_landmarker is not None:
        command.extend(("--face-landmarker", str(args.face_landmarker.expanduser().resolve())))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not output.is_file():
        raise RuntimeError(f"Inference exited successfully but did not produce: {output}")
    with output.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    print("\nPRAMAAN-X RESULT")
    print(json.dumps(concise_summary(payload), indent=2, sort_keys=True))
    print(f"\nFull forensic JSON: {output}")


if __name__ == "__main__":
    main()
