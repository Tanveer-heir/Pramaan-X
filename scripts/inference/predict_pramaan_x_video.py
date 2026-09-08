"""Run the final Pramaan-X fusion model on one raw video.

This wrapper creates disposable feature caches, invokes the existing frozen
branch extractors unchanged, and then calls the cache-based fusion predictor.
It never writes into the accepted feature-cache roots or changes a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VISUAL_EXTRACTOR = ROOT / "scripts" / "preprocessing" / "06_extract_convnext_features.py"
AUDIO_EXTRACTOR = ROOT / "scripts" / "preprocessing" / "13_extract_xlsr_sls_features.py"
DENSE_EXTRACTOR = ROOT / "scripts" / "preprocessing" / "18_extract_dense_av_sync_features.py"
CACHE_PREDICTOR = ROOT / "scripts" / "inference" / "predict_pramaan_x.py"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--video", type=Path, required=True, help="Explicit path to one input video.")
    result.add_argument("--fusion-checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    result.add_argument("--sample-id", help="Optional output-only identifier. Defaults to a video-content hash.")
    result.add_argument("--temp-root", type=Path, help="Parent directory for disposable per-video features.")
    result.add_argument("--keep-temporary-features", action="store_true", help="Keep temporary features for debugging.")
    result.add_argument("--face-landmarker", type=Path, default=ROOT / "models" / "mediapipe" / "face_landmarker.task")
    result.add_argument("--fps", type=float, default=4.0)
    result.add_argument("--visual-batch-size", type=int, default=32)
    result.add_argument("--audio-batch-size", type=int, choices=(1, 2), default=1)
    result.add_argument("--gpu-memory-limit-gb", type=float, default=10.0)
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--ffprobe", default="ffprobe")
    result.add_argument("--ffmpeg-timeout", type=float, default=180.0)
    result.add_argument("--visual-checkpoint", type=Path)
    result.add_argument("--audio-checkpoint", type=Path)
    result.add_argument("--av-checkpoint", type=Path)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_sample_id(video: Path, requested: str | None) -> str:
    if requested:
        return requested
    return f"raw_{sha256(video)[:16]}"


def write_manifest(path: Path, sample_id: str, video: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "video_path", "split", "semantic_class"))
        writer.writeheader()
        writer.writerow({"sample_id": sample_id, "video_path": str(video.resolve()), "split": "raw_video", "semantic_class": "UNKNOWN"})


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def device_for_audio(requested: str) -> str:
    """The existing ONNX extractor accepts cuda but not cuda:ordinal."""
    return "cuda" if requested.startswith("cuda") else requested


def run_step(name: str, command: list[str]) -> dict[str, Any]:
    started = perf_counter()
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        return {"status": "FAILED", "reason": f"{name}_LAUNCH_FAILED:{type(exc).__name__}", "elapsed_seconds": perf_counter() - started}
    if completed.returncode == 0:
        return {"status": "COMPLETED", "elapsed_seconds": perf_counter() - started}
    detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")[-600:]
    return {"status": "FAILED", "reason": f"{name}_FAILED:{detail or 'nonzero_exit'}", "elapsed_seconds": perf_counter() - started}


def extraction_commands(args: argparse.Namespace, workspace: Path, sample_id: str) -> dict[str, list[str]]:
    manifest = workspace / "input_manifest.csv"
    visual_root, audio_root, dense_root = workspace / "visual", workspace / "audio", workspace / "dense"
    common = ["--manifest", str(manifest), "--split", "raw_video", "--sample-id", sample_id, "--overwrite", "--fail-fast"]
    return {
        "visual": [
            sys.executable, str(VISUAL_EXTRACTOR), "--input-mode", "video", *common,
            "--feature-root", str(visual_root), "--log-root", str(workspace / "logs_visual"),
            "--face-landmarker", str(args.face_landmarker), "--fps", str(args.fps),
            "--batch-size", str(args.visual_batch_size), "--device", args.device,
        ],
        "audio": [
            sys.executable, str(AUDIO_EXTRACTOR), "--input-mode", "video", *common,
            "--feature-root", str(audio_root), "--log-root", str(workspace / "logs_audio"),
            "--augmented-model", str(workspace / "xlsr_sls_with_embedding.onnx"),
            "--batch-size", str(args.audio_batch_size), "--device", device_for_audio(args.device),
            "--gpu-memory-limit-gb", str(args.gpu_memory_limit_gb), "--ffmpeg", args.ffmpeg,
            "--ffprobe", args.ffprobe, "--ffmpeg-timeout", str(args.ffmpeg_timeout),
        ],
        "dense_av": [
            sys.executable, str(DENSE_EXTRACTOR), "--input-mode", "video", *common,
            "--visual-root", str(visual_root), "--output-root", str(dense_root),
            "--log-root", str(workspace / "logs_dense"), "--ffmpeg", args.ffmpeg,
            "--ffprobe", args.ffprobe, "--ffmpeg-timeout", str(args.ffmpeg_timeout),
        ],
    }


def main() -> None:
    args = parser().parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if args.fps <= 0 or args.visual_batch_size < 1 or args.ffmpeg_timeout <= 0:
        raise ValueError("FPS, visual batch size, and FFmpeg timeout must be positive")
    required = (VISUAL_EXTRACTOR, AUDIO_EXTRACTOR, DENSE_EXTRACTOR, CACHE_PREDICTOR)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required project scripts are missing: {missing}")

    sample_id = raw_sample_id(args.video, args.sample_id)
    if args.temp_root is not None:
        args.temp_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="pramaan_x_raw_", dir=args.temp_root))
    visual_path, audio_path, dense_path = (workspace / "visual" / f"{sample_id}.npz", workspace / "audio" / f"{sample_id}.npz", workspace / "dense" / f"{sample_id}.npz")
    status: dict[str, Any] = {}
    try:
        write_manifest(workspace / "input_manifest.csv", sample_id, args.video)
        commands = extraction_commands(args, workspace, sample_id)
        status["visual"] = run_step("VISUAL_EXTRACTION", commands["visual"])
        status["audio"] = run_step("AUDIO_EXTRACTION", commands["audio"])
        if status["visual"]["status"] == "COMPLETED" and visual_path.is_file():
            status["dense_av"] = run_step("DENSE_AV_EXTRACTION", commands["dense_av"])
        else:
            status["dense_av"] = {"status": "SKIPPED", "reason": "VISUAL_EXTRACTION_UNAVAILABLE"}

        prediction_path = workspace / "prediction.json"
        command = [
            sys.executable, str(CACHE_PREDICTOR), "--fusion-checkpoint", str(args.fusion_checkpoint),
            "--sample-id", sample_id, "--visual-root", str(workspace / "visual"),
            "--audio-root", str(workspace / "audio"), "--dense-root", str(workspace / "dense"),
            "--device", args.device, "--output", str(prediction_path),
        ]
        for option, path in (("--visual-checkpoint", args.visual_checkpoint), ("--audio-checkpoint", args.audio_checkpoint), ("--av-checkpoint", args.av_checkpoint)):
            if path is not None:
                command.extend((option, str(path)))
        prediction = run_step("FUSION_PREDICTION", command)
        status["fusion"] = prediction
        if prediction["status"] != "COMPLETED" or not prediction_path.is_file():
            raise RuntimeError(f"No usable fusion prediction was produced: {json.dumps(status, sort_keys=True)}")
        with prediction_path.open(encoding="utf-8") as handle:
            output = json.load(handle)
        output["raw_video"] = {
            "path": str(args.video.resolve()), "sha256": sha256(args.video), "bytes": args.video.stat().st_size,
            "temporary_feature_policy": "RETAINED_FOR_DEBUGGING" if args.keep_temporary_features else "DELETED_AFTER_PREDICTION",
            "extraction": status,
        }
        if args.keep_temporary_features:
            output["raw_video"]["temporary_feature_directory"] = str(workspace)
        atomic_json(args.output, output)
        print(json.dumps(output, indent=2, sort_keys=True))
    finally:
        if not args.keep_temporary_features:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
