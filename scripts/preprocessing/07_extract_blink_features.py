"""Stream compact EAR-based blink features from sampled FakeAVCeleb video.

This production extractor never writes eye crops or frames.  It uses the same
MediaPipe landmarks as the smoke-only script 05, persists a small per-video
sequence, and records an explicit NOT_APPLICABLE state when a usable eye
sequence is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "fakeavceleb_manifest_split.csv"
DEFAULT_FRAME_ROOT = PROJECT_ROOT / "data" / "interim" / "frames"
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "blink_features" / "ear_v1"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "interim" / "feature_extraction_logs"
DEFAULT_LANDMARKER = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"

SCHEMA_VERSION = "blink_ear_v1"
FEATURE_NAMES = (
    "left_ear",
    "right_ear",
    "mean_ear",
    "ear_difference",
    "relative_ear_difference",
    "mean_ear_velocity_per_second",
    "absolute_ear_velocity_per_second",
)
FEATURE_DIM = len(FEATURE_NAMES)
REQUIRED_NPZ_KEYS = {
    "schema_version", "sample_id", "availability", "not_applicable_reason",
    "timestamps_sec", "source_frame_indices", "features", "feature_names",
}

# Standard six-point EAR order, shared with preprocessing script 05.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)


class SampleExtractionError(RuntimeError):
    """A recoverable error that applies to one selected sample."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    video_path: Path | None
    frame_dir: Path | None
    semantic_class: str | None
    split: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("video", "frames"), default="video")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--face-landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--min-valid-frames", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def require_runtime() -> dict[str, Any]:
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Activate the ML environment with opencv-python, mediapipe, numpy, and pandas installed."
        ) from exc
    return {"cv2": cv2, "mp": mp, "np": np, "pd": pd}


def parse_sample_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def select_samples(args: argparse.Namespace, runtime: dict[str, Any]) -> list[Sample]:
    selected_ids = parse_sample_ids(args.sample_id)
    if args.input_mode == "frames":
        if not args.frame_root.exists():
            raise FileNotFoundError(f"Frame root not found: {args.frame_root}")
        directories = [path for path in sorted(args.frame_root.iterdir()) if path.is_dir()]
        if selected_ids:
            directories = [path for path in directories if path.name in selected_ids]
        samples = [Sample(path.name, None, path, None, None) for path in directories]
    else:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Split manifest not found: {args.manifest}")
        dataframe = runtime["pd"].read_csv(args.manifest)
        required = {"sample_id", "video_path", "split", "semantic_class"}
        missing = required - set(dataframe.columns)
        if missing:
            raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
        dataframe = dataframe[dataframe["split"] == args.split].copy()
        if selected_ids:
            dataframe = dataframe[dataframe["sample_id"].isin(selected_ids)]
        if args.per_class is not None:
            if args.per_class < 1:
                raise ValueError("--per-class must be at least 1")
            groups = []
            for _, group in dataframe.groupby("semantic_class", sort=True):
                if len(group) < args.per_class:
                    raise RuntimeError("A requested class has fewer rows than --per-class.")
                groups.append(group.sample(n=args.per_class, random_state=args.seed))
            dataframe = runtime["pd"].concat(groups).sort_values("sample_id")
        samples = [
            Sample(row.sample_id, PROJECT_ROOT / row.video_path, None, row.semantic_class, row.split)
            for row in dataframe.itertuples(index=False)
        ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        samples = samples[:args.limit]
    if not samples:
        raise RuntimeError("No samples selected. Nothing was silently skipped.")
    return samples


def iter_cached_frames(frame_dir: Path, fps: float, cv2: Any) -> Iterator[tuple[int, float, Any]]:
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SampleExtractionError(f"No cached JPEG frames found in {frame_dir}")
    for index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise SampleExtractionError(f"Unreadable cached frame: {frame_path}")
        yield index, index / fps, image


def iter_video_frames(video_path: Path, target_fps: float, cv2: Any) -> Iterator[tuple[int, float, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SampleExtractionError(f"Unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        capture.release()
        raise SampleExtractionError(f"Video reported invalid FPS: {video_path}")
    source_index, yielded, next_timestamp = 0, 0, 0.0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            timestamp = source_index / source_fps
            if timestamp + 1e-9 >= next_timestamp:
                yield source_index, timestamp, image
                yielded += 1
                next_timestamp += 1.0 / target_fps
            source_index += 1
    finally:
        capture.release()
    if yielded == 0:
        raise SampleExtractionError(f"No decodable frames sampled from: {video_path}")


def build_landmarker(runtime: dict[str, Any], model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe Face Landmarker model missing: {model_path}")
    mp = runtime["mp"]
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def ear_from_landmarks(landmarks: Any, indices: tuple[int, int, int, int, int, int]) -> float:
    points = [(landmarks[index].x, landmarks[index].y) for index in indices]
    horizontal = _distance(points[0], points[3])
    if horizontal <= 1e-8:
        return float("nan")
    return (_distance(points[1], points[5]) + _distance(points[2], points[4])) / (2.0 * horizontal)


def feature_vector(left_ear: float, right_ear: float, previous_mean: float | None, delta_seconds: float | None) -> tuple[list[float], float]:
    mean_ear = (left_ear + right_ear) / 2.0
    difference = abs(left_ear - right_ear)
    relative_difference = difference / max(mean_ear, 1e-6)
    velocity = 0.0 if previous_mean is None or not delta_seconds or delta_seconds <= 0 else (mean_ear - previous_mean) / delta_seconds
    return [left_ear, right_ear, mean_ear, difference, relative_difference, velocity, abs(velocity)], mean_ear


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_feature_file(path: Path, np: Any) -> tuple[bool, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = REQUIRED_NPZ_KEYS - set(payload.files)
            if missing:
                return False, f"missing keys: {sorted(missing)}"
            if str(payload["schema_version"].item()) != SCHEMA_VERSION:
                return False, "unsupported schema version"
            features = payload["features"]
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                return False, f"invalid feature shape: {features.shape}"
            if len(features) != len(payload["timestamps_sec"]) or len(features) != len(payload["source_frame_indices"]):
                return False, "feature/timestamp/source-index length mismatch"
            if tuple(str(value) for value in payload["feature_names"].tolist()) != FEATURE_NAMES:
                return False, "unexpected feature names"
            availability = str(payload["availability"].item())
            if availability not in {"AVAILABLE", "NOT_APPLICABLE"}:
                return False, "invalid availability"
            if availability == "AVAILABLE" and len(features) == 0:
                return False, "available feature artifact is empty"
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, f"unreadable npz: {exc}"
    return True, None


def write_feature_file(path: Path, payload: dict[str, Any], np: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        valid, reason = validate_feature_file(temporary, np)
        if not valid:
            raise SampleExtractionError(f"Refusing to publish invalid feature file: {reason}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path.stat().st_size


def process_sample(sample: Sample, args: argparse.Namespace, landmarker: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    np, cv2, mp = runtime["np"], runtime["cv2"], runtime["mp"]
    output_path = args.feature_root / f"{sample.sample_id}.npz"
    existing_output_state = "NEW_OUTPUT"
    if output_path.exists() and not args.overwrite:
        valid, reason = validate_feature_file(output_path, np)
        if valid:
            return {"sample_id": sample.sample_id, "status": "SKIPPED_VALID_OUTPUT", "output_path": str(output_path)}
        existing_output_state = "RECOMPUTED_INVALID_OUTPUT"
        logging.warning("Invalid cached output for %s (%s); recomputing.", sample.sample_id, reason)
    elif output_path.exists():
        existing_output_state = "OVERWROTE_EXISTING_OUTPUT"

    frame_iter = iter_cached_frames(sample.frame_dir, args.fps, cv2) if args.input_mode == "frames" else iter_video_frames(sample.video_path, args.fps, cv2)
    timestamps: list[float] = []
    source_indices: list[int] = []
    features: list[list[float]] = []
    decoded_frames = face_detected_frames = 0
    previous_mean: float | None = None
    previous_timestamp: float | None = None
    started = perf_counter()
    for source_index, timestamp, image_bgr in frame_iter:
        decoded_frames += 1
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
        if not result.face_landmarks:
            continue
        face_detected_frames += 1
        landmarks = result.face_landmarks[0]
        left_ear = ear_from_landmarks(landmarks, LEFT_EYE_EAR)
        right_ear = ear_from_landmarks(landmarks, RIGHT_EYE_EAR)
        if not np.isfinite((left_ear, right_ear)).all():
            continue
        vector, previous_mean = feature_vector(left_ear, right_ear, previous_mean, None if previous_timestamp is None else timestamp - previous_timestamp)
        previous_timestamp = timestamp
        timestamps.append(timestamp)
        source_indices.append(source_index)
        features.append(vector)

    if decoded_frames == 0:
        raise SampleExtractionError("No frames were decoded.")
    feature_array = np.asarray(features, dtype=np.float32).reshape((-1, FEATURE_DIM)) if features else np.empty((0, FEATURE_DIM), dtype=np.float32)
    availability = "AVAILABLE" if len(feature_array) >= args.min_valid_frames else "NOT_APPLICABLE"
    reason = "" if availability == "AVAILABLE" else f"INSUFFICIENT_VALID_EYE_FRAMES:{len(feature_array)}/{args.min_valid_frames}"
    bytes_written = write_feature_file(output_path, {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "sample_id": np.asarray(sample.sample_id),
        "availability": np.asarray(availability),
        "not_applicable_reason": np.asarray(reason),
        "timestamps_sec": np.asarray(timestamps, dtype=np.float32),
        "source_frame_indices": np.asarray(source_indices, dtype=np.int32),
        "features": feature_array,
        "feature_names": np.asarray(FEATURE_NAMES),
    }, np)
    elapsed = perf_counter() - started
    return {
        "sample_id": sample.sample_id,
        "status": "COMPLETED",
        "existing_output_state": existing_output_state,
        "input_mode": args.input_mode,
        "split": sample.split,
        "semantic_class": sample.semantic_class,
        "decoded_frames": decoded_frames,
        "face_detected_frames": face_detected_frames,
        "face_detection_failure_rate": 1.0 - face_detected_frames / decoded_frames,
        "blink_modality": availability,
        "not_applicable_reason": reason or None,
        "feature_shape": list(feature_array.shape),
        "elapsed_seconds": elapsed,
        "frames_per_second": decoded_frames / elapsed if elapsed else None,
        "output_bytes": bytes_written,
        "output_path": str(output_path),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.min_valid_frames < 1:
        raise ValueError("--fps and --min-valid-frames must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = require_runtime()
    samples = select_samples(args, runtime)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = args.log_root / f"blink_{run_id}.jsonl"
    logging.info("Selected %d samples, mode=%s", len(samples), args.input_mode)
    records: list[dict[str, Any]] = []
    with build_landmarker(runtime, args.face_landmarker) as landmarker:
        for position, sample in enumerate(samples, start=1):
            logging.info("[%d/%d] %s", position, len(samples), sample.sample_id)
            try:
                record = process_sample(sample, args, landmarker, runtime)
            except Exception as exc:
                record = {"sample_id": sample.sample_id, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=5)}
                logging.exception("Blink feature extraction failed for %s", sample.sample_id)
            record["recorded_at"] = datetime.now(timezone.utc).isoformat()
            append_jsonl(jsonl_path, record)
            records.append(record)
            if record["status"] == "FAILED" and args.fail_fast:
                raise RuntimeError(f"Stopping after failure of {sample.sample_id}")

    completed = [record for record in records if record["status"] == "COMPLETED"]
    skipped = [record for record in records if record["status"] == "SKIPPED_VALID_OUTPUT"]
    failed = [record for record in records if record["status"] == "FAILED"]
    recomputed = [record for record in completed if record["existing_output_state"] == "RECOMPUTED_INVALID_OUTPUT"]
    available = [record for record in completed if record["blink_modality"] == "AVAILABLE"]
    elapsed = sum(record.get("elapsed_seconds", 0.0) for record in completed)
    frames = sum(record.get("decoded_frames", 0) for record in completed)
    output_bytes = sum(record.get("output_bytes", 0) for record in completed)
    summary = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "input_mode": args.input_mode,
        "samples_selected": len(samples),
        "completed": len(completed),
        "completed_new": len(completed) - len(recomputed),
        "completed_recomputed_invalid_output": len(recomputed),
        "skipped_valid_output": len(skipped),
        "failed": len(failed),
        "accounted_samples": len(completed) + len(skipped) + len(failed),
        "available_blink_sequences": len(available),
        "not_applicable_blink_sequences": len(completed) - len(available),
        "failure_rate": len(failed) / len(samples),
        "frames": frames,
        "frames_per_second": frames / elapsed if elapsed else None,
        "videos_per_minute": len(completed) / elapsed * 60 if elapsed else None,
        "mean_output_bytes_per_video": output_bytes / len(completed) if completed else None,
        "per_sample_log": str(jsonl_path),
    }
    summary_path = args.log_root / f"blink_{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Run complete: %s", json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
