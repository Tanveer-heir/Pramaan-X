"""Extract compact video compression and re-encoding forensic features.

The extractor is CPU-only. It combines FFprobe GOP and packet statistics with
sampled-frame blockiness, sharpness, residual, and duplication measurements.
No decoded frames are persisted. Outputs are validated and atomically published.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
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
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "compression_features" / "video_v1"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "interim" / "feature_extraction_logs"
SCHEMA_VERSION = "compression_video_v1"

FEATURE_NAMES = (
    "log1p_bits_per_pixel_frame",
    "keyframe_ratio",
    "gop_mean_seconds",
    "gop_std_seconds",
    "gop_max_seconds",
    "packet_size_cv",
    "packet_size_p90_over_mean",
    "frame_interval_cv",
    "i_frame_ratio",
    "p_frame_ratio",
    "b_frame_ratio",
    "blockiness_ratio_mean",
    "blockiness_ratio_std",
    "log1p_laplacian_variance_mean",
    "laplacian_variance_cv",
    "noise_residual_mean",
    "noise_residual_cv",
    "near_duplicate_rate",
    "frame_difference_mean",
    "frame_difference_cv",
)
FEATURE_DIM = len(FEATURE_NAMES)
REQUIRED_NPZ_KEYS = {
    "schema_version", "sample_id", "availability", "features", "feature_names",
    "codec_name", "pixel_format", "duration_seconds", "source_width", "source_height",
    "source_fps", "sampled_frame_count", "probed_frame_count",
}


class SampleExtractionError(RuntimeError):
    """A recoverable error limited to one selected media sample."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    video_path: Path
    semantic_class: str
    split: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffprobe-timeout", type=float, default=60.0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.0025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def require_runtime() -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Activate the project environment with opencv-python, numpy, and pandas installed.") from exc
    return {"cv2": cv2, "np": np, "pd": pd}


def parse_sample_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def select_samples(args: argparse.Namespace, runtime: dict[str, Any]) -> list[Sample]:
    if not args.manifest.exists():
        raise FileNotFoundError(f"Split manifest not found: {args.manifest}")
    dataframe = runtime["pd"].read_csv(args.manifest)
    required = {"sample_id", "video_path", "split", "semantic_class"}
    missing = required - set(dataframe.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    dataframe = dataframe[dataframe["split"] == args.split].copy()
    selected_ids = parse_sample_ids(args.sample_id)
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
        Sample(str(row.sample_id), PROJECT_ROOT / row.video_path, str(row.semantic_class), str(row.split))
        for row in dataframe.itertuples(index=False)
    ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        samples = samples[:args.limit]
    if not samples:
        raise RuntimeError("No samples selected. Nothing was silently skipped.")
    return samples


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def fraction(value: Any) -> float:
    if value in (None, "", "N/A", "0/0"):
        return 0.0
    text = str(value)
    if "/" not in text:
        return finite_float(text)
    numerator, denominator = text.split("/", 1)
    denominator_value = finite_float(denominator)
    return finite_float(numerator) / denominator_value if denominator_value else 0.0


def coefficient_of_variation(values: Any, np: Any) -> float:
    if len(values) == 0:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if abs(mean) > 1e-12 else 0.0


def run_ffprobe(video_path: Path, executable: str, timeout: float) -> dict[str, Any]:
    command = [
        executable,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,avg_frame_rate,r_frame_rate,duration,bit_rate,nb_frames:"
        "format=duration,size,bit_rate,format_name:"
        "frame=key_frame,pkt_size,best_effort_timestamp_time,pkt_duration_time,pict_type",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFprobe executable not found: {executable}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SampleExtractionError(f"FFprobe timed out after {timeout:.1f}s") from exc
    if result.returncode != 0:
        message = result.stderr.strip()[-1000:]
        raise SampleExtractionError(f"FFprobe failed with exit code {result.returncode}: {message}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SampleExtractionError(f"FFprobe returned invalid JSON: {exc}") from exc
    if not payload.get("streams"):
        raise SampleExtractionError("FFprobe found no video stream.")
    return payload


def probe_statistics(payload: dict[str, Any], file_size: int, np: Any) -> dict[str, Any]:
    stream = payload["streams"][0]
    format_info = payload.get("format", {})
    frames = payload.get("frames", [])
    width = int(finite_float(stream.get("width")))
    height = int(finite_float(stream.get("height")))
    fps = fraction(stream.get("avg_frame_rate")) or fraction(stream.get("r_frame_rate"))
    duration = finite_float(stream.get("duration")) or finite_float(format_info.get("duration"))
    bitrate = finite_float(stream.get("bit_rate")) or finite_float(format_info.get("bit_rate"))
    if bitrate <= 0 and duration > 0:
        bitrate = file_size * 8.0 / duration
    if width <= 0 or height <= 0 or fps <= 0 or duration <= 0:
        raise SampleExtractionError(f"Invalid core video metadata: width={width}, height={height}, fps={fps}, duration={duration}")

    packet_sizes = np.asarray([finite_float(frame.get("pkt_size")) for frame in frames if finite_float(frame.get("pkt_size")) > 0], dtype=np.float64)
    timestamps = np.asarray([finite_float(frame.get("best_effort_timestamp_time"), float("nan")) for frame in frames], dtype=np.float64)
    timestamps = timestamps[np.isfinite(timestamps)]
    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]
    key_indices = np.asarray([index for index, frame in enumerate(frames) if int(finite_float(frame.get("key_frame"))) == 1], dtype=np.int64)
    gop_frames = np.diff(key_indices).astype(np.float64) if len(key_indices) > 1 else np.asarray([max(1.0, duration * fps)], dtype=np.float64)
    frame_types = [str(frame.get("pict_type", "")).upper() for frame in frames]
    probed_count = len(frames)
    denominator = max(probed_count, 1)
    packet_mean = float(np.mean(packet_sizes)) if len(packet_sizes) else max(1.0, bitrate / (8.0 * fps))
    return {
        "codec_name": str(stream.get("codec_name", "unknown")),
        "pixel_format": str(stream.get("pix_fmt", "unknown")),
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "bitrate": bitrate,
        "reported_frame_count": int(finite_float(stream.get("nb_frames"))),
        "probed_frame_count": probed_count,
        "keyframe_ratio": float(len(key_indices) / denominator),
        "gop_mean_seconds": float(np.mean(gop_frames) / fps),
        "gop_std_seconds": float(np.std(gop_frames) / fps),
        "gop_max_seconds": float(np.max(gop_frames) / fps),
        "packet_size_cv": coefficient_of_variation(packet_sizes, np),
        "packet_size_p90_over_mean": float(np.percentile(packet_sizes, 90) / packet_mean) if len(packet_sizes) else 1.0,
        "frame_interval_cv": coefficient_of_variation(intervals, np),
        "i_frame_ratio": frame_types.count("I") / denominator,
        "p_frame_ratio": frame_types.count("P") / denominator,
        "b_frame_ratio": frame_types.count("B") / denominator,
    }


def iter_video_frames(video_path: Path, target_fps: float, cv2: Any) -> Iterator[Any]:
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
                yield image
                yielded += 1
                next_timestamp += 1.0 / target_fps
            source_index += 1
    finally:
        capture.release()
    if yielded == 0:
        raise SampleExtractionError(f"No decodable frames sampled from: {video_path}")


def blockiness_ratio(gray: Any, np: Any) -> float:
    gray = gray.astype(np.float32, copy=False)
    vertical_all = np.abs(np.diff(gray, axis=1))
    horizontal_all = np.abs(np.diff(gray, axis=0))
    general = (float(vertical_all.mean()) + float(horizontal_all.mean())) / 2.0
    vertical_boundaries = np.abs(gray[:, 8::8] - gray[:, 7:-1:8]) if gray.shape[1] >= 16 else np.empty((0,))
    horizontal_boundaries = np.abs(gray[8::8, :] - gray[7:-1:8, :]) if gray.shape[0] >= 16 else np.empty((0,))
    boundary_parts = [part.reshape(-1) for part in (vertical_boundaries, horizontal_boundaries) if part.size]
    if not boundary_parts:
        return 0.0
    boundary = float(np.concatenate(boundary_parts).mean())
    return boundary / max(general, 1e-6)


def frame_statistics(video_path: Path, target_fps: float, duplicate_threshold: float, runtime: dict[str, Any]) -> dict[str, float]:
    cv2, np = runtime["cv2"], runtime["np"]
    blockiness_values: list[float] = []
    laplacian_values: list[float] = []
    residual_values: list[float] = []
    frame_differences: list[float] = []
    previous_small = None
    for image in iter_video_frames(video_path, target_fps, cv2):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blockiness_values.append(blockiness_ratio(gray, np))
        laplacian_values.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        residual_values.append(float(np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32))) / 255.0))
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if previous_small is not None:
            frame_differences.append(float(np.mean(np.abs(small - previous_small))))
        previous_small = small
    blockiness = np.asarray(blockiness_values, dtype=np.float64)
    laplacian = np.asarray(laplacian_values, dtype=np.float64)
    residual = np.asarray(residual_values, dtype=np.float64)
    differences = np.asarray(frame_differences, dtype=np.float64)
    return {
        "sampled_frame_count": len(blockiness_values),
        "blockiness_ratio_mean": float(blockiness.mean()),
        "blockiness_ratio_std": float(blockiness.std()),
        "laplacian_variance_mean": float(laplacian.mean()),
        "laplacian_variance_cv": coefficient_of_variation(laplacian, np),
        "noise_residual_mean": float(residual.mean()),
        "noise_residual_cv": coefficient_of_variation(residual, np),
        "near_duplicate_rate": float(np.mean(differences <= duplicate_threshold)) if len(differences) else 0.0,
        "frame_difference_mean": float(differences.mean()) if len(differences) else 0.0,
        "frame_difference_cv": coefficient_of_variation(differences, np),
    }


def build_feature_vector(probe: dict[str, Any], frame: dict[str, float], np: Any) -> Any:
    bits_per_pixel_frame = probe["bitrate"] / (probe["width"] * probe["height"] * probe["fps"])
    values = (
        math.log1p(bits_per_pixel_frame),
        probe["keyframe_ratio"], probe["gop_mean_seconds"], probe["gop_std_seconds"], probe["gop_max_seconds"],
        probe["packet_size_cv"], probe["packet_size_p90_over_mean"], probe["frame_interval_cv"],
        probe["i_frame_ratio"], probe["p_frame_ratio"], probe["b_frame_ratio"],
        frame["blockiness_ratio_mean"], frame["blockiness_ratio_std"],
        math.log1p(frame["laplacian_variance_mean"]), frame["laplacian_variance_cv"],
        frame["noise_residual_mean"], frame["noise_residual_cv"], frame["near_duplicate_rate"],
        frame["frame_difference_mean"], frame["frame_difference_cv"],
    )
    features = np.asarray(values, dtype=np.float32)
    if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
        raise SampleExtractionError(f"Invalid compression feature vector: shape={features.shape}")
    return features


def validate_feature_file(path: Path, np: Any) -> tuple[bool, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = REQUIRED_NPZ_KEYS - set(payload.files)
            if missing:
                return False, f"missing keys: {sorted(missing)}"
            if str(payload["schema_version"].item()) != SCHEMA_VERSION:
                return False, "unsupported schema version"
            if str(payload["availability"].item()) != "AVAILABLE":
                return False, "compression modality must be available for a completed video artifact"
            if payload["features"].shape != (FEATURE_DIM,):
                return False, f"invalid feature shape: {payload['features'].shape}"
            if not np.isfinite(payload["features"]).all():
                return False, "non-finite compression features"
            if tuple(str(value) for value in payload["feature_names"].tolist()) != FEATURE_NAMES:
                return False, "unexpected feature names"
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, f"unreadable npz: {exc}"
    return True, None


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def process_sample(sample: Sample, args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    np = runtime["np"]
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
    if not sample.video_path.exists():
        raise SampleExtractionError(f"Video does not exist: {sample.video_path}")
    started = perf_counter()
    probe = probe_statistics(run_ffprobe(sample.video_path, args.ffprobe, args.ffprobe_timeout), sample.video_path.stat().st_size, np)
    frame = frame_statistics(sample.video_path, args.fps, args.duplicate_threshold, runtime)
    features = build_feature_vector(probe, frame, np)
    bytes_written = write_feature_file(output_path, {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "sample_id": np.asarray(sample.sample_id),
        "availability": np.asarray("AVAILABLE"),
        "features": features,
        "feature_names": np.asarray(FEATURE_NAMES),
        "codec_name": np.asarray(probe["codec_name"]),
        "pixel_format": np.asarray(probe["pixel_format"]),
        "duration_seconds": np.asarray(probe["duration"], dtype=np.float32),
        "source_width": np.asarray(probe["width"], dtype=np.int32),
        "source_height": np.asarray(probe["height"], dtype=np.int32),
        "source_fps": np.asarray(probe["fps"], dtype=np.float32),
        "sampled_frame_count": np.asarray(frame["sampled_frame_count"], dtype=np.int32),
        "probed_frame_count": np.asarray(probe["probed_frame_count"], dtype=np.int32),
    }, np)
    elapsed = perf_counter() - started
    return {
        "sample_id": sample.sample_id,
        "status": "COMPLETED",
        "existing_output_state": existing_output_state,
        "split": sample.split,
        "semantic_class": sample.semantic_class,
        "codec_name": probe["codec_name"],
        "pixel_format": probe["pixel_format"],
        "sampled_frames": frame["sampled_frame_count"],
        "probed_frames": probe["probed_frame_count"],
        "feature_shape": list(features.shape),
        "elapsed_seconds": elapsed,
        "output_bytes": bytes_written,
        "output_path": str(output_path),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.ffprobe_timeout <= 0 or args.duplicate_threshold < 0:
        raise ValueError("fps and ffprobe-timeout must be positive; duplicate-threshold must be non-negative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = require_runtime()
    samples = select_samples(args, runtime)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = args.log_root / f"compression_{run_id}.jsonl"
    records: list[dict[str, Any]] = []
    logging.info("Selected %d samples, CPU-only compression extraction", len(samples))
    for position, sample in enumerate(samples, start=1):
        logging.info("[%d/%d] %s", position, len(samples), sample.sample_id)
        try:
            record = process_sample(sample, args, runtime)
        except Exception as exc:
            record = {"sample_id": sample.sample_id, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=5)}
            logging.exception("Compression extraction failed for %s", sample.sample_id)
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        append_jsonl(jsonl_path, record)
        records.append(record)
        if record["status"] == "FAILED" and args.fail_fast:
            raise RuntimeError(f"Stopping after failure of {sample.sample_id}")

    completed = [record for record in records if record["status"] == "COMPLETED"]
    skipped = [record for record in records if record["status"] == "SKIPPED_VALID_OUTPUT"]
    failed = [record for record in records if record["status"] == "FAILED"]
    recomputed = [record for record in completed if record["existing_output_state"] == "RECOMPUTED_INVALID_OUTPUT"]
    elapsed = sum(record.get("elapsed_seconds", 0.0) for record in completed)
    output_bytes = sum(record.get("output_bytes", 0) for record in completed)
    sampled_frames = sum(record.get("sampled_frames", 0) for record in completed)
    summary = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "samples_selected": len(samples),
        "completed": len(completed),
        "completed_new": len(completed) - len(recomputed),
        "completed_recomputed_invalid_output": len(recomputed),
        "skipped_valid_output": len(skipped),
        "failed": len(failed),
        "accounted_samples": len(completed) + len(skipped) + len(failed),
        "failure_rate": len(failed) / len(samples),
        "sampled_frames": sampled_frames,
        "frames_per_second": sampled_frames / elapsed if elapsed else None,
        "videos_per_minute": len(completed) / elapsed * 60.0 if elapsed else None,
        "mean_output_bytes_per_video": output_bytes / len(completed) if completed else None,
        "per_sample_log": str(jsonl_path),
    }
    summary_path = args.log_root / f"compression_{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Run complete: %s", json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
