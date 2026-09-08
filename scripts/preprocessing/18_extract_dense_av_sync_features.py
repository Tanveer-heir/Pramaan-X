"""Extract compact dense audio features aligned to cached mouth-frame timestamps.

This stage is intentionally separate from the 4.04-second XLSR-SLS anti-spoof
cache. It decodes audio, computes short-context spectral descriptors at each
tracked mouth timestamp, writes a compact artifact, and discards the waveform.
ConvNeXt is not loaded and its mouth embeddings are not duplicated.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import tempfile
from time import perf_counter
import traceback
from typing import Any
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "fakeavceleb_manifest_split.csv"
DEFAULT_SMOKE_FRAME_ROOT = PROJECT_ROOT / "data" / "interim" / "frames"
DEFAULT_VISUAL_ROOT = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "av_sync_features" / "dense_v1"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "interim" / "feature_extraction_logs"
SCHEMA_VERSION = "av_sync_dense_v1"
VISUAL_SCHEMA_VERSION = "visual_embedding_v1"
VISUAL_BACKBONE = "torchvision/convnext_tiny/IMAGENET1K_V1"
SAMPLE_RATE = 16_000
MEL_BANDS = 40
AUDIO_FEATURE_DIM = MEL_BANDS * 2 + 2
AUDIO_DESCRIPTOR = "LOG_MEL_MEAN_STD_RMS_ZCR_V1"
REQUIRED_KEYS = {
    "schema_version", "sample_id", "av_modality", "not_applicable_reason",
    "timestamps_sec", "mouth_frame_indices", "audio_features", "sample_rate",
    "audio_context_samples", "audio_descriptor", "visual_schema_version",
    "visual_backbone", "source_audio_samples", "source_mouth_frames",
}


class SampleExtractionError(RuntimeError):
    """A recoverable error affecting one selected sample."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("video", "smoke"), default="video")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke-frame-root", type=Path, default=DEFAULT_SMOKE_FRAME_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--audio-context-ms", type=float, default=250.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg-timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Activate the project environment with NumPy installed") from exc
    return np


def parse_sample_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "video_path", "split", "semantic_class"}
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"Manifest must contain {sorted(required)}")
    return rows


def select_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = load_rows(args.manifest)
    selected_ids = parse_sample_ids(args.sample_id)
    if args.input_mode == "smoke":
        if not args.smoke_frame_root.exists():
            raise FileNotFoundError(f"Smoke frame root missing: {args.smoke_frame_root}")
        smoke_ids = {path.name for path in args.smoke_frame_root.iterdir() if path.is_dir()}
        rows = [row for row in rows if row["sample_id"] in smoke_ids]
    else:
        rows = [row for row in rows if row["split"] == args.split]
    if selected_ids:
        rows = [row for row in rows if row["sample_id"] in selected_ids]
    if args.per_class is not None:
        if args.per_class < 1:
            raise ValueError("--per-class must be at least 1")
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["semantic_class"], []).append(row)
        rng, selected = random.Random(args.seed), []
        for name in sorted(grouped):
            group = sorted(grouped[name], key=lambda item: item["sample_id"])
            if len(group) < args.per_class:
                raise RuntimeError(f"Class {name} has fewer than {args.per_class} rows")
            selected.extend(rng.sample(group, args.per_class))
        rows = selected
    rows.sort(key=lambda row: row["sample_id"])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError("No samples selected. Nothing was silently skipped.")
    return rows


def has_audio_stream(video_path: Path, args: argparse.Namespace) -> bool:
    command = [
        args.ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.ffmpeg_timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFprobe executable not found: {args.ffprobe}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SampleExtractionError(f"FFprobe timed out after {args.ffmpeg_timeout:.1f}s") from exc
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise SampleExtractionError(f"FFprobe failed with code {result.returncode}: {error[-1000:]}")
    return bool(result.stdout.strip())


def decode_audio(video_path: Path, args: argparse.Namespace, np: Any) -> Any:
    if not has_audio_stream(video_path, args):
        return np.empty((0,), dtype=np.float32)
    command = [
        args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.ffmpeg_timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFmpeg executable not found: {args.ffmpeg}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SampleExtractionError(f"FFmpeg timed out after {args.ffmpeg_timeout:.1f}s") from exc
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise SampleExtractionError(f"FFmpeg failed with code {result.returncode}: {error[-1000:]}")
    if len(result.stdout) % 4:
        raise SampleExtractionError("FFmpeg returned a partial float32 sample")
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not np.isfinite(waveform).all():
        raise SampleExtractionError("Decoded waveform contains non-finite values")
    return waveform


def hz_to_mel(frequency: Any, np: Any) -> Any:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel: Any, np: Any) -> Any:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def build_mel_filterbank(np: Any, fft_size: int = 512) -> Any:
    frequencies = np.linspace(0.0, SAMPLE_RATE / 2.0, fft_size // 2 + 1)
    mel_edges = np.linspace(hz_to_mel(80.0, np), hz_to_mel(7_600.0, np), MEL_BANDS + 2)
    hz_edges = mel_to_hz(mel_edges, np)
    filters = np.zeros((MEL_BANDS, len(frequencies)), dtype=np.float32)
    for index in range(MEL_BANDS):
        left, centre, right = hz_edges[index:index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum((frequencies - left) / max(centre - left, 1e-8), (right - frequencies) / max(right - centre, 1e-8)),
        )
    filters /= np.maximum(filters.sum(axis=1, keepdims=True), 1e-8)
    return filters


def fixed_context(waveform: Any, centre_sample: int, context_samples: int, np: Any) -> Any:
    start = centre_sample - context_samples // 2
    end = start + context_samples
    source_start, source_end = max(0, start), min(len(waveform), end)
    segment = waveform[source_start:source_end]
    left_pad, right_pad = max(0, -start), max(0, end - len(waveform))
    if left_pad or right_pad:
        segment = np.pad(segment, (left_pad, right_pad), mode="constant")
    return segment.astype(np.float32, copy=False)


def audio_descriptor(segment: Any, mel_filterbank: Any, np: Any) -> Any:
    frame_size, hop_size, fft_size = 400, 160, 512
    if len(segment) < frame_size:
        segment = np.pad(segment, (0, frame_size - len(segment)), mode="constant")
    starts = range(0, len(segment) - frame_size + 1, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    spectra = []
    for start in starts:
        frame = segment[start:start + frame_size] * window
        spectra.append(np.abs(np.fft.rfft(frame, n=fft_size)) ** 2)
    power = np.stack(spectra).astype(np.float32)
    log_mel = np.log(np.maximum(power @ mel_filterbank.T, 1e-10))
    rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2) + 1e-12)
    zcr = np.mean(segment[1:] * segment[:-1] < 0) if len(segment) > 1 else 0.0
    return np.concatenate((log_mel.mean(axis=0), log_mel.std(axis=0), [np.log(rms + 1e-8), zcr])).astype(np.float32)


def extract_dense_features(waveform: Any, timestamps: Any, context_samples: int, np: Any) -> tuple[Any, Any]:
    if waveform.size == 0 or len(timestamps) == 0:
        return np.empty((0, AUDIO_FEATURE_DIM), dtype=np.float16), np.empty((0,), dtype=np.int64)
    valid = np.flatnonzero((timestamps >= 0.0) & (timestamps < len(waveform) / SAMPLE_RATE + 1e-6))
    mel_filterbank = build_mel_filterbank(np)
    descriptors = [
        audio_descriptor(fixed_context(waveform, int(round(float(timestamps[index]) * SAMPLE_RATE)), context_samples, np), mel_filterbank, np)
        for index in valid
    ]
    if not descriptors:
        return np.empty((0, AUDIO_FEATURE_DIM), dtype=np.float16), valid.astype(np.int64)
    return np.stack(descriptors).astype(np.float16), valid.astype(np.int64)


def validate_visual(payload: Any) -> str | None:
    required = {"schema_version", "backbone", "sample_id", "timestamps_sec", "mouth_embeddings", "mouth_frame_indices"}
    missing = required - set(payload.files)
    if missing:
        return f"visual_missing_keys:{','.join(sorted(missing))}"
    if str(payload["schema_version"].item()) != VISUAL_SCHEMA_VERSION:
        return "unsupported_visual_schema"
    if str(payload["backbone"].item()) != VISUAL_BACKBONE:
        return "visual_backbone_mismatch"
    mouth = payload["mouth_embeddings"]
    indices = payload["mouth_frame_indices"]
    if mouth.ndim != 2 or mouth.shape[1] != 768 or len(mouth) != len(indices):
        return "invalid_mouth_feature_shape"
    if len(indices) and (indices.min() < 0 or indices.max() >= len(payload["timestamps_sec"])):
        return "mouth_index_out_of_bounds"
    return None


def validate_output(path: Path, np: Any, expected_sample_id: str | None = None, expected_context_samples: int | None = None) -> tuple[bool, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = REQUIRED_KEYS - set(payload.files)
            if missing:
                return False, f"missing_keys:{','.join(sorted(missing))}"
            if str(payload["schema_version"].item()) != SCHEMA_VERSION:
                return False, "unsupported_schema"
            if expected_sample_id is not None and str(payload["sample_id"].item()) != expected_sample_id:
                return False, "sample_id_mismatch"
            if expected_context_samples is not None and int(payload["audio_context_samples"].item()) != expected_context_samples:
                return False, "audio_context_mismatch"
            if str(payload["audio_descriptor"].item()) != AUDIO_DESCRIPTOR:
                return False, "audio_descriptor_mismatch"
            if str(payload["visual_schema_version"].item()) != VISUAL_SCHEMA_VERSION or str(payload["visual_backbone"].item()) != VISUAL_BACKBONE:
                return False, "visual_provenance_mismatch"
            count = len(payload["audio_features"])
            if payload["audio_features"].shape != (count, AUDIO_FEATURE_DIM):
                return False, f"invalid_audio_feature_shape:{payload['audio_features'].shape}"
            if len(payload["timestamps_sec"]) != count or len(payload["mouth_frame_indices"]) != count:
                return False, "aligned_length_mismatch"
            if not np.isfinite(payload["audio_features"]).all():
                return False, "non_finite_audio_features"
            modality = str(payload["av_modality"].item())
            if modality == "AVAILABLE" and count == 0:
                return False, "available_without_features"
            if modality == "NOT_APPLICABLE" and count != 0:
                return False, "not_applicable_with_features"
            if modality not in {"AVAILABLE", "NOT_APPLICABLE"}:
                return False, "invalid_modality"
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, f"unreadable_npz:{type(exc).__name__}:{exc}"
    return True, None


def atomic_write_npz(path: Path, payload: dict[str, Any], np: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        valid, reason = validate_output(temporary, np)
        if not valid:
            raise SampleExtractionError(f"Refusing invalid output: {reason}")
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path.stat().st_size


def process_row(row: dict[str, str], args: argparse.Namespace, context_samples: int, np: Any) -> dict[str, Any]:
    sample_id = row["sample_id"]
    output_path = args.output_root / f"{sample_id}.npz"
    existing_state = "NEW_OUTPUT"
    if output_path.exists() and not args.overwrite:
        valid, reason = validate_output(output_path, np, sample_id, context_samples)
        if valid:
            return {"sample_id": sample_id, "status": "SKIPPED_VALID_OUTPUT", "output_path": str(output_path)}
        existing_state = "RECOMPUTED_INVALID_OUTPUT"
        logging.warning("Invalid cached output for %s (%s); recomputing", sample_id, reason)
    elif output_path.exists():
        existing_state = "OVERWROTE_EXISTING_OUTPUT"
    visual_path = args.visual_root / f"{sample_id}.npz"
    if not visual_path.exists():
        raise SampleExtractionError(f"Missing expected visual feature: {visual_path}")
    started = perf_counter()
    with np.load(visual_path, allow_pickle=False) as visual:
        reason = validate_visual(visual)
        if reason is not None:
            raise SampleExtractionError(reason)
        if str(visual["sample_id"].item()) != sample_id:
            raise SampleExtractionError("Visual sample ID mismatch")
        mouth_indices = np.asarray(visual["mouth_frame_indices"], dtype=np.int64)
        mouth_timestamps = np.asarray(visual["timestamps_sec"], dtype=np.float64)[mouth_indices]
    video_path = PROJECT_ROOT / row["video_path"]
    waveform = decode_audio(video_path, args, np)
    features, valid_positions = extract_dense_features(waveform, mouth_timestamps, context_samples, np)
    selected_indices = mouth_indices[valid_positions]
    selected_timestamps = mouth_timestamps[valid_positions]
    if waveform.size == 0:
        modality, not_applicable_reason = "NOT_APPLICABLE", "AUDIO_NOT_APPLICABLE"
    elif len(mouth_indices) == 0:
        modality, not_applicable_reason = "NOT_APPLICABLE", "MOUTH_NOT_APPLICABLE"
    elif len(features) == 0:
        modality, not_applicable_reason = "NOT_APPLICABLE", "NO_SHARED_AUDIO_MOUTH_TIMESTAMPS"
    else:
        modality, not_applicable_reason = "AVAILABLE", ""
    output_bytes = atomic_write_npz(output_path, {
        "schema_version": np.asarray(SCHEMA_VERSION), "sample_id": np.asarray(sample_id),
        "av_modality": np.asarray(modality), "not_applicable_reason": np.asarray(not_applicable_reason),
        "timestamps_sec": selected_timestamps.astype(np.float32),
        "mouth_frame_indices": selected_indices.astype(np.int32), "audio_features": features,
        "sample_rate": np.asarray(SAMPLE_RATE, dtype=np.int32),
        "audio_context_samples": np.asarray(context_samples, dtype=np.int32),
        "audio_descriptor": np.asarray(AUDIO_DESCRIPTOR),
        "visual_schema_version": np.asarray(VISUAL_SCHEMA_VERSION), "visual_backbone": np.asarray(VISUAL_BACKBONE),
        "source_audio_samples": np.asarray(len(waveform), dtype=np.int64),
        "source_mouth_frames": np.asarray(len(mouth_indices), dtype=np.int32),
    }, np)
    return {
        "sample_id": sample_id, "status": "COMPLETED", "existing_output_state": existing_state,
        "split": row["split"], "semantic_class": row["semantic_class"], "av_modality": modality,
        "not_applicable_reason": not_applicable_reason, "source_audio_seconds": len(waveform) / SAMPLE_RATE,
        "source_mouth_frames": len(mouth_indices), "aligned_frames": len(features),
        "audio_feature_shape": list(features.shape), "output_bytes": output_bytes,
        "elapsed_seconds": perf_counter() - started, "output_path": str(output_path),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.audio_context_ms <= 0 or args.ffmpeg_timeout <= 0:
        raise ValueError("Audio context and FFmpeg timeout must be positive")
    context_samples = int(round(args.audio_context_ms / 1000.0 * SAMPLE_RATE))
    if context_samples < 400:
        raise ValueError("Audio context must be at least 25 ms")
    np = require_numpy()
    rows = select_rows(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log_root / f"av_sync_dense_{run_id}.jsonl"
    records, started = [], perf_counter()
    for position, row in enumerate(rows, 1):
        logging.info("[%d/%d] %s", position, len(rows), row["sample_id"])
        try:
            record = process_row(row, args, context_samples, np)
        except Exception as exc:
            record = {
                "sample_id": row["sample_id"], "status": "FAILED", "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(limit=5),
            }
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        append_jsonl(log_path, record)
        records.append(record)
        if record["status"] == "FAILED" and args.fail_fast:
            raise RuntimeError(f"Stopping after failure of {row['sample_id']}")
    elapsed = perf_counter() - started
    completed = [record for record in records if record["status"] == "COMPLETED"]
    skipped = [record for record in records if record["status"] == "SKIPPED_VALID_OUTPUT"]
    failed = [record for record in records if record["status"] == "FAILED"]
    available = [record for record in completed if record["av_modality"] == "AVAILABLE"]
    not_applicable = [record for record in completed if record["av_modality"] == "NOT_APPLICABLE"]
    recomputed = [record for record in completed if record["existing_output_state"] == "RECOMPUTED_INVALID_OUTPUT"]
    total_frames = sum(record.get("aligned_frames", 0) for record in completed)
    total_audio_seconds = sum(record.get("source_audio_seconds", 0.0) for record in completed)
    total_bytes = sum(record.get("output_bytes", 0) for record in completed)
    processing_seconds = sum(record.get("elapsed_seconds", 0.0) for record in completed)
    report = {
        "run_id": run_id, "schema_version": SCHEMA_VERSION, "audio_descriptor": AUDIO_DESCRIPTOR,
        "audio_context_ms": args.audio_context_ms, "input_mode": args.input_mode,
        "samples_selected": len(rows), "completed": len(completed),
        "completed_new": len(completed) - len(recomputed), "completed_recomputed_invalid_output": len(recomputed),
        "skipped_valid_output": len(skipped), "failed": len(failed),
        "accounted_samples": len(completed) + len(skipped) + len(failed), "failure_rate": len(failed) / len(rows),
        "available_av": len(available), "not_applicable_av": len(not_applicable),
        "aligned_frames": total_frames, "audio_seconds": total_audio_seconds,
        "processing_seconds": processing_seconds, "elapsed_seconds": elapsed,
        "frames_per_second": total_frames / processing_seconds if processing_seconds else None,
        "videos_per_minute": len(completed) / processing_seconds * 60 if processing_seconds else None,
        "mean_output_bytes_per_video": total_bytes / len(completed) if completed else None,
        "per_sample_log": str(log_path),
    }
    atomic_json(args.log_root / f"av_sync_dense_{run_id}_summary.json", report)
    logging.info("Run complete: %s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
