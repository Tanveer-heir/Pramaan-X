"""Build compact timestamp-aligned audio and mouth-motion features.

This stage reads cached XLSR-SLS and ConvNeXt artifacts only. It does not load
either pretrained backbone and does not assign authenticity labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter
import traceback
import zipfile
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "fakeavceleb_manifest_split.csv"
DEFAULT_SMOKE_FRAME_ROOT = PROJECT_ROOT / "data" / "interim" / "frames"
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "data" / "interim" / "audio_embeddings" / "xlsr_sls_v1"
DEFAULT_VISUAL_ROOT = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "av_sync_features" / "aligned_v1"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "interim" / "feature_extraction_logs"

SCHEMA_VERSION = "av_sync_aligned_v1"
AUDIO_SCHEMA_VERSION = "audio_embedding_v1"
VISUAL_SCHEMA_VERSION = "visual_embedding_v1"
AUDIO_MODEL_SHA256 = "0aa36298a1a893bfb58a8d721dd30d046157f559da1165f95d2a0b69988c1bbd"
VISUAL_BACKBONE = "torchvision/convnext_tiny/IMAGENET1K_V1"
AUDIO_DIM = 1_024
MOUTH_DIM = 768
REQUIRED_KEYS = {
    "schema_version", "sample_id", "av_modality", "not_applicable_reason",
    "timestamps_start_sec", "timestamps_end_sec", "audio_embeddings",
    "mouth_mean_embeddings", "mouth_motion_embeddings", "mouth_frame_counts",
    "source_audio_windows", "source_mouth_frames", "minimum_mouth_frames",
    "audio_model_sha256", "visual_backbone",
}


class SampleAlignmentError(RuntimeError):
    """A recoverable failure affecting one selected sample."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("video", "smoke"), default="video")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke-frame-root", type=Path, default=DEFAULT_SMOKE_FRAME_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--minimum-mouth-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install NumPy in the project virtual environment") from exc
    return np


def parse_sample_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "split", "semantic_class"}
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


def validate_source_payloads(audio: Any, visual: Any) -> tuple[bool, str | None]:
    audio_required = {"schema_version", "model_sha256", "sample_id", "audio_modality", "timestamps_start_sec", "timestamps_end_sec", "embeddings"}
    visual_required = {"schema_version", "backbone", "sample_id", "timestamps_sec", "mouth_embeddings", "mouth_frame_indices"}
    missing_audio = audio_required - set(audio.files)
    missing_visual = visual_required - set(visual.files)
    if missing_audio:
        return False, f"audio_missing_keys:{','.join(sorted(missing_audio))}"
    if missing_visual:
        return False, f"visual_missing_keys:{','.join(sorted(missing_visual))}"
    if str(audio["schema_version"].item()) != AUDIO_SCHEMA_VERSION:
        return False, "unsupported_audio_schema"
    if str(visual["schema_version"].item()) != VISUAL_SCHEMA_VERSION:
        return False, "unsupported_visual_schema"
    if str(audio["model_sha256"].item()) != AUDIO_MODEL_SHA256:
        return False, "audio_model_digest_mismatch"
    if str(visual["backbone"].item()) != VISUAL_BACKBONE:
        return False, "visual_backbone_mismatch"
    if audio["embeddings"].ndim != 2 or audio["embeddings"].shape[1] != AUDIO_DIM:
        return False, f"invalid_audio_shape:{audio['embeddings'].shape}"
    if visual["mouth_embeddings"].ndim != 2 or visual["mouth_embeddings"].shape[1] != MOUTH_DIM:
        return False, f"invalid_mouth_shape:{visual['mouth_embeddings'].shape}"
    if len(audio["embeddings"]) != len(audio["timestamps_start_sec"]) or len(audio["embeddings"]) != len(audio["timestamps_end_sec"]):
        return False, "audio_timestamp_length_mismatch"
    if len(visual["mouth_embeddings"]) != len(visual["mouth_frame_indices"]):
        return False, "mouth_index_length_mismatch"
    indices = visual["mouth_frame_indices"]
    if len(indices) and (indices.min() < 0 or indices.max() >= len(visual["timestamps_sec"])):
        return False, "mouth_index_out_of_bounds"
    return True, None


def align_payloads(audio: Any, visual: Any, minimum_mouth_frames: int, np: Any) -> dict[str, Any]:
    valid, reason = validate_source_payloads(audio, visual)
    if not valid:
        raise SampleAlignmentError(reason or "invalid source payload")
    audio_embeddings = np.asarray(audio["embeddings"], dtype=np.float32)
    mouth_embeddings = np.asarray(visual["mouth_embeddings"], dtype=np.float32)
    mouth_indices = np.asarray(visual["mouth_frame_indices"], dtype=np.int64)
    mouth_timestamps = np.asarray(visual["timestamps_sec"], dtype=np.float64)[mouth_indices]
    audio_starts = np.asarray(audio["timestamps_start_sec"], dtype=np.float64)
    audio_ends = np.asarray(audio["timestamps_end_sec"], dtype=np.float64)
    source_audio_windows, source_mouth_frames = len(audio_embeddings), len(mouth_embeddings)
    if str(audio["audio_modality"].item()) == "NOT_APPLICABLE":
        return empty_alignment("AUDIO_NOT_APPLICABLE", source_audio_windows, source_mouth_frames, np)
    if source_mouth_frames == 0:
        return empty_alignment("MOUTH_NOT_APPLICABLE", source_audio_windows, source_mouth_frames, np)
    selected_audio, means, motions, starts, ends, counts = [], [], [], [], [], []
    for index, (start, end) in enumerate(zip(audio_starts, audio_ends, strict=True)):
        positions = np.flatnonzero((mouth_timestamps >= start) & (mouth_timestamps < end))
        if len(positions) < minimum_mouth_frames:
            continue
        sequence = mouth_embeddings[positions]
        selected_audio.append(audio_embeddings[index])
        means.append(sequence.mean(axis=0))
        motions.append(np.abs(np.diff(sequence, axis=0)).mean(axis=0))
        starts.append(start)
        ends.append(end)
        counts.append(len(sequence))
    if not selected_audio:
        return empty_alignment("NO_WINDOWS_WITH_SUFFICIENT_MOUTH_FRAMES", source_audio_windows, source_mouth_frames, np)
    return {
        "av_modality": "AVAILABLE", "not_applicable_reason": "",
        "timestamps_start_sec": np.asarray(starts, dtype=np.float32),
        "timestamps_end_sec": np.asarray(ends, dtype=np.float32),
        "audio_embeddings": np.stack(selected_audio).astype(np.float16),
        "mouth_mean_embeddings": np.stack(means).astype(np.float16),
        "mouth_motion_embeddings": np.stack(motions).astype(np.float16),
        "mouth_frame_counts": np.asarray(counts, dtype=np.int16),
        "source_audio_windows": source_audio_windows, "source_mouth_frames": source_mouth_frames,
    }


def empty_alignment(reason: str, audio_windows: int, mouth_frames: int, np: Any) -> dict[str, Any]:
    return {
        "av_modality": "NOT_APPLICABLE", "not_applicable_reason": reason,
        "timestamps_start_sec": np.empty((0,), dtype=np.float32),
        "timestamps_end_sec": np.empty((0,), dtype=np.float32),
        "audio_embeddings": np.empty((0, AUDIO_DIM), dtype=np.float16),
        "mouth_mean_embeddings": np.empty((0, MOUTH_DIM), dtype=np.float16),
        "mouth_motion_embeddings": np.empty((0, MOUTH_DIM), dtype=np.float16),
        "mouth_frame_counts": np.empty((0,), dtype=np.int16),
        "source_audio_windows": audio_windows, "source_mouth_frames": mouth_frames,
    }


def validate_output(
    path: Path, np: Any, expected_sample_id: str | None = None,
    expected_minimum_mouth_frames: int | None = None,
) -> tuple[bool, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = REQUIRED_KEYS - set(payload.files)
            if missing:
                return False, f"missing_keys:{','.join(sorted(missing))}"
            if str(payload["schema_version"].item()) != SCHEMA_VERSION:
                return False, "unsupported_schema"
            if expected_sample_id is not None and str(payload["sample_id"].item()) != expected_sample_id:
                return False, "sample_id_mismatch"
            if expected_minimum_mouth_frames is not None and int(payload["minimum_mouth_frames"].item()) != expected_minimum_mouth_frames:
                return False, "minimum_mouth_frames_mismatch"
            if str(payload["audio_model_sha256"].item()) != AUDIO_MODEL_SHA256:
                return False, "audio_model_digest_mismatch"
            if str(payload["visual_backbone"].item()) != VISUAL_BACKBONE:
                return False, "visual_backbone_mismatch"
            count = len(payload["audio_embeddings"])
            expected_shapes = {
                "audio_embeddings": (count, AUDIO_DIM), "mouth_mean_embeddings": (count, MOUTH_DIM),
                "mouth_motion_embeddings": (count, MOUTH_DIM),
            }
            for key, shape in expected_shapes.items():
                if payload[key].shape != shape:
                    return False, f"invalid_{key}_shape:{payload[key].shape}"
                if not np.isfinite(payload[key]).all():
                    return False, f"non_finite_{key}"
            for key in ("timestamps_start_sec", "timestamps_end_sec", "mouth_frame_counts"):
                if len(payload[key]) != count:
                    return False, f"{key}_length_mismatch"
            modality = str(payload["av_modality"].item())
            if modality == "AVAILABLE" and count == 0:
                return False, "available_without_pairs"
            if modality == "NOT_APPLICABLE" and count != 0:
                return False, "not_applicable_with_pairs"
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
            raise SampleAlignmentError(f"Refusing invalid output: {reason}")
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


def process_row(row: dict[str, str], args: argparse.Namespace, np: Any) -> dict[str, Any]:
    sample_id = row["sample_id"]
    output_path = args.output_root / f"{sample_id}.npz"
    existing_state = "NEW_OUTPUT"
    if output_path.exists() and not args.overwrite:
        valid, reason = validate_output(
            output_path, np, expected_sample_id=sample_id,
            expected_minimum_mouth_frames=args.minimum_mouth_frames,
        )
        if valid:
            return {"sample_id": sample_id, "status": "SKIPPED_VALID_OUTPUT", "output_path": str(output_path)}
        existing_state = "RECOMPUTED_INVALID_OUTPUT"
    elif output_path.exists():
        existing_state = "OVERWROTE_EXISTING_OUTPUT"
    audio_path, visual_path = args.audio_root / f"{sample_id}.npz", args.visual_root / f"{sample_id}.npz"
    if not audio_path.exists():
        raise SampleAlignmentError(f"Missing expected audio feature: {audio_path}")
    if not visual_path.exists():
        raise SampleAlignmentError(f"Missing expected visual feature: {visual_path}")
    started = perf_counter()
    with np.load(audio_path, allow_pickle=False) as audio, np.load(visual_path, allow_pickle=False) as visual:
        if str(audio["sample_id"].item()) != sample_id or str(visual["sample_id"].item()) != sample_id:
            raise SampleAlignmentError("Source feature sample ID mismatch")
        alignment = align_payloads(audio, visual, args.minimum_mouth_frames, np)
    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION), "sample_id": np.asarray(sample_id),
        "av_modality": np.asarray(alignment["av_modality"]),
        "not_applicable_reason": np.asarray(alignment["not_applicable_reason"]),
        "minimum_mouth_frames": np.asarray(args.minimum_mouth_frames, dtype=np.int16),
        "audio_model_sha256": np.asarray(AUDIO_MODEL_SHA256),
        "visual_backbone": np.asarray(VISUAL_BACKBONE),
        "timestamps_start_sec": alignment["timestamps_start_sec"], "timestamps_end_sec": alignment["timestamps_end_sec"],
        "audio_embeddings": alignment["audio_embeddings"], "mouth_mean_embeddings": alignment["mouth_mean_embeddings"],
        "mouth_motion_embeddings": alignment["mouth_motion_embeddings"], "mouth_frame_counts": alignment["mouth_frame_counts"],
        "source_audio_windows": np.asarray(alignment["source_audio_windows"], dtype=np.int32),
        "source_mouth_frames": np.asarray(alignment["source_mouth_frames"], dtype=np.int32),
    }
    output_bytes = atomic_write_npz(output_path, payload, np)
    return {
        "sample_id": sample_id, "status": "COMPLETED", "existing_output_state": existing_state,
        "split": row["split"], "semantic_class": row["semantic_class"],
        "av_modality": alignment["av_modality"], "not_applicable_reason": alignment["not_applicable_reason"],
        "source_audio_windows": alignment["source_audio_windows"], "source_mouth_frames": alignment["source_mouth_frames"],
        "aligned_windows": len(alignment["audio_embeddings"]),
        "alignment_coverage": len(alignment["audio_embeddings"]) / alignment["source_audio_windows"] if alignment["source_audio_windows"] else None,
        "output_bytes": output_bytes, "elapsed_seconds": perf_counter() - started, "output_path": str(output_path),
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
    if args.minimum_mouth_frames < 2:
        raise ValueError("--minimum-mouth-frames must be at least 2 to measure motion")
    np = require_numpy()
    rows = select_rows(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log_root / f"av_sync_{run_id}.jsonl"
    records, started = [], perf_counter()
    for position, row in enumerate(rows, 1):
        print(f"[{position}/{len(rows)}] {row['sample_id']}")
        try:
            record = process_row(row, args, np)
        except Exception as exc:
            record = {"sample_id": row["sample_id"], "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=5)}
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
    total_pairs = sum(record.get("aligned_windows", 0) for record in completed)
    total_source_windows = sum(record.get("source_audio_windows", 0) for record in completed)
    total_bytes = sum(record.get("output_bytes", 0) for record in completed)
    report = {
        "run_id": run_id, "schema_version": SCHEMA_VERSION, "input_mode": args.input_mode,
        "samples_selected": len(rows), "completed": len(completed), "completed_new": len(completed) - len(recomputed),
        "completed_recomputed_invalid_output": len(recomputed), "skipped_valid_output": len(skipped),
        "failed": len(failed), "accounted_samples": len(completed) + len(skipped) + len(failed),
        "failure_rate": len(failed) / len(rows), "available_av": len(available), "not_applicable_av": len(not_applicable),
        "aligned_windows": total_pairs, "source_audio_windows": total_source_windows,
        "overall_alignment_coverage": total_pairs / total_source_windows if total_source_windows else None,
        "elapsed_seconds": elapsed, "videos_per_minute": len(completed) / elapsed * 60 if elapsed else None,
        "mean_output_bytes_per_video": total_bytes / len(completed) if completed else None,
        "per_sample_log": str(log_path),
    }
    summary_path = args.log_root / f"av_sync_{run_id}_summary.json"
    atomic_json(summary_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
