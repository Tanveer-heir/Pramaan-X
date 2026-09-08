"""Generate controlled one-pass and two-pass compression feature pairs.

Only REAL videos from the requested component-disjoint split are eligible.
Temporary re-encoded MP4 files are deleted after compact features are written.
Generation 1 is the negative class and generation 2 is the re-encoding class.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "fakeavceleb_manifest_split.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "compression_features" / "controlled_v1"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "interim" / "feature_extraction_logs"
EXTRACTOR_PATH = Path(__file__).with_name("09_extract_compression_features.py")
PAIR_SCHEMA_VERSION = "compression_controlled_pair_v1"


def load_extractor() -> Any:
    spec = importlib.util.spec_from_file_location("compression_feature_extractor", EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load compression extractor: {EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sample-count", type=int, default=None, help="Deterministic random count of REAL source videos.")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.0025)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg-timeout", type=float, default=180.0)
    parser.add_argument("--ffprobe-timeout", type=float, default=60.0)
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def parse_sample_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def select_real_sources(args: argparse.Namespace, runtime: dict[str, Any]) -> list[tuple[str, Path]]:
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    dataframe = runtime["pd"].read_csv(args.manifest)
    required = {"sample_id", "video_path", "split", "semantic_class"}
    missing = required - set(dataframe.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    dataframe = dataframe[(dataframe["split"] == args.split) & (dataframe["semantic_class"] == "REAL")].copy()
    selected_ids = parse_sample_ids(args.sample_id)
    if selected_ids:
        dataframe = dataframe[dataframe["sample_id"].isin(selected_ids)]
    if args.sample_count is not None:
        if args.sample_count < 1:
            raise ValueError("--sample-count must be at least 1")
        if len(dataframe) < args.sample_count:
            raise RuntimeError(f"Requested {args.sample_count} REAL videos but only {len(dataframe)} are available.")
        dataframe = dataframe.sample(n=args.sample_count, random_state=args.seed)
    dataframe = dataframe.sort_values("sample_id")
    sources = [(str(row.sample_id), PROJECT_ROOT / row.video_path) for row in dataframe.itertuples(index=False)]
    if not sources:
        raise RuntimeError("No REAL source videos selected.")
    return sources


def run_encode(source: Path, destination: Path, args: argparse.Namespace) -> None:
    command = [
        args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-map_metadata", "-1",
        "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", "-threads", str(args.ffmpeg_threads),
        "-movflags", "+faststart", str(destination),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.ffmpeg_timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFmpeg executable not found: {args.ffmpeg}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg timed out after {args.ffmpeg_timeout:.1f}s for {source}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}: {result.stderr.strip()[-1000:]}")
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create a valid output: {destination}")


def pair_path(root: Path, sample_id: str, generation: int) -> Path:
    return root / f"{sample_id}_generation{generation}.npz"


def validate_pair_file(path: Path, generation: int, extractor: Any, np: Any) -> tuple[bool, str | None]:
    valid, reason = extractor.validate_feature_file(path, np)
    if not valid:
        return valid, reason
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"pair_schema_version", "source_sample_id", "controlled_generation", "compression_label"}
            missing = required - set(payload.files)
            if missing:
                return False, f"missing pair keys: {sorted(missing)}"
            if str(payload["pair_schema_version"].item()) != PAIR_SCHEMA_VERSION:
                return False, "unsupported pair schema"
            if int(payload["controlled_generation"].item()) != generation:
                return False, "generation mismatch"
            if int(payload["compression_label"].item()) != generation - 1:
                return False, "compression label mismatch"
    except (EOFError, OSError, ValueError, KeyError) as exc:
        return False, f"unreadable pair artifact: {exc}"
    return True, None


def extract_generation(video_path: Path, output_path: Path, sample_id: str, generation: int, args: argparse.Namespace, extractor: Any, runtime: dict[str, Any]) -> int:
    np = runtime["np"]
    probe = extractor.probe_statistics(
        extractor.run_ffprobe(video_path, args.ffprobe, args.ffprobe_timeout),
        video_path.stat().st_size,
        np,
    )
    frame = extractor.frame_statistics(video_path, args.fps, args.duplicate_threshold, runtime)
    features = extractor.build_feature_vector(probe, frame, np)
    return extractor.write_feature_file(output_path, {
        "schema_version": np.asarray(extractor.SCHEMA_VERSION),
        "sample_id": np.asarray(f"{sample_id}_generation{generation}"),
        "availability": np.asarray("AVAILABLE"),
        "features": features,
        "feature_names": np.asarray(extractor.FEATURE_NAMES),
        "codec_name": np.asarray(probe["codec_name"]),
        "pixel_format": np.asarray(probe["pixel_format"]),
        "duration_seconds": np.asarray(probe["duration"], dtype=np.float32),
        "source_width": np.asarray(probe["width"], dtype=np.int32),
        "source_height": np.asarray(probe["height"], dtype=np.int32),
        "source_fps": np.asarray(probe["fps"], dtype=np.float32),
        "sampled_frame_count": np.asarray(frame["sampled_frame_count"], dtype=np.int32),
        "probed_frame_count": np.asarray(probe["probed_frame_count"], dtype=np.int32),
        "pair_schema_version": np.asarray(PAIR_SCHEMA_VERSION),
        "source_sample_id": np.asarray(sample_id),
        "controlled_generation": np.asarray(generation, dtype=np.int8),
        "compression_label": np.asarray(generation - 1, dtype=np.int8),
        "encoding_crf": np.asarray(args.crf, dtype=np.int16),
        "encoding_preset": np.asarray(args.preset),
    }, np)


def process_source(sample_id: str, source: Path, args: argparse.Namespace, extractor: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    if not source.exists():
        raise RuntimeError(f"Source video not found: {source}")
    outputs = {generation: pair_path(args.output_root, sample_id, generation) for generation in (1, 2)}
    states: dict[int, str] = {}
    for generation, path in outputs.items():
        if path.exists() and not args.overwrite:
            valid, reason = validate_pair_file(path, generation, extractor, runtime["np"])
            states[generation] = "SKIPPED_VALID_OUTPUT" if valid else "RECOMPUTED_INVALID_OUTPUT"
            if not valid:
                logging.warning("Invalid controlled artifact %s (%s); recomputing.", path.name, reason)
        else:
            states[generation] = "OVERWROTE_EXISTING_OUTPUT" if path.exists() else "NEW_OUTPUT"
    if all(state == "SKIPPED_VALID_OUTPUT" for state in states.values()):
        return {"sample_id": sample_id, "status": "SKIPPED_VALID_OUTPUT", "artifacts": 2}

    started = perf_counter()
    bytes_written = 0
    with tempfile.TemporaryDirectory(prefix=f"pramaan_compression_{sample_id}_") as directory:
        temporary_root = Path(directory)
        generation1_video = temporary_root / "generation1.mp4"
        generation2_video = temporary_root / "generation2.mp4"
        run_encode(source, generation1_video, args)
        if states[1] != "SKIPPED_VALID_OUTPUT":
            bytes_written += extract_generation(generation1_video, outputs[1], sample_id, 1, args, extractor, runtime)
        run_encode(generation1_video, generation2_video, args)
        if states[2] != "SKIPPED_VALID_OUTPUT":
            bytes_written += extract_generation(generation2_video, outputs[2], sample_id, 2, args, extractor, runtime)
    for generation, path in outputs.items():
        valid, reason = validate_pair_file(path, generation, extractor, runtime["np"])
        if not valid:
            raise RuntimeError(f"Final controlled artifact failed validation: {path}: {reason}")
    elapsed = perf_counter() - started
    return {
        "sample_id": sample_id,
        "status": "COMPLETED",
        "generation1_state": states[1],
        "generation2_state": states[2],
        "artifacts": 2,
        "bytes_written": bytes_written,
        "elapsed_seconds": elapsed,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.ffmpeg_timeout <= 0 or args.ffprobe_timeout <= 0 or args.ffmpeg_threads < 1:
        raise ValueError("fps, timeouts, and ffmpeg-threads must be positive")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extractor = load_extractor()
    runtime = extractor.require_runtime()
    sources = select_real_sources(args, runtime)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = args.log_root / f"compression_pairs_{run_id}.jsonl"
    records: list[dict[str, Any]] = []
    logging.info("Selected %d REAL sources from split=%s", len(sources), args.split)
    for position, (sample_id, source) in enumerate(sources, start=1):
        logging.info("[%d/%d] %s", position, len(sources), sample_id)
        try:
            record = process_source(sample_id, source, args, extractor, runtime)
        except Exception as exc:
            record = {"sample_id": sample_id, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=6)}
            logging.exception("Controlled compression generation failed for %s", sample_id)
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        append_jsonl(jsonl_path, record)
        records.append(record)
        if record["status"] == "FAILED" and args.fail_fast:
            raise RuntimeError(f"Stopping after failure of {sample_id}")

    completed = [record for record in records if record["status"] == "COMPLETED"]
    skipped = [record for record in records if record["status"] == "SKIPPED_VALID_OUTPUT"]
    failed = [record for record in records if record["status"] == "FAILED"]
    elapsed = sum(record.get("elapsed_seconds", 0.0) for record in completed)
    bytes_written = sum(record.get("bytes_written", 0) for record in completed)
    summary = {
        "run_id": run_id,
        "pair_schema_version": PAIR_SCHEMA_VERSION,
        "split": args.split,
        "sources_selected": len(sources),
        "completed": len(completed),
        "skipped_valid_output": len(skipped),
        "failed": len(failed),
        "accounted_sources": len(completed) + len(skipped) + len(failed),
        "failure_rate": len(failed) / len(sources),
        "feature_artifacts_accounted": 2 * (len(completed) + len(skipped)),
        "sources_per_minute": len(completed) / elapsed * 60.0 if elapsed else None,
        "mean_new_output_bytes_per_completed_source": bytes_written / len(completed) if completed else None,
        "temporary_video_policy": "EPHEMERAL_PER_SOURCE",
        "per_sample_log": str(jsonl_path),
    }
    summary_path = args.log_root / f"compression_pairs_{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Run complete: %s", json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
