"""Extract frozen XLSR-SLS audio embeddings from FakeAVCeleb videos.

Audio is decoded in memory with FFmpeg as mono 16 kHz float32 PCM. The official
64,600-sample XLSR-SLS window is preserved. Only compact temporal embeddings,
timestamps, and pretrained classifier outputs are persisted.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "fakeavceleb_manifest_split.csv"
DEFAULT_SMOKE_FRAME_ROOT = PROJECT_ROOT / "data" / "interim" / "frames"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "xlsr_sls" / "xlsr-sls.onnx"
DEFAULT_AUGMENTED_MODEL = PROJECT_ROOT / "models" / "xlsr_sls" / "xlsr-sls-embeddings.onnx"
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "audio_embeddings" / "xlsr_sls_v1"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "interim" / "feature_extraction_logs"
GATE_SCRIPT = PROJECT_ROOT / "scripts" / "setup" / "12_verify_xlsr_sls.py"

SCHEMA_VERSION = "audio_embedding_v1"
BACKBONE_NAME = "SpeechAntiSpoofingBenchmarks/XLSR-SLS"
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 64_600
EMBEDDING_DIM = 1_024
AUDIO_FAKE_CLASSES = {"AUDIO_MANIPULATION", "AUDIO_VISUAL_MANIPULATION"}
REQUIRED_NPZ_KEYS = {
    "schema_version", "backbone", "model_sha256", "sample_id", "audio_modality",
    "sample_rate", "window_samples", "hop_samples", "source_audio_samples",
    "window_start_samples", "window_valid_samples", "timestamps_start_sec",
    "timestamps_end_sec", "embeddings", "pretrained_log_probabilities",
}


class SampleExtractionError(RuntimeError):
    """A recoverable failure affecting one video."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    video_path: Path
    semantic_class: str
    split: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("video", "smoke"), default="video")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke-frame-root", type=Path, default=DEFAULT_SMOKE_FRAME_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--augmented-model", type=Path, default=DEFAULT_AUGMENTED_MODEL)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hop-samples", type=int, default=WINDOW_SAMPLES)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg-timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def load_gate_module() -> Any:
    spec = importlib.util.spec_from_file_location("xlsr_sls_gate", GATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load compatibility helpers from {GATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_runtime() -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the CUDA PyTorch stack and requirements-audio.txt.") from exc
    return {"np": np, "onnx": onnx, "ort": ort, "torch": torch}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def augmented_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def validate_augmented_model(path: Path, source_sha256: str) -> tuple[bool, str | None]:
    metadata_path = augmented_metadata_path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") != source_sha256:
            return False, "source checkpoint digest changed"
        if metadata.get("embedding_dim") != EMBEDDING_DIM:
            return False, "embedding dimension changed"
        if path.stat().st_size < 1_000_000_000:
            return False, "derived model is unexpectedly small"
        if sha256_file(path) != metadata.get("derived_sha256"):
            return False, "derived model digest mismatch"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, None


def prepare_augmented_model(runtime: dict[str, Any], gate: Any, source: Path, destination: Path) -> tuple[Path, str, str]:
    if not source.exists():
        raise FileNotFoundError(f"XLSR-SLS checkpoint missing: {source}")
    source_sha256 = sha256_file(source)
    if source_sha256 != gate.MODEL_SHA256:
        raise RuntimeError(f"XLSR-SLS checkpoint digest mismatch: {source_sha256}")
    valid, reason = validate_augmented_model(destination, source_sha256) if destination.exists() else (False, "missing")
    if valid:
        metadata = json.loads(augmented_metadata_path(destination).read_text(encoding="utf-8"))
        return destination, str(metadata["embedding_tensor"]), source_sha256
    logging.info("Preparing crash-safe embedding-output ONNX model (%s).", reason)
    onnx = runtime["onnx"]
    model = onnx.load_model(str(source), load_external_data=False)
    onnx.checker.check_model(model)
    embedding_tensor, _ = gate.locate_sls_embedding_tensor(model)
    output_names = {item.name for item in model.graph.output}
    if embedding_tensor not in output_names:
        model.graph.output.append(onnx.helper.make_tensor_value_info(
            embedding_tensor, onnx.TensorProto.FLOAT, [None, EMBEDDING_DIM]
        ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        onnx.save_model(model, str(temporary))
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        derived_sha256 = sha256_file(temporary)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
        atomic_write_json(augmented_metadata_path(destination), {
            "source_sha256": source_sha256,
            "derived_sha256": derived_sha256,
            "embedding_tensor": embedding_tensor,
            "embedding_dim": EMBEDDING_DIM,
        })
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        del model
        gc.collect()
    valid, reason = validate_augmented_model(destination, source_sha256)
    if not valid:
        raise RuntimeError(f"Refusing invalid augmented model: {reason}")
    return destination, embedding_tensor, source_sha256


def build_session(runtime: dict[str, Any], gate: Any, model_path: Path, embedding_tensor: str, args: argparse.Namespace) -> tuple[Any, str, str, str]:
    ort, torch = runtime["ort"], runtime["torch"]
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch cannot access the GPU")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA requested but ONNX Runtime has no CUDAExecutionProvider")
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    resolved, providers = gate.resolve_providers(ort, requested, args.gpu_memory_limit_gb)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if resolved == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA silently fell back to {session.get_providers()}")
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one audio input, found {len(inputs)}")
    outputs = session.get_outputs()
    output_names = [item.name for item in outputs]
    if embedding_tensor not in output_names:
        raise RuntimeError(f"Embedding output {embedding_tensor!r} missing from derived model")
    classifier_outputs = [name for name in output_names if name != embedding_tensor]
    if len(classifier_outputs) != 1:
        raise RuntimeError(f"Expected one classifier output, found {classifier_outputs}")
    return session, resolved, inputs[0].name, classifier_outputs[0]


def parse_sample_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "video_path", "split", "semantic_class"}
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"Manifest must contain {sorted(required)}")
    return rows


def select_samples(args: argparse.Namespace) -> list[Sample]:
    rows = load_manifest(args.manifest)
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
        rng = random.Random(args.seed)
        chosen = []
        for semantic_class in sorted(grouped):
            group = sorted(grouped[semantic_class], key=lambda item: item["sample_id"])
            if len(group) < args.per_class:
                raise RuntimeError(f"Class {semantic_class} has fewer than {args.per_class} rows")
            chosen.extend(rng.sample(group, args.per_class))
        rows = chosen
    rows.sort(key=lambda item: item["sample_id"])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        rows = rows[:args.limit]
    samples = [Sample(
        sample_id=row["sample_id"], video_path=PROJECT_ROOT / row["video_path"],
        semantic_class=row["semantic_class"], split=row["split"],
    ) for row in rows]
    if not samples:
        raise RuntimeError("No samples selected. Nothing was silently skipped.")
    return samples


def has_audio_stream(sample: Sample, args: argparse.Namespace) -> bool:
    command = [
        args.ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(sample.video_path),
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


def decode_audio(sample: Sample, args: argparse.Namespace, np: Any) -> Any:
    if not has_audio_stream(sample, args):
        return np.empty((0,), dtype=np.float32)
    command = [
        args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(sample.video_path),
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


def make_windows(waveform: Any, hop_samples: int, np: Any) -> tuple[Any, Any, Any]:
    if waveform.size == 0:
        return (
            np.empty((0, WINDOW_SAMPLES), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )
    starts = np.arange(0, waveform.size, hop_samples, dtype=np.int64)
    windows, valid = [], []
    for start in starts.tolist():
        segment = waveform[start:start + WINDOW_SAMPLES]
        valid.append(len(segment))
        if len(segment) < WINDOW_SAMPLES:
            segment = np.resize(segment, WINDOW_SAMPLES)
        windows.append(segment.astype(np.float32, copy=False))
    return np.stack(windows), starts, np.asarray(valid, dtype=np.int32)


def atomic_write_npz(path: Path, payload: dict[str, Any], np: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        valid, reason = validate_feature_file(temporary, np)
        if not valid:
            raise SampleExtractionError(f"Refusing invalid feature artifact: {reason}")
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path.stat().st_size


def validate_feature_file(
    path: Path,
    np: Any,
    expected_model_sha256: str | None = None,
    expected_hop_samples: int | None = None,
    expected_sample_id: str | None = None,
) -> tuple[bool, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = REQUIRED_NPZ_KEYS - set(payload.files)
            if missing:
                return False, f"missing keys: {sorted(missing)}"
            if str(payload["schema_version"].item()) != SCHEMA_VERSION:
                return False, "unsupported schema version"
            if expected_model_sha256 is not None and str(payload["model_sha256"].item()) != expected_model_sha256:
                return False, "model checkpoint digest mismatch"
            if expected_hop_samples is not None and int(payload["hop_samples"].item()) != expected_hop_samples:
                return False, "hop size mismatch"
            if expected_sample_id is not None and str(payload["sample_id"].item()) != expected_sample_id:
                return False, "sample ID mismatch"
            embeddings = payload["embeddings"]
            scores = payload["pretrained_log_probabilities"]
            if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
                return False, f"invalid embedding shape: {embeddings.shape}"
            if scores.shape != (len(embeddings), 2):
                return False, f"invalid score shape: {scores.shape}"
            for key in ("window_start_samples", "window_valid_samples", "timestamps_start_sec", "timestamps_end_sec"):
                if len(payload[key]) != len(embeddings):
                    return False, f"{key} length mismatch"
            if not np.isfinite(embeddings).all() or not np.isfinite(scores).all():
                return False, "non-finite model output"
            modality = str(payload["audio_modality"].item())
            if modality not in {"AVAILABLE", "NOT_APPLICABLE"}:
                return False, f"invalid modality: {modality}"
            if modality == "NOT_APPLICABLE" and len(embeddings):
                return False, "NOT_APPLICABLE artifact contains evidence"
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, f"unreadable npz: {exc}"
    return True, None


def process_sample(sample: Sample, args: argparse.Namespace, runtime: dict[str, Any], session: Any, input_name: str, classifier_output: str, embedding_output: str, model_sha256: str) -> dict[str, Any]:
    np = runtime["np"]
    output_path = args.feature_root / f"{sample.sample_id}.npz"
    existing_state = "NEW_OUTPUT"
    if output_path.exists() and not args.overwrite:
        valid, reason = validate_feature_file(
            output_path, np,
            expected_model_sha256=model_sha256,
            expected_hop_samples=args.hop_samples,
            expected_sample_id=sample.sample_id,
        )
        if valid:
            return {"sample_id": sample.sample_id, "status": "SKIPPED_VALID_OUTPUT", "output_path": str(output_path)}
        existing_state = "RECOMPUTED_INVALID_OUTPUT"
        logging.warning("Invalid cached output for %s (%s); recomputing.", sample.sample_id, reason)
    elif output_path.exists():
        existing_state = "OVERWROTE_EXISTING_OUTPUT"
    started = perf_counter()
    waveform = decode_audio(sample, args, np)
    windows, starts, valid_samples = make_windows(waveform, args.hop_samples, np)
    embeddings, scores = [], []
    inference_seconds = 0.0
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset:offset + args.batch_size]
        inference_started = perf_counter()
        result = session.run([classifier_output, embedding_output], {input_name: batch})
        inference_seconds += perf_counter() - inference_started
        batch_scores, batch_embeddings = map(np.asarray, result)
        if batch_embeddings.shape != (len(batch), EMBEDDING_DIM):
            raise SampleExtractionError(f"Unexpected embedding shape: {batch_embeddings.shape}")
        if batch_scores.shape != (len(batch), 2):
            raise SampleExtractionError(f"Unexpected classifier shape: {batch_scores.shape}")
        embeddings.append(batch_embeddings.astype(np.float16))
        scores.append(batch_scores.astype(np.float32))
    embedding_array = np.concatenate(embeddings) if embeddings else np.empty((0, EMBEDDING_DIM), dtype=np.float16)
    score_array = np.concatenate(scores) if scores else np.empty((0, 2), dtype=np.float32)
    modality = "AVAILABLE" if len(embedding_array) else "NOT_APPLICABLE"
    timestamps_start = starts.astype(np.float64) / SAMPLE_RATE
    timestamps_end = np.minimum(starts + WINDOW_SAMPLES, waveform.size).astype(np.float64) / SAMPLE_RATE
    output_bytes = atomic_write_npz(output_path, {
        "schema_version": np.asarray(SCHEMA_VERSION), "backbone": np.asarray(BACKBONE_NAME),
        "model_sha256": np.asarray(model_sha256), "sample_id": np.asarray(sample.sample_id),
        "audio_modality": np.asarray(modality), "sample_rate": np.asarray(SAMPLE_RATE, dtype=np.int32),
        "window_samples": np.asarray(WINDOW_SAMPLES, dtype=np.int32),
        "hop_samples": np.asarray(args.hop_samples, dtype=np.int32),
        "source_audio_samples": np.asarray(waveform.size, dtype=np.int64),
        "window_start_samples": starts, "window_valid_samples": valid_samples,
        "timestamps_start_sec": timestamps_start.astype(np.float32),
        "timestamps_end_sec": timestamps_end.astype(np.float32),
        "embeddings": embedding_array, "pretrained_log_probabilities": score_array,
    }, np)
    elapsed = perf_counter() - started
    return {
        "sample_id": sample.sample_id, "status": "COMPLETED", "existing_output_state": existing_state,
        "split": sample.split, "semantic_class": sample.semantic_class,
        "audio_binary_label": int(sample.semantic_class in AUDIO_FAKE_CLASSES),
        "audio_modality": modality, "source_audio_samples": int(waveform.size),
        "audio_seconds": waveform.size / SAMPLE_RATE, "windows": len(embedding_array),
        "embedding_shape": list(embedding_array.shape), "classifier_shape": list(score_array.shape),
        "inference_seconds": inference_seconds, "elapsed_seconds": elapsed,
        "output_bytes": output_bytes, "output_path": str(output_path),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.hop_samples < 1 or args.gpu_memory_limit_gb <= 0 or args.ffmpeg_timeout <= 0:
        raise ValueError("Hop size, memory limit, and FFmpeg timeout must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = require_runtime()
    gate = load_gate_module()
    samples = select_samples(args)
    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = args.log_root / f"xlsr_sls_{run_id}.jsonl"
    logging.info("Selected %d samples, mode=%s, requested_device=%s", len(samples), args.input_mode, args.device)
    monitor = gate.GPUMemoryMonitor(args.device != "cpu")
    monitor.start()
    initialization_started = perf_counter()
    try:
        augmented_model, embedding_output, model_sha256 = prepare_augmented_model(runtime, gate, args.model, args.augmented_model)
        session, device, input_name, classifier_output = build_session(runtime, gate, augmented_model, embedding_output, args)
        initialization_seconds = perf_counter() - initialization_started
        records = []
        processing_started = perf_counter()
        for position, sample in enumerate(samples, 1):
            logging.info("[%d/%d] %s", position, len(samples), sample.sample_id)
            try:
                record = process_sample(sample, args, runtime, session, input_name, classifier_output, embedding_output, model_sha256)
            except Exception as exc:
                record = {
                    "sample_id": sample.sample_id, "status": "FAILED", "error_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc(limit=5),
                }
                logging.exception("Audio feature extraction failed for %s", sample.sample_id)
            record["recorded_at"] = datetime.now(timezone.utc).isoformat()
            append_jsonl(jsonl_path, record)
            records.append(record)
            if record["status"] == "FAILED" and args.fail_fast:
                raise RuntimeError(f"Stopping after failure of {sample.sample_id}")
        processing_seconds = perf_counter() - processing_started
    finally:
        monitor.stop()
    completed = [item for item in records if item["status"] == "COMPLETED"]
    skipped = [item for item in records if item["status"] == "SKIPPED_VALID_OUTPUT"]
    failed = [item for item in records if item["status"] == "FAILED"]
    recomputed = [item for item in completed if item["existing_output_state"] == "RECOMPUTED_INVALID_OUTPUT"]
    available = [item for item in completed if item["audio_modality"] == "AVAILABLE"]
    not_applicable = [item for item in completed if item["audio_modality"] == "NOT_APPLICABLE"]
    total_windows = sum(item.get("windows", 0) for item in completed)
    total_audio_seconds = sum(item.get("audio_seconds", 0.0) for item in completed)
    total_bytes = sum(item.get("output_bytes", 0) for item in completed)
    report = {
        "run_id": run_id, "schema_version": SCHEMA_VERSION, "backbone": BACKBONE_NAME,
        "model_sha256": model_sha256, "input_mode": args.input_mode, "device": device,
        "samples_selected": len(samples), "completed": len(completed),
        "completed_new": len(completed) - len(recomputed),
        "completed_recomputed_invalid_output": len(recomputed), "skipped_valid_output": len(skipped),
        "failed": len(failed), "accounted_samples": len(completed) + len(skipped) + len(failed),
        "failure_rate": len(failed) / len(samples), "available_audio": len(available),
        "not_applicable_audio": len(not_applicable), "windows": total_windows,
        "audio_seconds": total_audio_seconds, "processing_seconds": processing_seconds,
        "initialization_seconds": initialization_seconds,
        "windows_per_second": total_windows / processing_seconds if processing_seconds else None,
        "audio_realtime_factor": processing_seconds / total_audio_seconds if total_audio_seconds else None,
        "videos_per_minute": len(completed) / processing_seconds * 60 if processing_seconds else None,
        "mean_output_bytes_per_video": total_bytes / len(completed) if completed else None,
        "peak_process_gpu_vram_bytes": monitor.peak_bytes,
        "peak_host_rss_bytes": gate.peak_host_rss_bytes(),
        "augmented_model_bytes": augmented_model.stat().st_size,
        "active_providers": session.get_providers(),
        "per_sample_log": str(jsonl_path),
    }
    report_path = args.log_root / f"xlsr_sls_{run_id}_summary.json"
    atomic_write_json(report_path, report)
    logging.info("Run complete: %s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
