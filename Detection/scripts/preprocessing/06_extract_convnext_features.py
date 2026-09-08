"""Stream frozen ConvNeXt-Tiny face and mouth embeddings from FakeAVCeleb.

The production path decodes MP4s directly and never persists sampled frames or
crops.  ``--input-mode frames`` exists only to validate the already-created
12-video JPEG smoke corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "fakeavceleb_manifest_split.csv"
DEFAULT_FRAME_ROOT = PROJECT_ROOT / "data" / "interim" / "frames"
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "data" / "interim" / "feature_extraction_logs"
DEFAULT_LANDMARKER = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"

SCHEMA_VERSION = "visual_embedding_v1"
BACKBONE_NAME = "torchvision/convnext_tiny/IMAGENET1K_V1"
EMBEDDING_DIM = 768
REQUIRED_NPZ_KEYS = {
    "schema_version",
    "backbone",
    "sample_id",
    "timestamps_sec",
    "source_frame_indices",
    "face_embeddings",
    "face_frame_indices",
    "face_bboxes_xyxy",
    "mouth_embeddings",
    "mouth_frame_indices",
    "mouth_bboxes_xyxy",
}

# Standard MediaPipe Face Mesh indices.  Keep these aligned with script 05.
MOUTH_REGION = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311,
    310, 415,
]


class SampleExtractionError(RuntimeError):
    """A recoverable error that applies to one sample only."""


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
    parser.add_argument("--sample-id", action="append", default=[], help="Repeat or provide comma-separated IDs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None, help="Deterministic representative sample count per class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def parse_sample_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def require_runtime() -> dict[str, Any]:
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as torch_f
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Activate the project virtual environment and "
            "install torch, torchvision, opencv-python, mediapipe, numpy, and pandas."
        ) from exc

    return {
        "cv2": cv2,
        "mp": mp,
        "np": np,
        "pd": pd,
        "torch": torch,
        "torch_f": torch_f,
        "ConvNeXt_Tiny_Weights": ConvNeXt_Tiny_Weights,
        "convnext_tiny": convnext_tiny,
    }


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    return requested


def build_encoder(runtime: dict[str, Any], device: str) -> Any:
    model = runtime["convnext_tiny"](
        weights=runtime["ConvNeXt_Tiny_Weights"].IMAGENET1K_V1
    )
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def bbox_from_landmarks(landmarks: Any, indices: list[int], width: int, height: int, margin: float) -> tuple[int, int, int, int] | None:
    points = []
    for index in indices:
        landmark = landmarks[index]
        points.append((
            min(width - 1, max(0, int(landmark.x * width))),
            min(height - 1, max(0, int(landmark.y * height))),
        ))
    x_values, y_values = zip(*points)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = int((x_max - x_min) * margin)
    y_pad = int((y_max - y_min) * margin)
    bbox = (max(0, x_min - x_pad), max(0, y_min - y_pad), min(width, x_max + x_pad), min(height, y_max + y_pad))
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def crop(image: Any, bbox: tuple[int, int, int, int] | None) -> Any | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    result = image[y1:y2, x1:x2]
    return result.copy() if result.size else None


def iter_cached_frames(frame_dir: Path, fps: float, cv2: Any) -> Iterator[tuple[int, float, Any]]:
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SampleExtractionError(f"No cached JPEG frames found in {frame_dir}")
    for output_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise SampleExtractionError(f"Unreadable cached frame: {frame_path}")
        yield output_index, output_index / fps, image


def iter_video_frames(video_path: Path, target_fps: float, cv2: Any) -> Iterator[tuple[int, float, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SampleExtractionError(f"Unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        capture.release()
        raise SampleExtractionError(f"Video reported invalid FPS: {video_path}")

    next_timestamp = 0.0
    source_index = 0
    yielded = 0
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


def flush_batch(pending: list[tuple[str, int, tuple[int, int, int, int], Any]], stores: dict[str, dict[str, list[Any]]], encoder: Any, runtime: dict[str, Any], device: str) -> None:
    if not pending:
        return
    np, torch, cv2 = runtime["np"], runtime["torch"], runtime["cv2"]
    # Face and mouth crops have different native sizes, so normalise each crop
    # before batching.  This remains in-memory and bounds GPU use to one model.
    images_bgr = np.stack([
        cv2.resize(item[3], (224, 224), interpolation=cv2.INTER_LINEAR)
        for item in pending
    ])
    tensor = torch.from_numpy(images_bgr).to(device=device, non_blocking=device.startswith("cuda"))
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    tensor = tensor[:, [2, 1, 0], :, :]  # BGR to RGB
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    autocast_enabled = device.startswith("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
        features = encoder.features(tensor)
        embeddings = torch.flatten(encoder.avgpool(features), 1)
    embeddings = embeddings.float().cpu().numpy().astype(np.float16, copy=False)
    for (kind, frame_index, bbox, _), embedding in zip(pending, embeddings, strict=True):
        stores[kind]["embeddings"].append(embedding)
        stores[kind]["frame_indices"].append(frame_index)
        stores[kind]["bboxes"].append(bbox)
    pending.clear()


def arrays_for_store(store: dict[str, list[Any]], np: Any) -> tuple[Any, Any, Any]:
    if not store["embeddings"]:
        return (
            np.empty((0, EMBEDDING_DIM), dtype=np.float16),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 4), dtype=np.int32),
        )
    return (
        np.stack(store["embeddings"]).astype(np.float16, copy=False),
        np.asarray(store["frame_indices"], dtype=np.int32),
        np.asarray(store["bboxes"], dtype=np.int32),
    )


def write_feature_file(path: Path, payload: dict[str, Any], np: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
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


def _fsync_directory(directory: Path) -> None:
    """Make a completed rename durable against an abrupt system restart."""
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
            keys = set(payload.files)
            missing = REQUIRED_NPZ_KEYS - keys
            if missing:
                return False, f"missing keys: {sorted(missing)}"
            if str(payload["schema_version"].item()) != SCHEMA_VERSION:
                return False, "unsupported schema version"
            for name in ("face_embeddings", "mouth_embeddings"):
                if payload[name].ndim != 2 or payload[name].shape[1] != EMBEDDING_DIM:
                    return False, f"invalid {name} shape: {payload[name].shape}"
                prefix = name.removesuffix("embeddings")
                if len(payload[name]) != len(payload[f"{prefix}frame_indices"]):
                    return False, f"{name}/frame-index length mismatch"
                if len(payload[name]) != len(payload[f"{prefix}bboxes_xyxy"]):
                    return False, f"{name}/bbox length mismatch"
            if len(payload["timestamps_sec"]) != len(payload["source_frame_indices"]):
                return False, "timestamp/source-frame length mismatch"
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, f"unreadable npz: {exc}"
    return True, None


def process_sample(sample: Sample, args: argparse.Namespace, landmarker: Any, encoder: Any, runtime: dict[str, Any], device: str) -> dict[str, Any]:
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

    frame_iter = (
        iter_cached_frames(sample.frame_dir, args.fps, cv2)
        if args.input_mode == "frames"
        else iter_video_frames(sample.video_path, args.fps, cv2)
    )
    stores = {kind: {"embeddings": [], "frame_indices": [], "bboxes": []} for kind in ("face", "mouth")}
    pending: list[tuple[str, int, tuple[int, int, int, int], Any]] = []
    timestamps: list[float] = []
    source_indices: list[int] = []
    decoded_frames = face_detected_frames = 0
    started = perf_counter()

    for temporal_index, (source_index, timestamp, image_bgr) in enumerate(frame_iter):
        decoded_frames += 1
        timestamps.append(timestamp)
        source_indices.append(source_index)
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
        if not result.face_landmarks:
            continue
        face_detected_frames += 1
        landmarks = result.face_landmarks[0]
        face_bbox = bbox_from_landmarks(landmarks, list(range(len(landmarks))), width, height, margin=0.12)
        mouth_bbox = bbox_from_landmarks(landmarks, MOUTH_REGION, width, height, margin=0.30)
        for kind, bbox in (("face", face_bbox), ("mouth", mouth_bbox)):
            image_crop = crop(image_bgr, bbox)
            if image_crop is not None:
                pending.append((kind, temporal_index, bbox, image_crop))
        if len(pending) >= args.batch_size:
            flush_batch(pending, stores, encoder, runtime, device)
    flush_batch(pending, stores, encoder, runtime, device)

    if decoded_frames == 0:
        raise SampleExtractionError("No frames were decoded.")
    face_embeddings, face_indices, face_bboxes = arrays_for_store(stores["face"], np)
    mouth_embeddings, mouth_indices, mouth_bboxes = arrays_for_store(stores["mouth"], np)
    bytes_written = write_feature_file(output_path, {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "backbone": np.asarray(BACKBONE_NAME),
        "sample_id": np.asarray(sample.sample_id),
        "timestamps_sec": np.asarray(timestamps, dtype=np.float32),
        "source_frame_indices": np.asarray(source_indices, dtype=np.int32),
        "face_embeddings": face_embeddings,
        "face_frame_indices": face_indices,
        "face_bboxes_xyxy": face_bboxes,
        "mouth_embeddings": mouth_embeddings,
        "mouth_frame_indices": mouth_indices,
        "mouth_bboxes_xyxy": mouth_bboxes,
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
        "face_embedding_shape": list(face_embeddings.shape),
        "mouth_embedding_shape": list(mouth_embeddings.shape),
        "face_modality": "AVAILABLE" if len(face_embeddings) else "NOT_APPLICABLE",
        "mouth_modality": "AVAILABLE" if len(mouth_embeddings) else "NOT_APPLICABLE",
        "elapsed_seconds": elapsed,
        "frames_per_second": decoded_frames / elapsed if elapsed else None,
        "output_bytes": bytes_written,
        "output_path": str(output_path),
    }


def select_samples(args: argparse.Namespace, runtime: dict[str, Any]) -> list[Sample]:
    selected_ids = parse_sample_ids(args.sample_id)
    if args.input_mode == "frames":
        if not args.frame_root.exists():
            raise FileNotFoundError(f"Frame root not found: {args.frame_root}")
        dirs = [path for path in sorted(args.frame_root.iterdir()) if path.is_dir()]
        if selected_ids:
            dirs = [path for path in dirs if path.name in selected_ids]
        samples = [Sample(path.name, None, path, None, None) for path in dirs]
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
            chosen = []
            for _, group in dataframe.groupby("semantic_class", sort=True):
                if len(group) < args.per_class:
                    raise RuntimeError("A requested class has fewer rows than --per-class.")
                chosen.append(group.sample(n=args.per_class, random_state=args.seed))
            dataframe = runtime["pd"].concat(chosen).sort_values("sample_id")
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


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.batch_size < 1:
        raise ValueError("--fps must be positive and --batch-size must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = require_runtime()
    device = resolve_device(runtime["torch"], args.device)
    samples = select_samples(args, runtime)
    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = args.log_root / f"convnext_{run_id}.jsonl"
    logging.info("Selected %d samples, mode=%s, device=%s", len(samples), args.input_mode, device)
    if device.startswith("cuda"):
        runtime["torch"].cuda.reset_peak_memory_stats()
    encoder = build_encoder(runtime, device)
    summaries: list[dict[str, Any]] = []
    with build_landmarker(runtime, args.face_landmarker) as landmarker:
        for position, sample in enumerate(samples, start=1):
            logging.info("[%d/%d] %s", position, len(samples), sample.sample_id)
            try:
                record = process_sample(sample, args, landmarker, encoder, runtime, device)
            except Exception as exc:  # A failed sample must be visible in the run log.
                record = {
                    "sample_id": sample.sample_id,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
                logging.exception("Feature extraction failed for %s", sample.sample_id)
            record["recorded_at"] = datetime.now(timezone.utc).isoformat()
            append_jsonl(jsonl_path, record)
            summaries.append(record)
            if record["status"] == "FAILED" and args.fail_fast:
                raise RuntimeError(f"Stopping after failure of {sample.sample_id}")

    completed = [record for record in summaries if record["status"] == "COMPLETED"]
    failed = [record for record in summaries if record["status"] == "FAILED"]
    skipped = [record for record in summaries if record["status"] == "SKIPPED_VALID_OUTPUT"]
    recomputed = [
        record for record in completed
        if record["existing_output_state"] == "RECOMPUTED_INVALID_OUTPUT"
    ]
    total_frames = sum(record.get("decoded_frames", 0) for record in completed)
    total_elapsed = sum(record.get("elapsed_seconds", 0.0) for record in completed)
    total_bytes = sum(record.get("output_bytes", 0) for record in completed)
    report = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "backbone": BACKBONE_NAME,
        "input_mode": args.input_mode,
        "device": device,
        "samples_selected": len(samples),
        "completed": len(completed),
        "completed_new": len(completed) - len(recomputed),
        "completed_recomputed_invalid_output": len(recomputed),
        "skipped_valid_output": len(skipped),
        "failed": len(failed),
        "accounted_samples": len(completed) + len(skipped) + len(failed),
        "failure_rate": len(failed) / len(samples),
        "frames": total_frames,
        "frames_per_second": total_frames / total_elapsed if total_elapsed else None,
        "videos_per_minute": len(completed) / total_elapsed * 60 if total_elapsed else None,
        "mean_output_bytes_per_video": total_bytes / len(completed) if completed else None,
        "peak_gpu_vram_bytes": runtime["torch"].cuda.max_memory_allocated() if device.startswith("cuda") else 0,
        "per_sample_log": str(jsonl_path),
    }
    report_path = args.log_root / f"convnext_{run_id}_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Run complete: %s", json.dumps(report, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
