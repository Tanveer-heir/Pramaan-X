"""Train a lightweight coarse audio-visual synchrony model from cached features.

Only REAL videos are used. Timestamp-aligned audio-mouth windows are positive
pairs and circularly shifted mouth windows from the same video are negatives.
The frozen ConvNeXt and XLSR-SLS backbones are never loaded by this script.

The source audio windows are about 4.04 seconds long, so this model establishes
coarse window-level synchrony. It must not be described as a subsecond lip-sync
detector.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "av_sync_features" / "aligned_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "av_sync"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "interim" / "training_logs" / "av_sync"
FEATURE_SCHEMA_VERSION = "av_sync_aligned_v1"
AUDIO_MODEL_SHA256 = "0aa36298a1a893bfb58a8d721dd30d046157f559da1165f95d2a0b69988c1bbd"
VISUAL_BACKBONE = "torchvision/convnext_tiny/IMAGENET1K_V1"
AUDIO_DIM = 1_024
MOUTH_DIM = 768
PAIRING_STRATEGY = "REAL_ALIGNED_VS_WITHIN_VIDEO_CIRCULAR_SHIFT_V1"


@dataclass(frozen=True)
class SourceExample:
    sample_id: str
    feature_path: Path
    window_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-sources", type=int, default=None)
    parser.add_argument("--max-validation-sources", type=int, default=None)
    return parser


def require_runtime() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Activate the project environment with NumPy and PyTorch installed.") from exc
    return {"np": np, "torch": torch}


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_aligned_arrays(payload: Any, np: Any, expected_sample_id: str | None = None) -> tuple[tuple[Any, Any, Any] | None, str | None]:
    required = {
        "schema_version", "sample_id", "av_modality", "audio_model_sha256",
        "visual_backbone", "timestamps_start_sec", "timestamps_end_sec",
        "audio_embeddings", "mouth_mean_embeddings", "mouth_motion_embeddings",
    }
    missing = required - set(payload.files if hasattr(payload, "files") else payload)
    if missing:
        return None, f"missing_keys:{','.join(sorted(missing))}"
    if str(payload["schema_version"].item()) != FEATURE_SCHEMA_VERSION:
        return None, "unsupported_schema"
    sample_id = str(payload["sample_id"].item())
    if expected_sample_id is not None and sample_id != expected_sample_id:
        return None, "sample_id_mismatch"
    if str(payload["audio_model_sha256"].item()) != AUDIO_MODEL_SHA256:
        return None, "audio_model_digest_mismatch"
    if str(payload["visual_backbone"].item()) != VISUAL_BACKBONE:
        return None, "visual_backbone_mismatch"
    modality = str(payload["av_modality"].item())
    if modality == "NOT_APPLICABLE":
        return None, "AV_NOT_APPLICABLE"
    if modality != "AVAILABLE":
        return None, f"invalid_av_modality:{modality}"
    audio = np.asarray(payload["audio_embeddings"], dtype=np.float32)
    mouth_mean = np.asarray(payload["mouth_mean_embeddings"], dtype=np.float32)
    mouth_motion = np.asarray(payload["mouth_motion_embeddings"], dtype=np.float32)
    count = len(audio)
    if audio.shape != (count, AUDIO_DIM):
        return None, f"invalid_audio_shape:{audio.shape}"
    if mouth_mean.shape != (count, MOUTH_DIM):
        return None, f"invalid_mouth_mean_shape:{mouth_mean.shape}"
    if mouth_motion.shape != (count, MOUTH_DIM):
        return None, f"invalid_mouth_motion_shape:{mouth_motion.shape}"
    if len(payload["timestamps_start_sec"]) != count or len(payload["timestamps_end_sec"]) != count:
        return None, "timestamp_length_mismatch"
    if count < 2:
        return None, "INSUFFICIENT_WINDOWS_FOR_WITHIN_VIDEO_NEGATIVE"
    if not (np.isfinite(audio).all() and np.isfinite(mouth_mean).all() and np.isfinite(mouth_motion).all()):
        return None, "NON_FINITE_FEATURES"
    return (audio, mouth_mean, mouth_motion), None


def inspect_feature(path: Path, np: Any, sample_id: str) -> tuple[int | None, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            arrays, reason = load_aligned_arrays(payload, np, sample_id)
    except (EOFError, OSError, ValueError, KeyError) as exc:
        return None, f"unreadable_feature:{type(exc).__name__}:{exc}"
    return (len(arrays[0]), None) if arrays is not None else (None, reason)


def build_sources(manifest_path: Path, feature_root: Path, limit: int | None, np: Any) -> tuple[list[SourceExample], list[dict[str, str]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "semantic_class"}
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"Manifest must contain {sorted(required)}")
    real_rows = sorted((row for row in rows if row["semantic_class"] == "REAL"), key=lambda row: row["sample_id"])
    if limit is not None:
        if limit < 1:
            raise ValueError("source limits must be positive")
        real_rows = real_rows[:limit]
    sources, excluded = [], []
    for row in real_rows:
        sample_id = row["sample_id"]
        path = feature_root / f"{sample_id}.npz"
        if not path.exists():
            excluded.append({"sample_id": sample_id, "reason": "MISSING_FEATURE_FILE"})
            continue
        count, reason = inspect_feature(path, np, sample_id)
        if reason is not None or count is None:
            excluded.append({"sample_id": sample_id, "reason": reason or "INVALID_FEATURE"})
            continue
        sources.append(SourceExample(sample_id, path, count))
    return sources, excluded


def circular_negative_indices(count: int, sample_id: str, seed: int) -> Any:
    if count < 2:
        raise ValueError("At least two windows are required")
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    shift = 1 + int.from_bytes(digest[:8], "big") % (count - 1)
    return [(index + shift) % count for index in range(count)]


def build_pair_arrays(source: SourceExample, np: Any, seed: int) -> tuple[Any, Any, Any, Any]:
    with np.load(source.feature_path, allow_pickle=False) as payload:
        arrays, reason = load_aligned_arrays(payload, np, source.sample_id)
    if arrays is None:
        raise RuntimeError(f"Feature changed after preflight: {source.sample_id}: {reason}")
    audio, mouth_mean, mouth_motion = arrays
    negative_indices = np.asarray(circular_negative_indices(len(audio), source.sample_id, seed), dtype=np.int64)
    pair_audio = np.concatenate((audio, audio), axis=0)
    pair_mean = np.concatenate((mouth_mean, mouth_mean[negative_indices]), axis=0)
    pair_motion = np.concatenate((mouth_motion, mouth_motion[negative_indices]), axis=0)
    labels = np.concatenate((np.ones(len(audio)), np.zeros(len(audio)))).astype(np.float32)
    return pair_audio, pair_mean, pair_motion, labels


def build_components(torch: Any) -> tuple[Any, Any]:
    class AVPairDataset(torch.utils.data.Dataset):
        def __init__(self, sources: list[SourceExample], np: Any, seed: int):
            audio, mouth_mean, mouth_motion, labels, sample_ids = [], [], [], [], []
            for source in sources:
                source_audio, source_mean, source_motion, source_labels = build_pair_arrays(source, np, seed)
                audio.append(source_audio)
                mouth_mean.append(source_mean)
                mouth_motion.append(source_motion)
                labels.append(source_labels)
                sample_ids.extend([source.sample_id] * len(source_labels))
            self.audio = torch.from_numpy(np.concatenate(audio).copy())
            self.mouth_mean = torch.from_numpy(np.concatenate(mouth_mean).copy())
            self.mouth_motion = torch.from_numpy(np.concatenate(mouth_motion).copy())
            self.labels = torch.from_numpy(np.concatenate(labels).copy())
            self.sample_ids = sample_ids

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any, str]:
            return self.audio[index], self.mouth_mean[index], self.mouth_motion[index], self.labels[index], self.sample_ids[index]

    class AVSyncModel(torch.nn.Module):
        def __init__(self, projection_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.audio_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(AUDIO_DIM), torch.nn.Linear(AUDIO_DIM, projection_dim),
                torch.nn.GELU(), torch.nn.Dropout(dropout),
            )
            self.mouth_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(MOUTH_DIM * 2), torch.nn.Linear(MOUTH_DIM * 2, projection_dim),
                torch.nn.GELU(), torch.nn.Dropout(dropout),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(projection_dim * 4),
                torch.nn.Linear(projection_dim * 4, hidden_dim), torch.nn.GELU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, audio: Any, mouth_mean: Any, mouth_motion: Any) -> Any:
            audio_hidden = self.audio_projection(audio)
            mouth_hidden = self.mouth_projection(torch.cat((mouth_mean, mouth_motion), dim=-1))
            joint = torch.cat((audio_hidden, mouth_hidden, torch.abs(audio_hidden - mouth_hidden), audio_hidden * mouth_hidden), dim=-1)
            return self.classifier(joint).squeeze(-1)

    return AVPairDataset, AVSyncModel


def rankdata_average(values: Any, np: Any) -> Any:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return ranks


def binary_metrics(labels: Any, probabilities: Any, np: Any) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives, negatives = int(labels.sum()), len(labels) - int(labels.sum())
    if positives == 0 or negatives == 0:
        raise RuntimeError("Validation pairs must contain both classes")
    ranks = rankdata_average(probabilities, np)
    roc_auc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels, sorted_probabilities = labels[order], probabilities[order]
    cumulative = np.cumsum(sorted_labels)
    average_precision, start = 0.0, 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        group_positives = int(sorted_labels[start:end].sum())
        if group_positives:
            average_precision += float(cumulative[end - 1] / end) * group_positives / positives
        start = end
    best_balanced_accuracy, best_threshold = -1.0, 0.5
    for threshold in np.unique(probabilities):
        predictions = probabilities >= threshold
        score = (float(predictions[labels == 1].mean()) + float((~predictions[labels == 0]).mean())) / 2
        if score > best_balanced_accuracy:
            best_balanced_accuracy, best_threshold = score, float(threshold)
    return {
        "roc_auc": float(roc_auc), "average_precision": float(average_precision),
        "balanced_accuracy": best_balanced_accuracy, "validation_threshold": best_threshold,
    }


def evaluate(model: Any, loader: Any, device: str, runtime: dict[str, Any]) -> dict[str, float]:
    torch, np = runtime["torch"], runtime["np"]
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for audio, mouth_mean, mouth_motion, batch_labels, _ in loader:
            logits.append(model(audio.to(device), mouth_mean.to(device), mouth_motion.to(device)).cpu())
            labels.append(batch_labels)
    logits_array = torch.cat(logits).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -80.0, 80.0)))
    return binary_metrics(torch.cat(labels).numpy(), probabilities, np)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def atomic_torch_save(payload: dict[str, Any], path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
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


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.projection_dim, args.hidden_dim) < 1:
        raise ValueError("epochs, patience, batch size, projection dimension, and hidden dimension must be positive")
    runtime = require_runtime()
    torch, np = runtime["torch"], runtime["np"]
    device = resolve_device(torch, args.device)
    set_seed(args.seed, torch)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_sources, train_excluded = build_sources(args.train_manifest, args.feature_root, args.max_train_sources, np)
    validation_sources, validation_excluded = build_sources(args.validation_manifest, args.feature_root, args.max_validation_sources, np)
    if not train_sources or not validation_sources:
        raise RuntimeError("No usable REAL sources. Build train and validation AV alignment features first.")
    write_jsonl(args.log_dir / f"{run_id}_train_excluded.jsonl", train_excluded)
    write_jsonl(args.log_dir / f"{run_id}_validation_excluded.jsonl", validation_excluded)
    Dataset, AVSyncModel = build_components(torch)
    train_dataset = Dataset(train_sources, np, args.seed)
    validation_dataset = Dataset(validation_sources, np, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=device.startswith("cuda"),
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.startswith("cuda"),
    )
    model = AVSyncModel(args.projection_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, stale_epochs = float("-inf"), 0, 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.checkpoint_dir / f"av_sync_{run_id}.pt"
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        losses = []
        for audio, mouth_mean, mouth_motion, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(audio.to(device), mouth_mean.to(device), mouth_motion.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation_loader, device, runtime)
        metrics.update({"epoch": float(epoch), "train_loss": sum(losses) / len(losses), "epoch_seconds": perf_counter() - started})
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        if metrics["roc_auc"] > best_auc:
            best_auc, best_epoch, stale_epochs = metrics["roc_auc"], epoch, 0
            atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "audio_dim": AUDIO_DIM, "mouth_dim": MOUTH_DIM,
                    "projection_dim": args.projection_dim, "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
                "best_validation_metrics": metrics,
                "train_manifest_sha256": manifest_hash(args.train_manifest),
                "validation_manifest_sha256": manifest_hash(args.validation_manifest),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "audio_model_sha256": AUDIO_MODEL_SHA256,
                "visual_backbone": VISUAL_BACKBONE,
                "pairing_strategy": PAIRING_STRATEGY,
                "label_semantics": {"0": "WITHIN_VIDEO_SHIFTED", "1": "TIMESTAMP_ALIGNED"},
                "temporal_resolution": "COARSE_4_SECOND_WINDOWS",
                "seed": args.seed,
            }, checkpoint_path, torch)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    report = {
        "run_id": run_id, "device": device,
        "train_real_sources": len(train_sources), "validation_real_sources": len(validation_sources),
        "train_pairs": len(train_dataset), "validation_pairs": len(validation_dataset),
        "train_excluded": len(train_excluded), "validation_excluded": len(validation_excluded),
        "best_epoch": best_epoch, "best_roc_auc": best_auc,
        "checkpoint_path": str(checkpoint_path), "pairing_strategy": PAIRING_STRATEGY,
        "temporal_resolution": "COARSE_4_SECOND_WINDOWS", "history": history,
    }
    atomic_json_save(report, args.log_dir / f"{run_id}_summary.json")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
