"""Train a small, independent blink-consistency GRU from cached EAR features.

The model consumes only artifacts created by preprocessing script 07.  It
never reads video or images, and it never turns a missing eye sequence into
zero-valued evidence.  Audio-only manipulations use visual_label=0 because
their visible eye behaviour is not manipulated.
"""

from __future__ import annotations

import argparse
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
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "blink_features" / "ear_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "blink_gru"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "interim" / "training_logs" / "blink_gru"
SCHEMA_VERSION = "blink_ear_v1"
FEATURE_DIM = 7


@dataclass(frozen=True)
class Example:
    sample_id: str
    feature_path: Path
    label: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--min-valid-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    return parser


def require_runtime() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    except ImportError as exc:
        raise RuntimeError("Activate the ML environment with numpy, pandas, torch and scikit-learn installed.") from exc
    return {
        "np": np,
        "pd": pd,
        "torch": torch,
        "average_precision_score": average_precision_score,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
    }


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return requested


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def blink_sequence(payload: Any, np: Any, min_valid_frames: int) -> tuple[Any | None, str | None]:
    required = {"schema_version", "availability", "features", "timestamps_sec", "source_frame_indices"}
    missing = required - set(payload.files if hasattr(payload, "files") else payload)
    if missing:
        return None, f"missing_keys:{','.join(sorted(missing))}"
    if str(payload["schema_version"].item()) != SCHEMA_VERSION:
        return None, "unsupported_schema"
    if str(payload["availability"].item()) != "AVAILABLE":
        reason = str(payload.get("not_applicable_reason", "UNSPECIFIED")) if hasattr(payload, "get") else "UNSPECIFIED"
        return None, f"NOT_APPLICABLE:{reason}"
    sequence = payload["features"]
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        return None, f"invalid_feature_shape:{sequence.shape}"
    if len(sequence) != len(payload["timestamps_sec"]) or len(sequence) != len(payload["source_frame_indices"]):
        return None, "feature_temporal_length_mismatch"
    if len(sequence) < min_valid_frames:
        return None, f"INSUFFICIENT_VALID_EYE_FRAMES:{len(sequence)}/{min_valid_frames}"
    if not np.isfinite(sequence).all():
        return None, "NONFINITE_EAR_FEATURES"
    return sequence.astype(np.float32, copy=False), None


def inspect_feature(path: Path, np: Any, min_valid_frames: int) -> tuple[int | None, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            sequence, reason = blink_sequence(payload, np, min_valid_frames)
    except (EOFError, OSError, ValueError, KeyError) as exc:
        return None, f"unreadable_feature:{type(exc).__name__}:{exc}"
    return (len(sequence), None) if sequence is not None else (None, reason)


def build_examples(manifest_path: Path, feature_root: Path, limit: int | None, runtime: dict[str, Any], min_valid_frames: int) -> tuple[list[Example], list[dict[str, str]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    dataframe = runtime["pd"].read_csv(manifest_path)
    required = {"sample_id", "visual_label"}
    missing = required - set(dataframe.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    dataframe = dataframe.sort_values("sample_id")
    if limit is not None:
        dataframe = dataframe.head(limit)
    examples: list[Example] = []
    excluded: list[dict[str, str]] = []
    for row in dataframe.itertuples(index=False):
        sample_id = str(row.sample_id)
        feature_path = feature_root / f"{sample_id}.npz"
        if not feature_path.exists():
            excluded.append({"sample_id": sample_id, "reason": "MISSING_FEATURE_FILE"})
            continue
        _, reason = inspect_feature(feature_path, runtime["np"], min_valid_frames)
        if reason is not None:
            excluded.append({"sample_id": sample_id, "reason": reason})
            continue
        # visual_label deliberately makes AUDIO_MANIPULATION a negative label here.
        examples.append(Example(sample_id, feature_path, int(row.visual_label)))
    return examples, excluded


def train_normalizer(examples: list[Example], np: Any, min_valid_frames: int) -> tuple[Any, Any]:
    total = np.zeros(FEATURE_DIM, dtype=np.float64)
    total_squared = np.zeros(FEATURE_DIM, dtype=np.float64)
    count = 0
    for example in examples:
        with np.load(example.feature_path, allow_pickle=False) as payload:
            sequence, reason = blink_sequence(payload, np, min_valid_frames)
        if sequence is None:
            raise RuntimeError(f"Feature changed after preflight: {example.sample_id}: {reason}")
        total += sequence.sum(axis=0, dtype=np.float64)
        total_squared += np.square(sequence, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(sequence)
    if count == 0:
        raise RuntimeError("No frames available to compute training normalization.")
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def build_components(torch: Any) -> tuple[Any, Any, Any]:
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list[Example], np: Any, mean: Any, std: Any, min_valid_frames: int):
            self.examples, self.np, self.mean, self.std = examples, np, mean, std
            self.min_valid_frames = min_valid_frames

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> tuple[Any, int, str]:
            example = self.examples[index]
            with self.np.load(example.feature_path, allow_pickle=False) as payload:
                sequence, reason = blink_sequence(payload, self.np, self.min_valid_frames)
            if sequence is None:
                raise RuntimeError(f"Feature changed after preflight: {example.sample_id}: {reason}")
            normalized = (sequence - self.mean) / self.std
            return torch.from_numpy(normalized), example.label, example.sample_id

    def collate(batch: list[tuple[Any, int, str]]) -> tuple[Any, Any, Any, list[str]]:
        sequences, labels, sample_ids = zip(*batch, strict=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        return padded, lengths, torch.tensor(labels, dtype=torch.float32), list(sample_ids)

    class BlinkGRU(torch.nn.Module):
        def __init__(self, hidden_dim: int, dropout: float):
            super().__init__()
            self.gru = torch.nn.GRU(FEATURE_DIM, hidden_dim, num_layers=1, batch_first=True)
            self.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, 1))

        def forward(self, padded: Any, lengths: Any) -> Any:
            packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
            return self.classifier(hidden[-1]).squeeze(-1)

    return FeatureDataset, collate, BlinkGRU


def evaluate(model: Any, loader: Any, device: str, runtime: dict[str, Any]) -> dict[str, float]:
    torch, np = runtime["torch"], runtime["np"]
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for padded, lengths, batch_labels, _ in loader:
            logits.append(model(padded.to(device), lengths.to(device)).cpu())
            labels.append(batch_labels)
    logits_np = torch.cat(logits).numpy()
    labels_np = torch.cat(labels).numpy().astype(int)
    if len(np.unique(labels_np)) != 2:
        raise RuntimeError("Validation data must contain both visual classes.")
    probabilities = 1.0 / (1.0 + np.exp(-logits_np))
    fpr, tpr, thresholds = runtime["roc_curve"](labels_np, probabilities)
    balanced_accuracy = (tpr + (1.0 - fpr)) / 2.0
    best_index = int(np.argmax(balanced_accuracy))
    return {
        "roc_auc": float(runtime["roc_auc_score"](labels_np, probabilities)),
        "average_precision": float(runtime["average_precision_score"](labels_np, probabilities)),
        "balanced_accuracy": float(balanced_accuracy[best_index]),
        "validation_threshold": float(thresholds[best_index]),
    }


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


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim, args.min_valid_frames) < 1:
        raise ValueError("epochs, patience, batch-size, hidden-dim and min-valid-frames must be positive")
    runtime = require_runtime()
    torch = runtime["torch"]
    device = resolve_device(torch, args.device)
    set_seed(args.seed, torch)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_examples, train_excluded = build_examples(args.train_manifest, args.feature_root, args.max_train_samples, runtime, args.min_valid_frames)
    validation_examples, validation_excluded = build_examples(args.validation_manifest, args.feature_root, args.max_validation_samples, runtime, args.min_valid_frames)
    if not train_examples or not validation_examples:
        raise RuntimeError("No usable examples. Extract blink features for both train and validation first.")
    write_jsonl(args.log_dir / f"{run_id}_train_excluded.jsonl", train_excluded)
    write_jsonl(args.log_dir / f"{run_id}_validation_excluded.jsonl", validation_excluded)
    labels = [example.label for example in train_examples]
    class_counts = {label: labels.count(label) for label in (0, 1)}
    if not class_counts[0] or not class_counts[1]:
        raise RuntimeError(f"Training data requires both visual classes; got {class_counts}.")
    mean, std = train_normalizer(train_examples, runtime["np"], args.min_valid_frames)
    FeatureDataset, collate, BlinkGRU = build_components(torch)
    train_dataset = FeatureDataset(train_examples, runtime["np"], mean, std, args.min_valid_frames)
    validation_dataset = FeatureDataset(validation_examples, runtime["np"], mean, std, args.min_valid_frames)
    weights = torch.DoubleTensor([1.0 / class_counts[example.label] for example in train_examples])
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    common_loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.startswith("cuda"), "collate_fn": collate, "persistent_workers": args.num_workers > 0}
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, **common_loader_args)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, **common_loader_args)
    model = BlinkGRU(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_score, best_epoch, stale_epochs = float("-inf"), 0, 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.checkpoint_dir / f"blink_gru_{run_id}.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        model.train()
        losses = []
        for padded, lengths, batch_labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(padded.to(device), lengths.to(device)), batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation_loader, device, runtime)
        metrics.update({"epoch": float(epoch), "train_loss": sum(losses) / len(losses), "epoch_seconds": perf_counter() - epoch_started})
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        if metrics["average_precision"] > best_score:
            best_score, best_epoch, stale_epochs = metrics["average_precision"], epoch, 0
            atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "model_config": {"feature_dim": FEATURE_DIM, "hidden_dim": args.hidden_dim, "dropout": args.dropout},
                "normalizer": {"mean": mean, "std": std},
                "feature_schema_version": SCHEMA_VERSION,
                "best_validation_metrics": metrics,
                "train_manifest_sha256": manifest_hash(args.train_manifest),
                "validation_manifest_sha256": manifest_hash(args.validation_manifest),
                "seed": args.seed,
            }, checkpoint_path, torch)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    report = {
        "run_id": run_id,
        "device": device,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_excluded": len(train_excluded),
        "validation_excluded": len(validation_excluded),
        "train_class_counts": class_counts,
        "best_epoch": best_epoch,
        "best_average_precision": best_score,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / f"{run_id}_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
