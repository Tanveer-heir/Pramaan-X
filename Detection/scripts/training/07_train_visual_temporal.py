"""Train a lightweight temporal visual classifier from cached ConvNeXt features.

This script never loads ConvNeXt.  It trains only a small GRU and attention
head from the compact embeddings written by preprocessing script 06.
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
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "visual_temporal"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "interim" / "training_logs" / "visual_temporal"
SCHEMA_VERSION = "visual_embedding_v1"
FACE_DIM = 768
MOUTH_DIM = 768
INPUT_DIM = FACE_DIM + MOUTH_DIM


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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.25)
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
        raise RuntimeError(
            "Activate the ML environment with numpy, pandas, torch and scikit-learn installed."
        ) from exc
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


def shared_frame_sequence(payload: Any, np: Any) -> tuple[Any | None, str | None]:
    """Return aligned face+mouth embeddings without inventing missing values."""
    required = {
        "schema_version", "face_embeddings", "face_frame_indices",
        "mouth_embeddings", "mouth_frame_indices",
    }
    missing = required - set(payload.files if hasattr(payload, "files") else payload)
    if missing:
        return None, f"missing_keys:{','.join(sorted(missing))}"
    if str(payload["schema_version"].item()) != SCHEMA_VERSION:
        return None, "unsupported_schema"
    face = payload["face_embeddings"]
    mouth = payload["mouth_embeddings"]
    face_indices = payload["face_frame_indices"]
    mouth_indices = payload["mouth_frame_indices"]
    if face.ndim != 2 or face.shape[1] != FACE_DIM:
        return None, "invalid_face_embedding_shape"
    if mouth.ndim != 2 or mouth.shape[1] != MOUTH_DIM:
        return None, "invalid_mouth_embedding_shape"
    if len(face) != len(face_indices) or len(mouth) != len(mouth_indices):
        return None, "embedding_index_length_mismatch"
    common, face_positions, mouth_positions = np.intersect1d(
        face_indices, mouth_indices, assume_unique=False, return_indices=True
    )
    if len(common) == 0:
        return None, "NO_SHARED_FACE_MOUTH_FRAMES"
    return np.concatenate((face[face_positions], mouth[mouth_positions]), axis=1).astype(np.float32, copy=False), None


def inspect_feature(path: Path, np: Any) -> tuple[int | None, str | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            sequence, reason = shared_frame_sequence(payload, np)
    except (EOFError, OSError, ValueError, KeyError) as exc:
        return None, f"unreadable_feature:{type(exc).__name__}:{exc}"
    return (len(sequence), None) if sequence is not None else (None, reason)


def build_examples(manifest_path: Path, feature_root: Path, limit: int | None, runtime: dict[str, Any]) -> tuple[list[Example], list[dict[str, str]]]:
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
        path = feature_root / f"{sample_id}.npz"
        if not path.exists():
            excluded.append({"sample_id": sample_id, "reason": "MISSING_FEATURE_FILE"})
            continue
        _, reason = inspect_feature(path, runtime["np"])
        if reason is not None:
            excluded.append({"sample_id": sample_id, "reason": reason})
            continue
        examples.append(Example(sample_id, path, int(row.visual_label)))
    return examples, excluded


def build_components(torch: Any) -> tuple[Any, Any, Any]:
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list[Example], np: Any):
            self.examples = examples
            self.np = np

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> tuple[Any, int, str]:
            example = self.examples[index]
            with self.np.load(example.feature_path, allow_pickle=False) as payload:
                sequence, reason = shared_frame_sequence(payload, self.np)
            if reason is not None or sequence is None:
                raise RuntimeError(f"Feature changed after preflight: {example.sample_id}: {reason}")
            return torch.from_numpy(sequence), example.label, example.sample_id

    def collate(batch: list[tuple[Any, int, str]]) -> tuple[Any, Any, Any, list[str]]:
        sequences, labels, sample_ids = zip(*batch, strict=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Padding is strictly masked before temporal pooling. It is not evidence.
        return padded, lengths, torch.tensor(labels, dtype=torch.float32), list(sample_ids)

    class TemporalVisualClassifier(torch.nn.Module):
        def __init__(self, hidden_dim: int, dropout: float):
            super().__init__()
            self.input_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(INPUT_DIM),
                torch.nn.Linear(INPUT_DIM, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.gru = torch.nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = torch.nn.Linear(hidden_dim * 2, 1)
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim * 2),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim * 2, 1),
            )

        def forward(self, padded: Any, lengths: Any) -> Any:
            projected = self.input_projection(padded)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                projected, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.gru(packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            positions = torch.arange(output.shape[1], device=output.device).unsqueeze(0)
            valid = positions < lengths.unsqueeze(1)
            scores = self.attention(output).squeeze(-1).masked_fill(~valid, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(output * weights.unsqueeze(-1), dim=1)
            return self.classifier(pooled).squeeze(-1)

    return FeatureDataset, collate, TemporalVisualClassifier


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
    probabilities = 1.0 / (1.0 + np.exp(-logits_np))
    if len(np.unique(labels_np)) != 2:
        raise RuntimeError("Validation split must contain both visual classes.")
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
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch-size must all be positive")
    runtime = require_runtime()
    torch = runtime["torch"]
    device = resolve_device(torch, args.device)
    set_seed(args.seed, torch)
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    train_examples, train_excluded = build_examples(args.train_manifest, args.feature_root, args.max_train_samples, runtime)
    validation_examples, validation_excluded = build_examples(args.validation_manifest, args.feature_root, args.max_validation_samples, runtime)
    if not train_examples or not validation_examples:
        raise RuntimeError("No usable examples. Extract cached features for both train and validation first.")
    write_jsonl(args.log_dir / f"{run_id}_train_excluded.jsonl", train_excluded)
    write_jsonl(args.log_dir / f"{run_id}_validation_excluded.jsonl", validation_excluded)
    FeatureDataset, collate, TemporalVisualClassifier = build_components(torch)
    train_dataset = FeatureDataset(train_examples, runtime["np"])
    validation_dataset = FeatureDataset(validation_examples, runtime["np"])
    labels = [example.label for example in train_examples]
    counts = {label: labels.count(label) for label in (0, 1)}
    if not counts[0] or not counts[1]:
        raise RuntimeError(f"Training data requires both visual classes; got {counts}.")
    weights = torch.DoubleTensor([1.0 / counts[example.label] for example in train_examples])
    generator = torch.Generator().manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=device.startswith("cuda"), collate_fn=collate, persistent_workers=args.num_workers > 0)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.startswith("cuda"), collate_fn=collate, persistent_workers=args.num_workers > 0)
    model = TemporalVisualClassifier(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_score, best_epoch, stale_epochs = float("-inf"), 0, 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.checkpoint_dir / f"visual_temporal_{run_id}.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        model.train()
        losses = []
        for padded, lengths, batch_labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(padded.to(device), lengths.to(device))
            loss = criterion(logits, batch_labels.to(device))
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
                "model_config": {"hidden_dim": args.hidden_dim, "dropout": args.dropout, "input_dim": INPUT_DIM},
                "best_validation_metrics": metrics,
                "train_manifest_sha256": manifest_hash(args.train_manifest),
                "validation_manifest_sha256": manifest_hash(args.validation_manifest),
                "feature_schema_version": SCHEMA_VERSION,
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
        "train_class_counts": counts,
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
