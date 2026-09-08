"""Train a lightweight audio anti-spoofing head from cached XLSR-SLS features.

The frozen 340M-parameter XLSR-SLS model is never loaded here. Training uses
only compact per-window embeddings produced by preprocessing script 13.
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
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "audio_embeddings" / "xlsr_sls_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "audio_classifier"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "interim" / "training_logs" / "audio_classifier"
FEATURE_SCHEMA_VERSION = "audio_embedding_v1"
MODEL_SHA256 = "0aa36298a1a893bfb58a8d721dd30d046157f559da1165f95d2a0b69988c1bbd"
INPUT_DIM = 1_024


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
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    return parser


def require_runtime() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Activate the project environment with NumPy and CUDA PyTorch installed.") from exc
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


def load_sequence(payload: Any, np: Any) -> tuple[Any | None, str | None]:
    required = {"schema_version", "model_sha256", "audio_modality", "embeddings"}
    missing = required - set(payload.files if hasattr(payload, "files") else payload)
    if missing:
        return None, f"missing_keys:{','.join(sorted(missing))}"
    if str(payload["schema_version"].item()) != FEATURE_SCHEMA_VERSION:
        return None, "unsupported_schema"
    if str(payload["model_sha256"].item()) != MODEL_SHA256:
        return None, "checkpoint_digest_mismatch"
    modality = str(payload["audio_modality"].item())
    if modality == "NOT_APPLICABLE":
        return None, "AUDIO_NOT_APPLICABLE"
    if modality != "AVAILABLE":
        return None, f"invalid_audio_modality:{modality}"
    embeddings = payload["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[1] != INPUT_DIM:
        return None, f"invalid_embedding_shape:{embeddings.shape}"
    if len(embeddings) == 0:
        return None, "EMPTY_AUDIO_SEQUENCE"
    if not np.isfinite(embeddings).all():
        return None, "NON_FINITE_AUDIO_SEQUENCE"
    return embeddings.astype(np.float32, copy=False), None


def inspect_feature(path: Path, np: Any) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            _, reason = load_sequence(payload, np)
    except (EOFError, OSError, ValueError, KeyError) as exc:
        return f"unreadable_feature:{type(exc).__name__}:{exc}"
    return reason


def build_examples(manifest_path: Path, feature_root: Path, limit: int | None, np: Any) -> tuple[list[Example], list[dict[str, str]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "audio_label"}
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"Manifest must contain {sorted(required)}")
    rows.sort(key=lambda row: row["sample_id"])
    if limit is not None:
        if limit < 1:
            raise ValueError("sample limits must be positive")
        rows = rows[:limit]
    examples, excluded = [], []
    for row in rows:
        sample_id = row["sample_id"]
        feature_path = feature_root / f"{sample_id}.npz"
        if not feature_path.exists():
            excluded.append({"sample_id": sample_id, "reason": "MISSING_FEATURE_FILE"})
            continue
        reason = inspect_feature(feature_path, np)
        if reason is not None:
            excluded.append({"sample_id": sample_id, "reason": reason})
            continue
        label = int(row["audio_label"])
        if label not in (0, 1):
            raise RuntimeError(f"Invalid audio label for {sample_id}: {label}")
        examples.append(Example(sample_id, feature_path, label))
    return examples, excluded


def build_components(torch: Any) -> tuple[Any, Any, Any]:
    class AudioFeatureDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list[Example], np: Any):
            self.items = []
            for example in examples:
                with np.load(example.feature_path, allow_pickle=False) as payload:
                    sequence, reason = load_sequence(payload, np)
                if sequence is None:
                    raise RuntimeError(f"Feature changed after preflight: {example.sample_id}: {reason}")
                self.items.append((torch.from_numpy(sequence.copy()), example.label, example.sample_id))

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> tuple[Any, int, str]:
            return self.items[index]

    def collate(batch: list[tuple[Any, int, str]]) -> tuple[Any, Any, Any, list[str]]:
        sequences, labels, sample_ids = zip(*batch, strict=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        return padded, lengths, torch.tensor(labels, dtype=torch.float32), list(sample_ids)

    class AudioClassifier(torch.nn.Module):
        def __init__(self, hidden_dim: int, dropout: float):
            super().__init__()
            self.projection = torch.nn.Sequential(
                torch.nn.LayerNorm(INPUT_DIM),
                torch.nn.Linear(INPUT_DIM, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.attention = torch.nn.Linear(hidden_dim, 1)
            self.window_classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, padded: Any, lengths: Any, return_window_logits: bool = False) -> Any:
            hidden = self.projection(padded)
            positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            valid = positions < lengths.unsqueeze(1)
            attention_logits = self.attention(hidden).squeeze(-1).masked_fill(~valid, float("-inf"))
            attention_weights = torch.softmax(attention_logits, dim=1)
            window_logits = self.window_classifier(hidden).squeeze(-1)
            video_logits = torch.sum(window_logits * attention_weights, dim=1)
            if return_window_logits:
                return video_logits, window_logits.masked_fill(~valid, float("nan")), attention_weights
            return video_logits

    return AudioFeatureDataset, collate, AudioClassifier


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
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("Validation split must contain both audio classes")
    ranks = rankdata_average(probabilities, np)
    roc_auc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    descending = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[descending]
    sorted_probabilities = probabilities[descending]
    cumulative_positives = np.cumsum(sorted_labels)
    average_precision = 0.0
    group_start = 0
    while group_start < len(labels):
        group_end = group_start + 1
        while group_end < len(labels) and sorted_probabilities[group_end] == sorted_probabilities[group_start]:
            group_end += 1
        group_positives = int(sorted_labels[group_start:group_end].sum())
        if group_positives:
            precision_at_group = float(cumulative_positives[group_end - 1] / group_end)
            average_precision += precision_at_group * group_positives / positives
        group_start = group_end
    thresholds = np.unique(probabilities)
    best_balanced_accuracy, best_threshold = -1.0, 0.5
    for threshold in thresholds:
        predictions = probabilities >= threshold
        true_positive_rate = float(predictions[labels == 1].mean())
        true_negative_rate = float((~predictions[labels == 0]).mean())
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy, best_threshold = balanced_accuracy, float(threshold)
    return {
        "roc_auc": float(roc_auc), "average_precision": float(average_precision),
        "balanced_accuracy": best_balanced_accuracy, "validation_threshold": best_threshold,
    }


def evaluate(model: Any, loader: Any, device: str, runtime: dict[str, Any]) -> dict[str, float]:
    torch, np = runtime["torch"], runtime["np"]
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for padded, lengths, batch_labels, _ in loader:
            logits.append(model(padded.to(device, non_blocking=True), lengths.to(device, non_blocking=True)).cpu())
            labels.append(batch_labels)
    logits_array = torch.cat(logits).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits_array))
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
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        raise ValueError("epochs, patience, batch size, and hidden dimension must be positive")
    runtime = require_runtime()
    torch, np = runtime["torch"], runtime["np"]
    device = resolve_device(torch, args.device)
    set_seed(args.seed, torch)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_examples, train_excluded = build_examples(args.train_manifest, args.feature_root, args.max_train_samples, np)
    validation_examples, validation_excluded = build_examples(args.validation_manifest, args.feature_root, args.max_validation_samples, np)
    if not train_examples or not validation_examples:
        raise RuntimeError("No usable examples. Extract train and validation audio features first.")
    write_jsonl(args.log_dir / f"{run_id}_train_excluded.jsonl", train_excluded)
    write_jsonl(args.log_dir / f"{run_id}_validation_excluded.jsonl", validation_excluded)
    Dataset, collate, AudioClassifier = build_components(torch)
    train_dataset, validation_dataset = Dataset(train_examples, np), Dataset(validation_examples, np)
    labels = [example.label for example in train_examples]
    counts = {label: labels.count(label) for label in (0, 1)}
    if not counts[0] or not counts[1]:
        raise RuntimeError(f"Training data requires both audio classes; got {counts}")
    weights = torch.DoubleTensor([1.0 / counts[example.label] for example in train_examples])
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=device.startswith("cuda"), collate_fn=collate,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.startswith("cuda"), collate_fn=collate,
    )
    model = AudioClassifier(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_score, best_epoch, stale_epochs = float("-inf"), 0, 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.checkpoint_dir / f"audio_classifier_{run_id}.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        model.train()
        losses = []
        for padded, lengths, batch_labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(padded.to(device, non_blocking=True), lengths.to(device, non_blocking=True))
            loss = criterion(logits, batch_labels.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation_loader, device, runtime)
        metrics.update({
            "epoch": float(epoch), "train_loss": sum(losses) / len(losses),
            "epoch_seconds": perf_counter() - epoch_started,
        })
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        if metrics["roc_auc"] > best_score:
            best_score, best_epoch, stale_epochs = metrics["roc_auc"], epoch, 0
            atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "model_config": {"input_dim": INPUT_DIM, "hidden_dim": args.hidden_dim, "dropout": args.dropout},
                "best_validation_metrics": metrics,
                "train_manifest_sha256": manifest_hash(args.train_manifest),
                "validation_manifest_sha256": manifest_hash(args.validation_manifest),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_model_sha256": MODEL_SHA256,
                "label_semantics": {"0": "REAL_AUDIO", "1": "FAKE_AUDIO"},
                "seed": args.seed,
            }, checkpoint_path, torch)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    report = {
        "run_id": run_id, "device": device, "train_examples": len(train_examples),
        "validation_examples": len(validation_examples), "train_excluded": len(train_excluded),
        "validation_excluded": len(validation_excluded), "train_class_counts": counts,
        "best_epoch": best_epoch, "best_roc_auc": best_score,
        "checkpoint_path": str(checkpoint_path), "history": history,
    }
    atomic_json_save(report, args.log_dir / f"{run_id}_summary.json")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
