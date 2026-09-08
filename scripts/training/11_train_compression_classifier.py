"""Train a lightweight compression classifier from controlled feature pairs."""

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
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "compression_features" / "controlled_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "compression_classifier"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "interim" / "training_logs" / "compression_classifier"
FEATURE_SCHEMA_VERSION = "compression_video_v1"
PAIR_SCHEMA_VERSION = "compression_controlled_pair_v1"
FEATURE_DIM = 20


@dataclass(frozen=True)
class Example:
    source_sample_id: str
    feature_path: Path
    label: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
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
        "np": np, "pd": pd, "torch": torch,
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


def pair_feature(payload: Any, np: Any, expected_source: str, expected_label: int) -> tuple[Any | None, str | None]:
    required = {
        "schema_version", "pair_schema_version", "source_sample_id",
        "compression_label", "controlled_generation", "features", "availability",
    }
    missing = required - set(payload.files if hasattr(payload, "files") else payload)
    if missing:
        return None, f"missing_keys:{','.join(sorted(missing))}"
    if str(payload["schema_version"].item()) != FEATURE_SCHEMA_VERSION:
        return None, "unsupported_feature_schema"
    if str(payload["pair_schema_version"].item()) != PAIR_SCHEMA_VERSION:
        return None, "unsupported_pair_schema"
    if str(payload["availability"].item()) != "AVAILABLE":
        return None, "NOT_APPLICABLE"
    if str(payload["source_sample_id"].item()) != expected_source:
        return None, "source_sample_id_mismatch"
    if int(payload["compression_label"].item()) != expected_label:
        return None, "compression_label_mismatch"
    if int(payload["controlled_generation"].item()) != expected_label + 1:
        return None, "controlled_generation_mismatch"
    features = payload["features"]
    if features.shape != (FEATURE_DIM,):
        return None, f"invalid_feature_shape:{features.shape}"
    if not np.isfinite(features).all():
        return None, "nonfinite_features"
    return features.astype(np.float32, copy=False), None


def pair_path(feature_root: Path, sample_id: str, generation: int) -> Path:
    return feature_root / f"{sample_id}_generation{generation}.npz"


def inspect_pair(path: Path, np: Any, sample_id: str, label: int) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            _, reason = pair_feature(payload, np, sample_id, label)
    except (EOFError, OSError, ValueError, KeyError) as exc:
        return f"unreadable_feature:{type(exc).__name__}:{exc}"
    return reason


def build_examples(manifest_path: Path, feature_root: Path, runtime: dict[str, Any]) -> tuple[list[Example], list[dict[str, str]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    dataframe = runtime["pd"].read_csv(manifest_path)
    required = {"sample_id", "semantic_class"}
    missing = required - set(dataframe.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    dataframe = dataframe[dataframe["semantic_class"] == "REAL"].sort_values("sample_id")
    examples: list[Example] = []
    excluded: list[dict[str, str]] = []
    for row in dataframe.itertuples(index=False):
        sample_id = str(row.sample_id)
        source_examples = []
        source_reasons = []
        for generation, label in ((1, 0), (2, 1)):
            path = pair_path(feature_root, sample_id, generation)
            reason = "MISSING_FEATURE_FILE" if not path.exists() else inspect_pair(path, runtime["np"], sample_id, label)
            if reason is not None:
                source_reasons.append(f"generation{generation}:{reason}")
            source_examples.append(Example(sample_id, path, label))
        if source_reasons:
            excluded.append({"sample_id": sample_id, "reason": ";".join(source_reasons)})
        else:
            examples.extend(source_examples)
    return examples, excluded


def load_feature(example: Example, np: Any) -> Any:
    with np.load(example.feature_path, allow_pickle=False) as payload:
        features, reason = pair_feature(payload, np, example.source_sample_id, example.label)
    if features is None:
        raise RuntimeError(f"Feature changed after preflight: {example.source_sample_id}: {reason}")
    return features


def train_normalizer(examples: list[Example], np: Any) -> tuple[Any, Any]:
    matrix = np.stack([load_feature(example, np) for example in examples]).astype(np.float64)
    mean = matrix.mean(axis=0)
    std = np.maximum(matrix.std(axis=0), 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def build_components(torch: Any) -> tuple[Any, Any]:
    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list[Example], np: Any, mean: Any, std: Any):
            self.examples, self.np, self.mean, self.std = examples, np, mean, std

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            example = self.examples[index]
            normalized = (load_feature(example, self.np) - self.mean) / self.std
            return torch.from_numpy(normalized), torch.tensor(example.label, dtype=torch.float32)

    class CompressionClassifier(torch.nn.Module):
        def __init__(self, hidden_dim: int, dropout: float):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(FEATURE_DIM, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(max(8, hidden_dim // 2), 1),
            )

        def forward(self, features: Any) -> Any:
            return self.network(features).squeeze(-1)

    return PairDataset, CompressionClassifier


def evaluate(model: Any, loader: Any, device: str, runtime: dict[str, Any]) -> dict[str, float]:
    torch, np = runtime["torch"], runtime["np"]
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for features, batch_labels in loader:
            logits.append(model(features.to(device)).cpu())
            labels.append(batch_labels)
    logits_np = torch.cat(logits).numpy()
    labels_np = torch.cat(labels).numpy().astype(int)
    probabilities = 1.0 / (1.0 + np.exp(-logits_np))
    fpr, tpr, thresholds = runtime["roc_curve"](labels_np, probabilities)
    balanced_accuracy = (tpr + 1.0 - fpr) / 2.0
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
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1:
        raise ValueError("epochs, patience, batch-size, and hidden-dim must be positive")
    runtime = require_runtime()
    torch = runtime["torch"]
    device = resolve_device(torch, args.device)
    set_seed(args.seed, torch)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_examples, train_excluded = build_examples(args.train_manifest, args.feature_root, runtime)
    validation_examples, validation_excluded = build_examples(args.validation_manifest, args.feature_root, runtime)
    if not train_examples or not validation_examples:
        raise RuntimeError("Generate controlled compression pairs for both train and validation first.")
    write_jsonl(args.log_dir / f"{run_id}_train_excluded.jsonl", train_excluded)
    write_jsonl(args.log_dir / f"{run_id}_validation_excluded.jsonl", validation_excluded)
    mean, std = train_normalizer(train_examples, runtime["np"])
    PairDataset, CompressionClassifier = build_components(torch)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(PairDataset(train_examples, runtime["np"], mean, std), batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_loader = torch.utils.data.DataLoader(PairDataset(validation_examples, runtime["np"], mean, std), batch_size=args.batch_size, shuffle=False)
    model = CompressionClassifier(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, stale_epochs = float("-inf"), 0, 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.checkpoint_dir / f"compression_classifier_{run_id}.pt"
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        losses = []
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), labels.to(device))
            loss.backward()
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
                "model_config": {"feature_dim": FEATURE_DIM, "hidden_dim": args.hidden_dim, "dropout": args.dropout},
                "normalizer": {"mean": mean, "std": std},
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "pair_schema_version": PAIR_SCHEMA_VERSION,
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
        "train_sources": len(train_examples) // 2,
        "validation_sources": len(validation_examples) // 2,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_excluded_sources": len(train_excluded),
        "validation_excluded_sources": len(validation_excluded),
        "best_epoch": best_epoch,
        "best_roc_auc": best_auc,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / f"{run_id}_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
