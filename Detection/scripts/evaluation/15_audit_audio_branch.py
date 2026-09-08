"""Audit the near-perfect XLSR-SLS audio validation result before fusion."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "audio_embeddings" / "xlsr_sls_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "audio_classifier"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "audits"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "14_train_audio_classifier.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser


def load_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("audio_training_for_audit", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "audio_label", "semantic_class", "identity", "content_id"}
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"{path} must contain {sorted(required)}")
    return rows


def feature_fingerprint(payload: Any, np: Any) -> str:
    digest = hashlib.sha256()
    for key in ("source_audio_samples", "window_start_samples", "window_valid_samples", "embeddings"):
        array = np.ascontiguousarray(payload[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def sigmoid(value: float, np: Any) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0))))


def load_feature_record(row: dict[str, str], feature_root: Path, np: Any, training: Any) -> dict[str, Any]:
    path = feature_root / f"{row['sample_id']}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    with np.load(path, allow_pickle=False) as payload:
        sequence, reason = training.load_sequence(payload, np)
        if sequence is None:
            raise RuntimeError(f"Unusable feature {row['sample_id']}: {reason}")
        log_probabilities = np.asarray(payload["pretrained_log_probabilities"], dtype=np.float64)
        if log_probabilities.shape != (len(sequence), 2):
            raise RuntimeError(f"Invalid pretrained scores for {row['sample_id']}")
        # Official output index 0 is spoof and index 1 is bona fide. Averaging
        # log-odds preserves the released classifier's direction per window.
        baseline_probability = sigmoid(float(np.mean(log_probabilities[:, 0] - log_probabilities[:, 1])), np)
        fingerprint = feature_fingerprint(payload, np)
    return {
        "sample_id": row["sample_id"], "label": int(row["audio_label"]),
        "semantic_class": row["semantic_class"], "method_hint": row.get("method_hint", "unknown") or "unknown",
        "identity": row["identity"], "content_id": row["content_id"],
        "content_key": f"{row['identity']}::{row['content_id']}",
        "feature_fingerprint": fingerprint, "pretrained_probability": baseline_probability,
    }


def overlap_examples(left: list[dict[str, Any]], right: list[dict[str, Any]], key: str, limit: int = 100) -> tuple[int, list[dict[str, str]]]:
    left_index: dict[str, list[str]] = {}
    right_index: dict[str, list[str]] = {}
    for record in left:
        left_index.setdefault(str(record[key]), []).append(record["sample_id"])
    for record in right:
        right_index.setdefault(str(record[key]), []).append(record["sample_id"])
    shared = sorted(set(left_index) & set(right_index))
    examples = [
        {"key": value, "train_sample_id": left_index[value][0], "validation_sample_id": right_index[value][0]}
        for value in shared[:limit]
    ]
    return len(shared), examples


def separation_summary(labels: Any, probabilities: Any, np: Any) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    maximum_real = float(probabilities[labels == 0].max())
    minimum_fake = float(probabilities[labels == 1].min())
    return {
        "maximum_real_probability": maximum_real,
        "minimum_fake_probability": minimum_fake,
        "separation_margin": minimum_fake - maximum_real,
    }


def group_summary(records: list[dict[str, Any]], probability_key: str, threshold: float, np: Any) -> dict[str, Any]:
    result = {}
    groups = sorted({record["semantic_class"] for record in records})
    for group in groups:
        members = [record for record in records if record["semantic_class"] == group]
        probabilities = np.asarray([record[probability_key] for record in members], dtype=np.float64)
        labels = np.asarray([record["label"] for record in members], dtype=np.int64)
        predictions = (probabilities >= threshold).astype(np.int64)
        result[group] = {
            "count": len(members), "audio_label": int(labels[0]),
            "mean_probability": float(probabilities.mean()),
            "minimum_probability": float(probabilities.min()),
            "maximum_probability": float(probabilities.max()),
            "errors_at_threshold": int((predictions != labels).sum()),
        }
    return result


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    candidates = sorted(args.checkpoint_dir.glob("audio_classifier_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No audio checkpoint found in {args.checkpoint_dir}")
    return candidates[-1]


def learned_probabilities(checkpoint_path: Path, validation_rows: list[dict[str, str]], args: argparse.Namespace, runtime: dict[str, Any], training: Any) -> tuple[dict[str, float], dict[str, Any]]:
    torch, np = runtime["torch"], runtime["np"]
    device = training.resolve_device(torch, args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("train_manifest_sha256") != training.manifest_hash(args.train_manifest):
        raise RuntimeError("Checkpoint train manifest hash does not match current manifest")
    if checkpoint.get("validation_manifest_sha256") != training.manifest_hash(args.validation_manifest):
        raise RuntimeError("Checkpoint validation manifest hash does not match current manifest")
    examples, excluded = training.build_examples(args.validation_manifest, args.feature_root, None, np)
    if excluded or len(examples) != len(validation_rows):
        raise RuntimeError(f"Validation feature coverage changed: {len(excluded)} exclusions")
    Dataset, collate, AudioClassifier = training.build_components(torch)
    dataset = Dataset(examples, np)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    config = checkpoint["model_config"]
    model = AudioClassifier(int(config["hidden_dim"]), float(config["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    predictions: dict[str, float] = {}
    with torch.inference_mode():
        for padded, lengths, _, sample_ids in loader:
            logits = model(padded.to(device), lengths.to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions.update({sample_id: float(probability) for sample_id, probability in zip(sample_ids, probabilities, strict=True)})
    return predictions, checkpoint


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    training = load_training_module()
    runtime = training.require_runtime()
    np = runtime["np"]
    train_rows, validation_rows = load_manifest(args.train_manifest), load_manifest(args.validation_manifest)
    train_records = [load_feature_record(row, args.feature_root, np, training) for row in train_rows]
    validation_records = [load_feature_record(row, args.feature_root, np, training) for row in validation_rows]
    sample_overlap, sample_examples = overlap_examples(train_records, validation_records, "sample_id")
    feature_overlap, feature_examples = overlap_examples(train_records, validation_records, "feature_fingerprint")
    content_overlap, content_examples = overlap_examples(train_records, validation_records, "content_key")
    checkpoint_path = resolve_checkpoint(args)
    predictions, checkpoint = learned_probabilities(checkpoint_path, validation_rows, args, runtime, training)
    for record in validation_records:
        record["learned_probability"] = predictions[record["sample_id"]]
    labels = np.asarray([record["label"] for record in validation_records], dtype=np.int64)
    baseline = np.asarray([record["pretrained_probability"] for record in validation_records])
    learned = np.asarray([record["learned_probability"] for record in validation_records])
    baseline_metrics = training.binary_metrics(labels, baseline, np)
    learned_metrics = training.binary_metrics(labels, learned, np)
    saved_metrics = checkpoint["best_validation_metrics"]
    metric_delta = {key: abs(float(learned_metrics[key]) - float(saved_metrics[key])) for key in ("roc_auc", "average_precision", "balanced_accuracy")}
    if any(value > 1e-6 for value in metric_delta.values()):
        raise RuntimeError(f"Recomputed checkpoint metrics differ from saved metrics: {metric_delta}")
    report = {
        "status": "PASS" if sample_overlap == feature_overlap == content_overlap == 0 else "FAIL_LEAKAGE_AUDIT",
        "train_examples": len(train_records), "validation_examples": len(validation_records),
        "checkpoint_path": str(checkpoint_path),
        "sample_id_overlap": sample_overlap, "sample_id_overlap_examples": sample_examples,
        "exact_feature_fingerprint_overlap": feature_overlap, "feature_overlap_examples": feature_examples,
        "identity_content_key_overlap": content_overlap, "content_overlap_examples": content_examples,
        "pretrained_baseline_metrics": baseline_metrics,
        "pretrained_baseline_separation": separation_summary(labels, baseline, np),
        "learned_head_metrics": learned_metrics,
        "learned_head_separation": separation_summary(labels, learned, np),
        "learned_head_semantic_breakdown": group_summary(validation_records, "learned_probability", learned_metrics["validation_threshold"], np),
        "saved_metric_absolute_deltas": metric_delta,
        "warning": "In-dataset validation does not establish cross-dataset generalization.",
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"audio_branch_{run_id}.json"
    training.atomic_json_save(report, output_path)
    report["report_path"] = str(output_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
