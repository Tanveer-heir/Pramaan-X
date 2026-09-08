"""Audit dense AV synchronization before admitting it to evidence fusion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "19_train_dense_av_sync.py"
DEFAULT_DENSE_ROOT = PROJECT_ROOT / "data" / "interim" / "av_sync_features" / "dense_v1"
DEFAULT_VISUAL_ROOT = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "av_sync_dense"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "audits"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def load_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("dense_av_training_for_audit", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    candidates = sorted(args.checkpoint_dir.glob("av_sync_dense_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No dense AV checkpoint found in {args.checkpoint_dir}")
    return candidates[-1]


def source_fingerprint(source: Any, training: Any, np: Any) -> str:
    audio, mouth, timestamps = training.load_arrays(source, np)
    digest = hashlib.sha256()
    for name, value in (("audio", audio), ("mouth", mouth), ("timestamps", timestamps)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def overlap(left: list[Any], right: list[Any], key: Any) -> tuple[int, list[str]]:
    shared = sorted({key(item) for item in left} & {key(item) for item in right})
    return len(shared), shared[:20]


def score_pairs(model: Any, dataset: Any, batch_size: int, device: str, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, np = runtime["torch"], runtime["np"]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    logits, labels, offsets, sample_ids = [], [], [], []
    with torch.inference_mode():
        for audio, mouth, mouth_delta, batch_labels, batch_offsets, batch_ids in loader:
            logits.append(model(audio.to(device), mouth.to(device), mouth_delta.to(device)).cpu())
            labels.append(batch_labels)
            offsets.append(batch_offsets)
            sample_ids.extend(batch_ids)
    logits_array = torch.cat(logits).numpy()
    return {
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "probabilities": 1.0 / (1.0 + np.exp(-np.clip(logits_array, -80.0, 80.0))),
        "offsets": torch.cat(offsets).numpy().astype(np.int64),
        "sample_ids": np.asarray(sample_ids),
    }


def offset_breakdown(scored: dict[str, Any], training: Any, np: Any) -> dict[str, Any]:
    labels, probabilities, offsets = scored["labels"], scored["probabilities"], scored["offsets"]
    result = {}
    for magnitude in training.DEFAULT_SHIFT_FRAMES:
        negative_indices = np.flatnonzero(np.abs(offsets) == magnitude)
        positive_indices = negative_indices - 1
        indices = np.ravel(np.column_stack((positive_indices, negative_indices)))
        if len(negative_indices):
            result[f"{magnitude * 250}ms"] = {
                **training.binary_metrics(labels[indices], probabilities[indices], np),
                "negative_pairs": int(len(negative_indices)),
                "mean_aligned_probability": float(probabilities[positive_indices].mean()),
                "mean_shifted_probability": float(probabilities[negative_indices].mean()),
            }
    for direction, selector in (("negative", offsets < 0), ("positive", offsets > 0)):
        negative_indices = np.flatnonzero(selector)
        positive_indices = negative_indices - 1
        indices = np.ravel(np.column_stack((positive_indices, negative_indices)))
        if len(negative_indices):
            result[f"direction_{direction}"] = {
                **training.binary_metrics(labels[indices], probabilities[indices], np),
                "negative_pairs": int(len(negative_indices)),
            }
    return result


def identity_bootstrap_ci(scored: dict[str, Any], training: Any, np: Any, repetitions: int, seed: int) -> dict[str, float]:
    identities = np.unique(scored["sample_ids"])
    index = {sample_id: np.flatnonzero(scored["sample_ids"] == sample_id) for sample_id in identities}
    rng, values = np.random.default_rng(seed), []
    for _ in range(repetitions):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        positions = np.concatenate([index[sample_id] for sample_id in sampled])
        values.append(training.binary_metrics(scored["labels"][positions], scored["probabilities"][positions], np)["roc_auc"])
    return {
        "repetitions": repetitions,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.bootstrap_repetitions < 100:
        raise ValueError("Batch size must be positive and bootstrap repetitions must be at least 100")
    training = load_training_module()
    runtime = training.require_runtime()
    torch, np = runtime["torch"], runtime["np"]
    device = training.resolve_device(torch, args.device)
    checkpoint_path = resolve_checkpoint(args)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint is not safe-load compatible. Pull the audit update and rerun dense AV training once before auditing."
        ) from exc
    if checkpoint.get("train_manifest_sha256") != training.manifest_hash(args.train_manifest):
        raise RuntimeError("Checkpoint train manifest hash does not match")
    if checkpoint.get("validation_manifest_sha256") != training.manifest_hash(args.validation_manifest):
        raise RuntimeError("Checkpoint validation manifest hash does not match")
    if checkpoint.get("pairing_strategy") != training.PAIRING_STRATEGY:
        raise RuntimeError("Checkpoint pairing strategy does not match")
    config = checkpoint["model_config"]
    segment_frames = int(config["segment_frames"])
    minimum_frames = segment_frames + min(training.DEFAULT_SHIFT_FRAMES)
    train_sources, train_preflight = training.build_sources(
        args.train_manifest, args.dense_root, args.visual_root, None, minimum_frames, np,
    )
    validation_sources, validation_preflight = training.build_sources(
        args.validation_manifest, args.dense_root, args.visual_root, None, minimum_frames, np,
    )
    stride_frames = int(config["stride_frames"])
    train_pairs, train_segment_excluded = training.build_all_pairs(train_sources, np, segment_frames, stride_frames, int(checkpoint["seed"]))
    validation_pairs, validation_segment_excluded = training.build_all_pairs(validation_sources, np, segment_frames, stride_frames, int(checkpoint["seed"]))
    if train_preflight or validation_preflight or train_segment_excluded or validation_segment_excluded:
        exclusions = {
            "train": train_preflight + train_segment_excluded,
            "validation": validation_preflight + validation_segment_excluded,
        }
    else:
        exclusions = {"train": [], "validation": []}
    train_sample_overlap, train_sample_examples = overlap(train_sources, validation_sources, lambda source: source.sample_id)
    train_fingerprints = [(source.sample_id, source_fingerprint(source, training, np)) for source in train_sources]
    validation_fingerprints = [(source.sample_id, source_fingerprint(source, training, np)) for source in validation_sources]
    fingerprint_overlap, fingerprint_examples = overlap(train_fingerprints, validation_fingerprints, lambda item: item[1])
    Dataset, Model = training.build_components(torch)
    mean = checkpoint["audio_normalizer_mean"].cpu().numpy()
    std = checkpoint["audio_normalizer_std"].cpu().numpy()
    model = Model(int(config["projection_dim"]), int(config["hidden_dim"]), float(config["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    train_scored = score_pairs(model, Dataset(train_pairs, mean, std, np), args.batch_size, device, runtime)
    validation_scored = score_pairs(model, Dataset(validation_pairs, mean, std, np), args.batch_size, device, runtime)
    train_metrics = training.binary_metrics(train_scored["labels"], train_scored["probabilities"], np)
    validation_metrics = training.binary_metrics(validation_scored["labels"], validation_scored["probabilities"], np)
    saved = checkpoint["best_validation_metrics"]
    deltas = {key: abs(float(validation_metrics[key]) - float(saved[key])) for key in ("roc_auc", "average_precision", "balanced_accuracy")}
    breakdown = offset_breakdown(validation_scored, training, np)
    aligned_mean = float(validation_scored["probabilities"][validation_scored["labels"] == 1].mean())
    shifted_mean = float(validation_scored["probabilities"][validation_scored["labels"] == 0].mean())
    checks = {
        "sample_overlap_zero": train_sample_overlap == 0,
        "feature_fingerprint_overlap_zero": fingerprint_overlap == 0,
        "saved_metrics_reproduced": all(value <= 1e-6 for value in deltas.values()),
        "validation_roc_above_gate": validation_metrics["roc_auc"] >= 0.65,
        "aligned_scores_higher": aligned_mean > shifted_mean,
    }
    report = {
        "status": "PASS_CONTROLLED_SHIFT" if all(checks.values()) else "FAIL",
        "checkpoint_path": str(checkpoint_path), "checks": checks,
        "train_sources": len(train_sources), "validation_sources": len(validation_sources),
        "train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs),
        "exclusions": exclusions, "sample_id_overlap": train_sample_overlap,
        "sample_overlap_examples": train_sample_examples,
        "exact_source_feature_fingerprint_overlap": fingerprint_overlap,
        "feature_fingerprint_overlap_examples": fingerprint_examples,
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
        "saved_metric_absolute_deltas": deltas,
        "generalization_gap_roc_auc": train_metrics["roc_auc"] - validation_metrics["roc_auc"],
        "mean_aligned_probability": aligned_mean, "mean_shifted_probability": shifted_mean,
        "offset_breakdown": breakdown,
        "identity_bootstrap_roc_auc_95_ci": identity_bootstrap_ci(validation_scored, training, np, args.bootstrap_repetitions, args.seed),
        "warning": "This validates controlled temporal-shift detection on FakeAVCeleb REAL videos, not real deepfake lip-sync generalization.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"dense_av_sync_{run_id}.json"
    training.atomic_json_save(report, output_path)
    report["report_path"] = str(output_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
