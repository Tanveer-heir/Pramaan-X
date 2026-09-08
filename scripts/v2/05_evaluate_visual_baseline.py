"""Phase 3c: Evaluate the ACCEPTED visual temporal classifier on external data.

Loads the frozen accepted visual checkpoint and evaluates on external
video datasets (AV-Deepfake1M). Reports per-dataset binary metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "modern_challenge_manifest.csv"
VISUAL_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "modern_challenge" / "visual_embeddings" / "convnext_tiny_v1"
ACCEPTED_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "visual_temporal" / "visual_temporal_20260904T145626Z.pt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "v2_evaluations" / "visual_baseline_external.json"

FAC_FEATURE_ROOT = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
FAC_VALIDATION = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"


def load_visual_trainer():
    script = PROJECT_ROOT / "scripts" / "training" / "07_train_visual_temporal.py"
    spec = importlib.util.spec_from_file_location("visual_trainer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_visual(model: Any, examples: list[dict], feature_root: Path,
                    trainer: Any, torch: Any, np: Any, device: str,
                    dataset_name: str) -> dict[str, Any]:
    """Evaluate visual model. Binary: real face (0) vs. fake face (1)."""
    valid_items = []
    skipped = []

    for ex in examples:
        feat_path = feature_root / f"{ex['sample_id']}.npz"
        if not feat_path.is_file():
            skipped.append({"sample_id": ex["sample_id"], "reason": "missing_feature"})
            continue
        try:
            with np.load(feat_path, allow_pickle=False) as payload:
                sequence, reason = trainer.shared_frame_sequence(payload, np)
            if sequence is None:
                skipped.append({"sample_id": ex["sample_id"], "reason": reason or "invalid"})
                continue
            valid_items.append({
                "tensor": torch.from_numpy(sequence.copy()),
                "label": int(ex["visual_label"]),
                "sample_id": ex["sample_id"],
            })
        except Exception as exc:
            skipped.append({"sample_id": ex["sample_id"], "reason": str(exc)})

    if not valid_items:
        return {
            "dataset": dataset_name,
            "total_samples": len(examples),
            "valid_samples": 0,
            "skipped": len(skipped),
            "error": "no_valid_features",
        }

    model.eval()
    all_logits = []
    all_labels = []
    batch_size = 16

    for i in range(0, len(valid_items), batch_size):
        batch = valid_items[i:i + batch_size]
        sequences = [item["tensor"] for item in batch]
        labels = [item["label"] for item in batch]

        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        with torch.inference_mode():
            logits = model(padded.to(device), lengths.to(device))

        all_logits.extend(logits.cpu().numpy().tolist())
        all_labels.extend(labels)

    logits_arr = np.asarray(all_logits, dtype=np.float64)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-logits_arr))

    result: dict[str, Any] = {
        "dataset": dataset_name,
        "total_samples": len(examples),
        "valid_samples": len(valid_items),
        "skipped": len(skipped),
        "label_distribution": {
            "real_face_count": int((labels_arr == 0).sum()),
            "fake_face_count": int((labels_arr == 1).sum()),
        },
    }

    has_both = len(set(all_labels)) >= 2

    if has_both:
        # Use the audio trainer's binary_metrics (same math)
        audio_trainer_path = PROJECT_ROOT / "scripts" / "training" / "14_train_audio_classifier.py"
        audio_trainer = load_module(audio_trainer_path, "audio_for_metrics")
        metrics = audio_trainer.binary_metrics(labels_arr, probabilities, np)
        result["metrics"] = metrics

        predictions_05 = (probabilities >= 0.5).astype(int)
        tp = int(((predictions_05 == 1) & (labels_arr == 1)).sum())
        tn = int(((predictions_05 == 0) & (labels_arr == 0)).sum())
        fp = int(((predictions_05 == 1) & (labels_arr == 0)).sum())
        fn = int(((predictions_05 == 0) & (labels_arr == 1)).sum())
        result["threshold_05"] = {
            "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
    elif all(l == 1 for l in all_labels):
        result["detection_rate_at_05"] = float((probabilities >= 0.5).mean())
        result["mean_probability"] = float(probabilities.mean())
    else:
        result["false_positive_rate_at_05"] = float((probabilities >= 0.5).mean())
        result["mean_probability"] = float(probabilities.mean())

    return result


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--feature-root", type=Path, default=VISUAL_FEATURE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ACCEPTED_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="Device: cuda, mps, cpu, or auto (default: auto)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("PRAMAAN-X V2: Visual Branch Baseline Evaluation", flush=True)
    print("=" * 60, flush=True)

    import numpy as np
    import torch

    trainer = load_visual_trainer()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"  Device: {device}", flush=True)

    # Load accepted checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    _, _, VisualClassifier = trainer.build_components(torch)
    model = VisualClassifier(int(config["hidden_dim"]), float(config["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    print(f"  Model loaded: hidden_dim={config['hidden_dim']}", flush=True)

    all_results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": device,
        "evaluations": {},
    }

    # External samples (video only)
    if args.manifest.is_file():
        with args.manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        video_rows = [r for r in rows if r["modality"] == "audio_video"]
        print(f"\n  External video samples: {len(video_rows)}")

        # Group by dataset
        datasets: dict[str, list[dict]] = {}
        for row in video_rows:
            ds = row["dataset"]
            if ds not in datasets:
                datasets[ds] = []
            datasets[ds].append(row)

        for ds_name in sorted(datasets.keys()):
            ds_rows = datasets[ds_name]
            print(f"\n  Evaluating {ds_name} ({len(ds_rows)} video samples)...")
            result = evaluate_visual(model, ds_rows, args.feature_root,
                                      trainer, torch, np, device, ds_name)
            all_results["evaluations"][ds_name] = result
            print(f"    Valid: {result['valid_samples']}, Skipped: {result['skipped']}")
            if "metrics" in result:
                m = result["metrics"]
                print(f"    AUC: {m['roc_auc']:.4f}, BA: {m['balanced_accuracy']:.4f}")
            elif "detection_rate_at_05" in result:
                print(f"    Detection rate @0.5: {result['detection_rate_at_05']:.4f}")
    else:
        print(f"  ⚠ External manifest not found: {args.manifest}")

    # FakeAVCeleb validation reference
    if FAC_VALIDATION.is_file() and FAC_FEATURE_ROOT.is_dir():
        print(f"\n  Evaluating FakeAVCeleb validation (reference)...")
        with FAC_VALIDATION.open(newline="", encoding="utf-8") as f:
            fac_rows = list(csv.DictReader(f))
        result = evaluate_visual(model, fac_rows, FAC_FEATURE_ROOT,
                                  trainer, torch, np, device, "FakeAVCeleb_validation")
        all_results["evaluations"]["FakeAVCeleb_validation"] = result
        if "metrics" in result:
            m = result["metrics"]
            print(f"    AUC: {m['roc_auc']:.4f}, BA: {m['balanced_accuracy']:.4f}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, sort_keys=True, default=str)
    print(f"\n  Results saved to: {args.output}")

    # Summary
    print(f"\n{'=' * 60}")
    print("VISUAL BASELINE SUMMARY")
    print(f"{'=' * 60}")
    for name, eval_result in all_results["evaluations"].items():
        valid = eval_result.get("valid_samples", 0)
        if "metrics" in eval_result:
            m = eval_result["metrics"]
            print(f"  {name:30s}  n={valid:4d}  AUC={m['roc_auc']:.3f}  BA={m['balanced_accuracy']:.3f}")
        elif "detection_rate_at_05" in eval_result:
            print(f"  {name:30s}  n={valid:4d}  DetRate@0.5={eval_result['detection_rate_at_05']:.3f}")
        elif "false_positive_rate_at_05" in eval_result:
            print(f"  {name:30s}  n={valid:4d}  FPR@0.5={eval_result['false_positive_rate_at_05']:.3f}")
        else:
            print(f"  {name:30s}  n={valid:4d}  (no metrics)")


if __name__ == "__main__":
    main()
