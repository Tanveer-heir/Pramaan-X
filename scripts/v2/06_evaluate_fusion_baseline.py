"""Phase 3d: Evaluate the ACCEPTED fusion model on external AV data.

Uses ONLY complete audio_video samples (AV-Deepfake1M) for four-class fusion
evaluation. Does NOT mix audio-only samples into fusion testing.

Also re-evaluates on FakeAVCeleb validation for reference.
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
FUSION_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "fusion" / "pramaan_x_hackathon_final.pth"

# External feature roots
EXT_VISUAL = PROJECT_ROOT / "data" / "interim" / "modern_challenge" / "visual_embeddings" / "convnext_tiny_v1"
EXT_AUDIO = PROJECT_ROOT / "data" / "interim" / "modern_challenge" / "audio_embeddings" / "xlsr_sls_v1"
EXT_DENSE = PROJECT_ROOT / "data" / "interim" / "modern_challenge" / "av_sync_features" / "dense_v1"

# FakeAVCeleb feature roots
FAC_VISUAL = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
FAC_AUDIO = PROJECT_ROOT / "data" / "interim" / "audio_embeddings" / "xlsr_sls_v1"
FAC_DENSE = PROJECT_ROOT / "data" / "interim" / "av_sync_features" / "dense_v1"
FAC_VALIDATION = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "v2_evaluations" / "fusion_baseline_external.json"

CLASS_NAMES = ("REAL", "VISUAL_MANIPULATION", "AUDIO_MANIPULATION", "AUDIO_VISUAL_MANIPULATION")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def load_fusion_core():
    path = PROJECT_ROOT / "scripts" / "fusion" / "pramaan_x_fusion.py"
    spec = importlib.util.spec_from_file_location("pramaan_x_fusion", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_four_class_metrics(labels: list[int], predicted: list[int]) -> dict[str, Any]:
    """Compute four-class accuracy, macro-F1, per-class, confusion matrix."""
    import numpy as np
    labels_arr = np.asarray(labels, dtype=int)
    pred_arr = np.asarray(predicted, dtype=int)

    matrix = np.zeros((4, 4), dtype=int)
    for a, p in zip(labels_arr, pred_arr):
        matrix[a, p] += 1

    per_class: dict[str, Any] = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(matrix[i, :].sum()),
        }

    accuracy = float(matrix.diagonal().sum() / max(1, matrix.sum()))
    macro_f1 = float(sum(pc["f1"] for pc in per_class.values()) / 4)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "total_samples": int(matrix.sum()),
    }


def evaluate_fusion(samples: list[dict], runner: Any, fusion_core: Any,
                    visual_root: Path, audio_root: Path, dense_root: Path,
                    fusion_payload: dict, torch: Any, np: Any,
                    device: str, dataset_name: str) -> dict[str, Any]:
    """Run fusion inference on complete AV samples."""
    enabled = fusion_payload["enabled_branches"]
    order = fusion_payload["input_feature_order"]
    norm_mean = fusion_payload["normalization"]["mean"].numpy()
    norm_std = fusion_payload["normalization"]["std"].numpy()

    # Build fusion MLP
    arch = fusion_payload["architecture"]

    class Fusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(arch["input_dim"]),
                torch.nn.Linear(arch["input_dim"], arch["hidden_dim"]),
                torch.nn.GELU(),
                torch.nn.Dropout(arch["dropout"]),
                torch.nn.Linear(arch["hidden_dim"], 4),
            )

        def forward(self, x):
            return self.net(x)

    fusion_model = Fusion()
    fusion_model.load_state_dict(fusion_payload["state_dict"], strict=True)
    fusion_model = fusion_model.to(device).eval()

    all_labels = []
    all_predicted = []
    skipped = []
    evidence_log = []

    for sample in samples:
        sid = sample["sample_id"]
        sem_class = sample["semantic_class"]

        if sem_class not in CLASS_TO_INDEX:
            skipped.append({"sample_id": sid, "reason": f"unknown_class:{sem_class}"})
            continue

        target = CLASS_TO_INDEX[sem_class]

        # Score evidence through branches
        paths = {
            "visual": visual_root / f"{sid}.npz" if (visual_root / f"{sid}.npz").is_file() else None,
            "audio": audio_root / f"{sid}.npz" if (audio_root / f"{sid}.npz").is_file() else None,
            "dense_av": dense_root / f"{sid}.npz" if (dense_root / f"{sid}.npz").is_file() else None,
        }

        try:
            scored = fusion_core.score_evidence(runner, paths, enabled)
        except RuntimeError as exc:
            skipped.append({"sample_id": sid, "reason": str(exc)})
            continue

        # Vectorize and normalize
        vector = np.asarray([fusion_core.vectorize(scored, order)], dtype=np.float32)
        logit_indices = [i for i, name in enumerate(order) if name.endswith("_logit")]
        if logit_indices:
            vector[:, logit_indices] = (vector[:, logit_indices] - norm_mean[logit_indices]) / norm_std[logit_indices]

        # Predict
        with torch.inference_mode():
            logits = fusion_model(torch.from_numpy(vector).to(device))
            pred = int(logits.argmax(1).item())

        all_labels.append(target)
        all_predicted.append(pred)

        evidence_log.append({
            "sample_id": sid,
            "target": target,
            "predicted": pred,
            "correct": target == pred,
            "visual_available": scored.get("visual_available", False),
            "audio_available": scored.get("audio_available", False),
            "av_available": scored.get("av_available", False),
        })

    result: dict[str, Any] = {
        "dataset": dataset_name,
        "total_samples": len(samples),
        "evaluated_samples": len(all_labels),
        "skipped": len(skipped),
    }

    if all_labels:
        metrics = compute_four_class_metrics(all_labels, all_predicted)
        result["metrics"] = metrics

        # Modality coverage
        vis_avail = sum(1 for e in evidence_log if e["visual_available"])
        aud_avail = sum(1 for e in evidence_log if e["audio_available"])
        av_avail = sum(1 for e in evidence_log if e["av_available"])
        result["modality_coverage"] = {
            "visual_available": vis_avail,
            "audio_available": aud_avail,
            "av_available": av_avail,
            "total": len(evidence_log),
        }

    if skipped:
        result["skipped_examples"] = skipped[:20]

    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--fusion-checkpoint", type=Path, default=FUSION_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="Device: cuda, mps, cpu, or auto (default: auto)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("PRAMAAN-X V2: Fusion Baseline Evaluation", flush=True)
    print("=" * 60, flush=True)

    import numpy as np
    import torch

    fusion_core = load_fusion_core()
    device = fusion_core.resolve_device(torch, args.device)
    print(f"  Device: {device}", flush=True)

    # Load fusion checkpoint
    fusion_payload = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=True)
    enabled = fusion_payload["enabled_branches"]
    print(f"  Enabled branches: {enabled}", flush=True)
    print(f"  Feature order: {fusion_payload['input_feature_order']}", flush=True)

    # Create branch runner with accepted checkpoints
    default_branches = {
        "visual": PROJECT_ROOT / "checkpoints" / "visual_temporal" / "visual_temporal_20260904T145626Z.pt",
        "audio": PROJECT_ROOT / "checkpoints" / "audio_classifier" / "audio_classifier_20260904T181732Z.pt",
        "dense_av": PROJECT_ROOT / "checkpoints" / "av_sync_dense" / "av_sync_dense_20260905T055537Z.pt",
    }
    branch_paths = {}
    for branch in enabled:
        resolved = None
        if branch in fusion_payload.get("accepted_branch_checkpoints", {}):
            old_path_str = fusion_payload["accepted_branch_checkpoints"][branch]["path"]
            fname = Path(old_path_str.replace("\\", "/")).name
            candidates = list((PROJECT_ROOT / "checkpoints").rglob(fname))
            if candidates:
                resolved = candidates[0]
        if not resolved or not resolved.is_file():
            resolved = default_branches.get(branch)
        branch_paths[branch] = resolved

    runner = fusion_core.BranchRunner.create(PROJECT_ROOT, branch_paths, enabled, device)

    all_results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fusion_checkpoint": str(args.fusion_checkpoint),
        "enabled_branches": enabled,
        "device": device,
        "evaluations": {},
    }

    # 1. External data (audio_video samples only)
    if args.manifest.is_file():
        with args.manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        av_rows = [r for r in rows if r["modality"] == "audio_video"]
        print(f"\n  External audio_video samples: {len(av_rows)}")

        datasets: dict[str, list[dict]] = {}
        for row in av_rows:
            ds = row["dataset"]
            if ds not in datasets:
                datasets[ds] = []
            datasets[ds].append(row)

        for ds_name in sorted(datasets.keys()):
            ds_rows = datasets[ds_name]
            print(f"\n  Evaluating {ds_name} ({len(ds_rows)} samples)...")
            result = evaluate_fusion(
                ds_rows, runner, fusion_core,
                EXT_VISUAL, EXT_AUDIO, EXT_DENSE,
                fusion_payload, torch, np, device, ds_name,
            )
            all_results["evaluations"][ds_name] = result
            if "metrics" in result:
                m = result["metrics"]
                print(f"    Accuracy: {m['accuracy']:.4f}, Macro-F1: {m['macro_f1']:.4f}")
                for cls_name, cls_metrics in m["per_class"].items():
                    print(f"      {cls_name:30s}  P={cls_metrics['precision']:.3f}  R={cls_metrics['recall']:.3f}  F1={cls_metrics['f1']:.3f}  n={cls_metrics['support']}")
    else:
        print(f"  ⚠ External manifest not found")

    # 2. FakeAVCeleb validation reference
    if FAC_VALIDATION.is_file():
        print(f"\n  Evaluating FakeAVCeleb validation (reference)...")
        with FAC_VALIDATION.open(newline="", encoding="utf-8") as f:
            fac_rows = list(csv.DictReader(f))

        result = evaluate_fusion(
            fac_rows, runner, fusion_core,
            FAC_VISUAL, FAC_AUDIO, FAC_DENSE,
            fusion_payload, torch, np, device, "FakeAVCeleb_validation",
        )
        all_results["evaluations"]["FakeAVCeleb_validation"] = result
        if "metrics" in result:
            m = result["metrics"]
            print(f"    Accuracy: {m['accuracy']:.4f}, Macro-F1: {m['macro_f1']:.4f}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, sort_keys=True, default=str)
    print(f"\n  Results saved to: {args.output}")

    # Summary
    print(f"\n{'=' * 60}")
    print("FUSION BASELINE SUMMARY")
    print(f"{'=' * 60}")
    for name, eval_result in all_results["evaluations"].items():
        evaluated = eval_result.get("evaluated_samples", 0)
        if "metrics" in eval_result:
            m = eval_result["metrics"]
            print(f"  {name:30s}  n={evaluated:4d}  Acc={m['accuracy']:.3f}  F1={m['macro_f1']:.3f}")
        else:
            print(f"  {name:30s}  n={evaluated:4d}  (no metrics)")


if __name__ == "__main__":
    main()
