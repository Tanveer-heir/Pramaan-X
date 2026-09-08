"""Run cache-based inference with a Pramaan-X fusion checkpoint.

Raw video extraction is intentionally outside this deadline-safe entry point.
Supply a sample id and cache roots, or explicit feature paths.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "scripts" / "fusion" / "pramaan_x_fusion.py"
STANDARD_BRANCH_CHECKPOINTS = {
    "visual": Path("checkpoints/visual_temporal/visual_temporal_20260904T145626Z.pt"),
    "audio": Path("checkpoints/audio_classifier/audio_classifier_20260904T181732Z.pt"),
    "dense_av": Path("checkpoints/av_sync_dense/av_sync_dense_20260905T055537Z.pt"),
}


def core() -> Any:
    spec = importlib.util.spec_from_file_location("pramaan_x_fusion_core", CORE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {CORE_PATH}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fusion-checkpoint", type=Path, required=True)
    p.add_argument("--sample-id")
    p.add_argument("--visual-feature", type=Path); p.add_argument("--audio-feature", type=Path); p.add_argument("--dense-feature", type=Path)
    p.add_argument("--visual-root", type=Path, default=ROOT / "data/interim/visual_embeddings/convnext_tiny_v1")
    p.add_argument("--audio-root", type=Path, default=ROOT / "data/interim/audio_embeddings/xlsr_sls_v1")
    p.add_argument("--dense-root", type=Path, default=ROOT / "data/interim/av_sync_features/dense_v1")
    p.add_argument("--visual-checkpoint", type=Path); p.add_argument("--audio-checkpoint", type=Path); p.add_argument("--av-checkpoint", type=Path)
    p.add_argument("--device", default="auto"); p.add_argument("--output", type=Path, required=True)
    return p


def atomic_json(path: Path, value: dict[str, Any], c: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as h:
        temp = Path(h.name); json.dump(c.json_safe(value), h, indent=2, sort_keys=True, allow_nan=False); h.write("\n"); h.flush(); os.fsync(h.fileno())
    os.replace(temp, path)


def resolve_branch_checkpoint(name: str, override: Path | None, saved: dict[str, Any], project_root: Path = ROOT) -> Path:
    if override is not None:
        return override
    repository_path = project_root / STANDARD_BRANCH_CHECKPOINTS[name]
    if repository_path.is_file():
        return repository_path
    recorded_path = Path(saved[name]["path"])
    if recorded_path.is_file():
        return recorded_path
    # Return the portable expected location so the eventual error tells a fresh
    # clone exactly where the required artifact belongs.
    return repository_path


def main() -> None:
    args = parser().parse_args(); c = core()
    try:
        import numpy as np
        import torch
    except ImportError as exc: raise RuntimeError("NumPy and PyTorch are required") from exc
    checkpoint = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != c.FORMAT_VERSION: raise RuntimeError("Unsupported fusion checkpoint format")
    enabled, order = checkpoint["enabled_branches"], checkpoint["input_feature_order"]
    saved = checkpoint["accepted_branch_checkpoints"]
    paths = {
        "visual": args.visual_feature or (args.visual_root / f"{args.sample_id}.npz" if args.sample_id else None),
        "audio": args.audio_feature or (args.audio_root / f"{args.sample_id}.npz" if args.sample_id else None),
        "dense_av": args.dense_feature or (args.dense_root / f"{args.sample_id}.npz" if args.sample_id else None),
    }
    if not args.sample_id and not any(paths.values()): raise ValueError("Provide --sample-id or at least one explicit feature path")
    override = {"visual": args.visual_checkpoint, "audio": args.audio_checkpoint, "dense_av": args.av_checkpoint}
    branch_paths = {name: resolve_branch_checkpoint(name, override[name], saved) for name in enabled}
    runner = c.BranchRunner.create(ROOT, branch_paths, enabled, args.device)
    evidence = c.score_evidence(runner, paths, enabled)
    vector = np.asarray([c.vectorize(evidence, order)], dtype=np.float32)
    normalization = checkpoint["normalization"]
    normalized, _, _ = c.normalize_features(vector, order, normalization["mean"].detach().cpu().numpy(), normalization["std"].detach().cpu().numpy())
    class Fusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__(); a = checkpoint["architecture"]
            self.net = torch.nn.Sequential(torch.nn.LayerNorm(a["input_dim"]), torch.nn.Linear(a["input_dim"], a["hidden_dim"]), torch.nn.GELU(), torch.nn.Dropout(a["dropout"]), torch.nn.Linear(a["hidden_dim"], 4))
        def forward(self, x: Any) -> Any: return self.net(x)
    model = Fusion(); model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()

    def score(masked_evidence: dict[str, Any]) -> dict[str, Any]:
        masked_vector = np.asarray([c.vectorize(masked_evidence, order)], dtype=np.float32)
        masked_normalized, _, _ = c.normalize_features(
            masked_vector, order,
            normalization["mean"].detach().cpu().numpy(),
            normalization["std"].detach().cpu().numpy(),
        )
        with torch.inference_mode():
            masked_logits = model(torch.from_numpy(masked_normalized)).squeeze(0)
        masked_probabilities = torch.softmax(masked_logits, dim=0)
        selected_index = int(masked_logits.argmax().item())
        return {
            "selected_class": checkpoint["class_names"][selected_index],
            "class_logits": {name: float(value) for name, value in zip(checkpoint["class_names"], masked_logits.tolist(), strict=True)},
            "softmax_scores": {name: float(value) for name, value in zip(checkpoint["class_names"], masked_probabilities.tolist(), strict=True)},
            "probability_note": "Softmax values are uncalibrated class scores.",
        }

    classification = score(evidence)
    counterfactuals = c.counterfactual_modality_analysis(evidence, enabled, score)
    full_scores = counterfactuals["full_evidence"]["softmax_scores"]
    for intervention in counterfactuals["interventions"].values():
        intervention["winning_class_score_delta"] = (
            intervention["softmax_scores"][classification["selected_class"]]
            - full_scores[classification["selected_class"]]
        )
    temporal_evidence = evidence.pop("temporal_evidence")
    output = {
        "schema_version": "pramaan_x_prediction_v2",
        "sample_id": args.sample_id,
        "probability_note": classification["probability_note"],
        "class_names": checkpoint["class_names"],
        "class_logits": classification["class_logits"],
        "softmax_probabilities": classification["softmax_scores"],
        "selected_class": classification["selected_class"],
        "classification": classification,
        "branch_evidence": evidence,
        "counterfactual_modality_analysis": counterfactuals,
        "temporal_evidence": temporal_evidence,
        "compression": {"status": "NOT_AVAILABLE_OR_DEFERRED"},
        "checkpoint_provenance": {"fusion_checkpoint": str(args.fusion_checkpoint), "enabled_branches": enabled, "accepted_branch_checkpoints": saved},
    }
    atomic_json(args.output, output, c); print(json.dumps(c.json_safe(output), indent=2, sort_keys=True))


if __name__ == "__main__": main()
