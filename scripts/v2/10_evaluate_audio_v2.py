"""Evaluate a V1 or V2 cached-feature audio head without non-Hindi MLAAD."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from v2_common import HINDI_LANGUAGE, atomic_json, binary_metrics, read_csv, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def trainer() -> Any:
    path = ROOT / "scripts/training/14_train_audio_classifier.py"; spec = importlib.util.spec_from_file_location("v2_audio_trainer", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "data/interim/v2_protocol/mlaad_hindi_protocol.csv")
    p.add_argument("--external-manifest", type=Path, default=ROOT / "data/metadata/modern_challenge_manifest.csv")
    p.add_argument("--fac-validation", type=Path, default=ROOT / "data/metadata/splits/validation.csv")
    p.add_argument("--external-feature-root", type=Path, default=ROOT / "data/interim/modern_challenge/audio_embeddings/xlsr_sls_v1")
    p.add_argument("--fac-feature-root", type=Path, default=ROOT / "data/interim/audio_embeddings/xlsr_sls_v1")
    p.add_argument("--threshold", type=float, default=.5); p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=ROOT / "data/interim/v2_evaluations/audio_v2_evaluation.json")
    return p


def score(model: Any, rows: list[dict[str, str]], root: Path, tr: Any, torch: Any, np: Any, device: str) -> tuple[list[int], Any, list[dict[str, str]]]:
    values, labels, skipped = [], [], []
    model.eval()
    for row in sorted(rows, key=lambda value: value["sample_id"]):
        path = root / f"{row['sample_id']}.npz"
        if not path.is_file(): skipped.append({"sample_id": row["sample_id"], "reason": "MISSING_FEATURE"}); continue
        try:
            with np.load(path, allow_pickle=False) as payload: sequence, reason = tr.load_sequence(payload, np)
            if sequence is None: skipped.append({"sample_id": row["sample_id"], "reason": reason or "INVALID_FEATURE"}); continue
            with torch.inference_mode(): probability = float(torch.sigmoid(model(torch.from_numpy(sequence).unsqueeze(0).to(device), torch.tensor([len(sequence)], device=device))).item())
            values.append(probability); labels.append(int(row["audio_label"]))
        except (OSError, ValueError, KeyError) as exc: skipped.append({"sample_id": row["sample_id"], "reason": f"UNREADABLE:{type(exc).__name__}"})
    return labels, np.asarray(values, dtype=np.float64), skipped


def main() -> None:
    args = parser().parse_args(); import numpy as np; import torch
    tr = trainer(); device = tr.resolve_device(torch, args.device); payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _, _, Model = tr.build_components(torch); config = payload["model_config"]; model = Model(int(config["hidden_dim"]), float(config["dropout"])); model.load_state_dict(payload["model_state_dict"], strict=True); model.to(device).eval()
    protocol = read_csv(args.protocol)
    if any(row.get("dataset", "").casefold() != "mlaad" or row.get("language", "").casefold() != HINDI_LANGUAGE for row in protocol): raise RuntimeError("Evaluation protocol contains non-Hindi MLAAD")
    external = read_csv(args.external_manifest); itw = [row for row in external if row.get("dataset") == "In-the-Wild"]
    if any(row.get("dataset", "").casefold() == "mlaad" and row.get("language", "").casefold() != HINDI_LANGUAGE for row in protocol): raise RuntimeError("Non-Hindi MLAAD included")
    fac = read_csv(args.fac_validation)
    results: dict[str, Any] = {"schema_version": "pramaan_x_audio_v2_evaluation_v1", "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "threshold": args.threshold, "device": device, "limitations": ["MLAAD Hindi is fake-only; its detection rate is not accuracy and ranking metrics are null.", "FakeAVCeleb validation remains development evidence and calibration/test are not read.", "In-the-Wild has been observed before the V2 protocol was frozen and is a diagnostic protection set."]}
    collections = {"FakeAVCeleb_validation": (fac, args.fac_feature_root), "In-the-Wild_diagnostic": (itw, args.external_feature_root)}
    for role in ("adaptation_dev", "adaptation_holdout"):
        collections[f"MLAAD_Hindi_{role}"] = ([row for row in protocol if row["role"] == role], args.external_feature_root)
    evaluations = {}
    for name, (rows, root) in collections.items():
        labels, scores, skipped = score(model, rows, root, tr, torch, np, device)
        data = binary_metrics(np.asarray(labels), scores, args.threshold, np) if len(labels) else {"sample_count": 0, "error": "NO_USABLE_FEATURES"}
        data["skipped"] = skipped; data["dataset"] = name; evaluations[name] = data
    results["evaluations"] = evaluations; atomic_json(args.output, results); print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__": main()
