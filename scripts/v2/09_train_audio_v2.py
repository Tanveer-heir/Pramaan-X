"""Continue the frozen-feature audio head using Hindi MLAAD train rows only.

The XLSR-SLS ONNX backbone is intentionally never loaded here.  The V1 head is
the parent, and this script writes only a separate V2 checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

from v2_common import HINDI_LANGUAGE, atomic_json, read_csv, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def load_trainer() -> Any:
    path = ROOT / "scripts/training/14_train_audio_classifier.py"
    spec = importlib.util.spec_from_file_location("pramaan_v1_audio", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "data/interim/v2_protocol/mlaad_hindi_protocol.csv")
    p.add_argument("--fac-train-manifest", type=Path, default=ROOT / "data/metadata/splits/train.csv")
    p.add_argument("--fac-feature-root", type=Path, default=ROOT / "data/interim/audio_embeddings/xlsr_sls_v1")
    p.add_argument("--external-feature-root", type=Path, default=ROOT / "data/interim/modern_challenge/audio_embeddings/xlsr_sls_v1")
    p.add_argument("--parent-checkpoint", type=Path, default=ROOT / "checkpoints/audio_classifier/audio_classifier_20260904T181732Z.pt")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/audio_classifier_v2")
    p.add_argument("--log-dir", type=Path, default=ROOT / "data/interim/v2_logs/audio")
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--fac-dev-fraction", type=float, default=.10)
    p.add_argument("--max-train-samples", type=int); p.add_argument("--max-validation-samples", type=int)
    p.add_argument("--seed", type=int, default=20260907); p.add_argument("--device", default="auto")
    return p


def load_examples(rows: list[dict[str, str]], root: Path, trainer: Any, np: Any, limit: int | None) -> tuple[list[tuple[Any, int, str]], list[dict[str, str]]]:
    items, excluded = [], []
    for row in sorted(rows, key=lambda value: value["sample_id"]):
        if limit is not None and len(items) >= limit: break
        path = root / f"{row['sample_id']}.npz"
        if not path.is_file(): excluded.append({"sample_id": row["sample_id"], "reason": "MISSING_FEATURE"}); continue
        try:
            with np.load(path, allow_pickle=False) as payload: sequence, reason = trainer.load_sequence(payload, np)
            if sequence is None: excluded.append({"sample_id": row["sample_id"], "reason": reason or "INVALID_FEATURE"}); continue
            items.append((sequence, int(row["audio_label"]), row["sample_id"]))
        except (OSError, ValueError, KeyError) as exc: excluded.append({"sample_id": row["sample_id"], "reason": f"UNREADABLE:{type(exc).__name__}"})
    return items, excluded


def deterministic_fac_split(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < fraction < .5: raise ValueError("--fac-dev-fraction must be in (0, .5)")
    # Group by declared identity where possible to avoid source identity leakage.
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("split") != "train": raise RuntimeError("Audio V2 may consume FakeAVCeleb train.csv only")
        key = row.get("identity") or row["sample_id"]
        groups.setdefault(key, []).append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    dev_keys = set(ordered[:max(1, round(len(ordered) * fraction))])
    train, dev = [row for key in groups if key not in dev_keys for row in groups[key]], [row for key in groups if key in dev_keys for row in groups[key]]
    if not {int(row["audio_label"]) for row in train} == {0, 1} or not {int(row["audio_label"]) for row in dev} == {0, 1}: raise RuntimeError("Deterministic FakeAVCeleb development split lacks both audio labels")
    return train, dev


def main() -> None:
    args = parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.learning_rate <= 0: raise ValueError("positive training settings required")
    import numpy as np
    import torch
    trainer = load_trainer(); device = trainer.resolve_device(torch, args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    protocol = read_csv(args.protocol)
    required = {"sample_id", "dataset", "language", "audio_label", "role"}
    if not protocol or required - set(protocol[0]): raise RuntimeError(f"Protocol missing {sorted(required - set(protocol[0]) if protocol else required)}")
    if any(row["dataset"].casefold() != "mlaad" or row["language"].casefold() != HINDI_LANGUAGE for row in protocol): raise RuntimeError("Protocol contains non-Hindi MLAAD rows")
    if any(row["role"] not in {"adaptation_train", "adaptation_dev", "adaptation_holdout"} for row in protocol): raise RuntimeError("Invalid V2 role")
    if any(row["role"] == "adaptation_holdout" for row in protocol if row["role"] == "adaptation_train"): raise RuntimeError("Internal protocol role validation failed")
    hindi_train = [row for row in protocol if row["role"] == "adaptation_train"]
    hindi_dev = [row for row in protocol if row["role"] == "adaptation_dev"]
    fac_rows = read_csv(args.fac_train_manifest); fac_train, fac_dev = deterministic_fac_split(fac_rows, args.fac_dev_fraction, args.seed)
    parent_sha = sha256_file(args.parent_checkpoint)
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    Dataset, collate, Model = trainer.build_components(torch)
    config = parent["model_config"]; model = Model(int(config["hidden_dim"]), float(config["dropout"]))
    model.load_state_dict(parent["model_state_dict"], strict=True); model = model.to(device)
    fac_train_items, fac_train_excluded = load_examples(fac_train, args.fac_feature_root, trainer, np, args.max_train_samples)
    fac_dev_items, fac_dev_excluded = load_examples(fac_dev, args.fac_feature_root, trainer, np, args.max_validation_samples)
    hindi_train_items, hindi_train_excluded = load_examples(hindi_train, args.external_feature_root, trainer, np, args.max_train_samples)
    hindi_dev_items, hindi_dev_excluded = load_examples(hindi_dev, args.external_feature_root, trainer, np, args.max_validation_samples)
    train_items, dev_items = fac_train_items + hindi_train_items, fac_dev_items + hindi_dev_items
    if not train_items or not dev_items or set(label for _, label, _ in train_items) != {0, 1}: raise RuntimeError("V2 training requires cached FakeAVCeleb real/fake examples and Hindi fake examples")
    class Data(torch.utils.data.Dataset):
        def __init__(self, values: list[tuple[Any, int, str]]): self.values = [(torch.from_numpy(x.copy()), y, sid) for x, y, sid in values]
        def __len__(self) -> int: return len(self.values)
        def __getitem__(self, index: int) -> tuple[Any, int, str]: return self.values[index]
    labels = [label for _, label, _ in train_items]; weights = torch.DoubleTensor([1 / labels.count(label) for label in labels])
    train_loader = torch.utils.data.DataLoader(Data(train_items), batch_size=args.batch_size, sampler=torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(args.seed)), collate_fn=collate)
    dev_loader = torch.utils.data.DataLoader(Data(dev_items), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); criterion = torch.nn.BCEWithLogitsLoss()
    best, best_state, stale, history = float("-inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for padded, lengths, labels_tensor, _ in train_loader:
            optimizer.zero_grad(set_to_none=True); logits = model(padded.to(device), lengths.to(device)); loss = criterion(logits, labels_tensor.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); probs, y = [], []
        with torch.inference_mode():
            for padded, lengths, labels_tensor, _ in dev_loader: probs.extend(torch.sigmoid(model(padded.to(device), lengths.to(device))).cpu().tolist()); y.extend(labels_tensor.tolist())
        metric = trainer.binary_metrics(np.asarray(y, dtype=np.int64), np.asarray(probs), np); metric.update({"epoch": epoch, "train_loss": sum(losses) / len(losses)})
        history.append(metric)
        if metric["roc_auc"] > best: best, best_state, stale = metric["roc_auc"], {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, 0
        else: stale += 1
        if stale >= args.patience: break
    if best_state is None: raise RuntimeError("No V2 checkpoint state was selected")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); checkpoint_path = args.checkpoint_dir / f"audio_classifier_v2_{run_id}.pt"; checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": "pramaan_x_audio_v2", "model_state_dict": best_state, "model_config": config, "feature_schema_version": trainer.FEATURE_SCHEMA_VERSION, "feature_model_sha256": trainer.MODEL_SHA256, "label_semantics": parent.get("label_semantics"), "parent_v1_checkpoint": {"path": str(args.parent_checkpoint), "sha256": parent_sha}, "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol), "hindi_language": HINDI_LANGUAGE}, "training_seed": args.seed, "hyperparameters": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "fac_dev_fraction": args.fac_dev_fraction}, "best_epoch": max(history, key=lambda row: row["roc_auc"])["epoch"], "best_development_metrics": max(history, key=lambda row: row["roc_auc"]), "limitations": "V2 training uses FakeAVCeleb train plus MLAAD Hindi adaptation_train. MLAAD dev/holdout are excluded from gradients."}
    with tempfile.NamedTemporaryFile(dir=checkpoint_path.parent, delete=False) as handle:
        temporary = Path(handle.name); torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, checkpoint_path); torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    report = {"run_id": run_id, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "device": device, "history": history, "counts": {"fac_train": len(fac_train_items), "fac_dev": len(fac_dev_items), "hindi_train": len(hindi_train_items), "hindi_dev": len(hindi_dev_items)}, "exclusions": {"fac_train": fac_train_excluded, "fac_dev": fac_dev_excluded, "hindi_train": hindi_train_excluded, "hindi_dev": hindi_dev_excluded}}
    atomic_json(args.log_dir / f"audio_v2_{run_id}_report.json", report); print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
