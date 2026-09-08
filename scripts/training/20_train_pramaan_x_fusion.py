"""Train the bounded, cache-only four-class Pramaan-X fusion model."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "scripts" / "fusion" / "pramaan_x_fusion.py"


def core() -> Any:
    spec = importlib.util.spec_from_file_location("pramaan_x_fusion_core", CORE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {CORE_PATH}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-manifest", type=Path, default=ROOT / "data/metadata/splits/validation.csv")
    p.add_argument("--visual-root", type=Path, default=ROOT / "data/interim/visual_embeddings/convnext_tiny_v1")
    p.add_argument("--audio-root", type=Path, default=ROOT / "data/interim/audio_embeddings/xlsr_sls_v1")
    p.add_argument("--dense-root", type=Path, default=ROOT / "data/interim/av_sync_features/dense_v1")
    p.add_argument("--visual-checkpoint", type=Path, default=ROOT / "checkpoints/visual_temporal/visual_temporal_20260904T145626Z.pt")
    p.add_argument("--audio-checkpoint", type=Path, default=ROOT / "checkpoints/audio_classifier/audio_classifier_20260904T181732Z.pt")
    p.add_argument("--av-checkpoint", type=Path, default=ROOT / "checkpoints/av_sync_dense/av_sync_dense_20260905T055537Z.pt")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/fusion/pramaan_x_hackathon_final.pth")
    p.add_argument("--evidence-output", type=Path, default=ROOT / "data/interim/fusion_evidence/validation_evidence.jsonl")
    p.add_argument("--report-output", type=Path, default=ROOT / "data/interim/training_logs/fusion/pramaan_x_fusion_report.json")
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--dropout", type=float, default=.15); p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--seed", type=int, default=9173)
    p.add_argument("--device", default="auto")
    return p


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any, c: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as h:
        tmp = Path(h.name); json.dump(c.json_safe(value), h, indent=2, sort_keys=True, allow_nan=False); h.write("\n"); h.flush(); os.fsync(h.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]], c: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as h:
        tmp = Path(h.name)
        for row in rows: h.write(json.dumps(c.json_safe(row), sort_keys=True, allow_nan=False) + "\n")
        h.flush(); os.fsync(h.fileno())
    os.replace(tmp, path)


def git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError): return {"sha": None, "dirty": None}


def grouped_split(rows: list[dict[str, Any]], c: Any, seed: int) -> tuple[set[str], set[str]]:
    identities = sorted({row["identity"] for row in rows}, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    dev = set(identities[:max(1, round(len(identities) * .2))]); train = set(identities) - dev
    for name, group in (("fusion-train", train), ("fusion-dev", dev)):
        seen = {row["semantic_class"] for row in rows if row["identity"] in group}
        missing = set(c.CLASS_NAMES) - seen
        if missing: raise RuntimeError(f"{name} identity split lacks classes: {sorted(missing)}")
    return train, dev


def metrics(labels: Any, predicted: Any, c: Any) -> dict[str, Any]:
    import numpy as np
    matrix = np.zeros((4, 4), dtype=int)
    for actual, guess in zip(labels, predicted, strict=True): matrix[int(actual), int(guess)] += 1
    per = {}
    for index, name in enumerate(c.CLASS_NAMES):
        tp, fp, fn = matrix[index, index], matrix[:, index].sum() - matrix[index, index], matrix[index, :].sum() - matrix[index, index]
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        per[name] = {"precision": float(precision), "recall": float(recall), "f1": float(2 * precision * recall / max(1e-12, precision + recall)), "support": int(matrix[index, :].sum())}
    return {"accuracy": float((matrix.diagonal().sum()) / max(1, matrix.sum())), "macro_f1": float(sum(x["f1"] for x in per.values()) / 4), "confusion_matrix": matrix.tolist(), "per_class": per}


def train_variant(torch: Any, np: Any, x_train: Any, y_train: Any, x_dev: Any, y_dev: Any, config: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    class Fusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.net = torch.nn.Sequential(torch.nn.LayerNorm(config["input_dim"]), torch.nn.Linear(config["input_dim"], config["hidden_dim"]), torch.nn.GELU(), torch.nn.Dropout(config["dropout"]), torch.nn.Linear(config["hidden_dim"], 4))
        def forward(self, x: Any) -> Any: return self.net(x)
    torch.manual_seed(seed); model = Fusion().to(config["device"])
    counts = np.bincount(y_train, minlength=4); weights = torch.tensor(len(y_train) / np.maximum(counts, 1) / 4, dtype=torch.float32, device=config["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    data = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = torch.utils.data.DataLoader(data, batch_size=config["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, best_state, stale = None, None, 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(x.to(config["device"])), y.to(config["device"]), weight=weights); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.inference_mode(): pred = model(torch.from_numpy(x_dev).to(config["device"])).argmax(1).cpu().numpy()
        result = metrics(y_dev, pred, config["core"]); result["epoch"] = epoch
        if best is None or result["macro_f1"] > best["macro_f1"]:
            best, best_state, stale = result, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= config["patience"]: break
    assert best is not None and best_state is not None
    return best, best_state


def main() -> None:
    args = parser().parse_args(); c = core()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1: raise ValueError("positive training settings required")
    import numpy as np
    import torch
    device = c.resolve_device(torch, args.device); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with args.validation_manifest.open(newline="", encoding="utf-8") as h: rows = list(csv.DictReader(h))
    needed = {"sample_id", "identity", "component_id", "semantic_class", "split"}
    if not rows or needed - set(rows[0]): raise RuntimeError(f"Validation manifest missing {sorted(needed - set(rows[0]) if rows else needed)}")
    if any(row["split"] != "validation" for row in rows): raise RuntimeError("Fusion may consume validation.csv only")
    if set(row["semantic_class"] for row in rows) - set(c.CLASS_NAMES): raise RuntimeError("Unknown semantic class in validation manifest")
    runner = c.BranchRunner.create(ROOT, {"visual": args.visual_checkpoint, "audio": args.audio_checkpoint, "dense_av": args.av_checkpoint}, ["visual", "audio", "dense_av"], device)
    evidence = []
    for row in sorted(rows, key=lambda x: x["sample_id"]):
        sid = row["sample_id"]; scored = c.score_evidence(runner, {"visual": args.visual_root / f"{sid}.npz", "audio": args.audio_root / f"{sid}.npz", "dense_av": args.dense_root / f"{sid}.npz"}, ["visual", "audio", "dense_av"])
        evidence.append({"sample_id": sid, "identity": row["identity"], "component_id": row["component_id"], "semantic_class": row["semantic_class"], "target": c.CLASS_TO_INDEX[row["semantic_class"]], **scored})
    atomic_jsonl(args.evidence_output, evidence, c)
    train_ids, dev_ids = grouped_split(evidence, c, args.seed)
    report: dict[str, Any] = {"format_version": c.FORMAT_VERSION, "protocol_limitation": "Fusion is fitted from validation branch outputs. Branch selection already used validation, so results are hackathon-only and indirectly selection-biased.", "evidence_path": str(args.evidence_output), "evidence_sha256": digest(args.evidence_output), "branch_coverage": {name: sum(bool(row.get(name)) for row in evidence) for name in ("visual_available", "audio_available", "av_available")}, "identity_split": {"train": sorted(train_ids), "dev": sorted(dev_ids)}}
    variants: dict[str, Any] = {}; states: dict[str, Any] = {}; norms: dict[str, Any] = {}
    base_config = {"hidden_dim": args.hidden_dim, "dropout": args.dropout, "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "device": device, "core": c}
    for name, enabled in (("visual_audio", ["visual", "audio"]), ("visual_audio_av", ["visual", "audio", "dense_av"])):
        order = c.feature_order(enabled); train = [row for row in evidence if row["identity"] in train_ids]; dev = [row for row in evidence if row["identity"] in dev_ids]
        raw_train = np.asarray([c.vectorize(row, order) for row in train], dtype=np.float32); raw_dev = np.asarray([c.vectorize(row, order) for row in dev], dtype=np.float32)
        x_train, mean, std = c.normalize_features(raw_train, order); x_dev, _, _ = c.normalize_features(raw_dev, order, mean, std)
        config = {**base_config, "input_dim": len(order)}; result, state = train_variant(torch, np, x_train, np.asarray([row["target"] for row in train]), x_dev, np.asarray([row["target"] for row in dev]), config, args.seed)
        variants[name], states[name], norms[name] = {"enabled_branches": enabled, "input_feature_order": order, "metrics": result}, state, (mean, std)
    selected, reason = c.selection_reason(variants["visual_audio"]["metrics"], variants["visual_audio_av"]["metrics"])
    selected_meta = variants[selected]; mean, std = norms[selected]
    payload = {"format_version": c.FORMAT_VERSION, "state_dict": states[selected], "architecture": {"input_dim": len(selected_meta["input_feature_order"]), "hidden_dim": args.hidden_dim, "dropout": args.dropout}, "class_names": list(c.CLASS_NAMES), "input_feature_order": selected_meta["input_feature_order"], "normalization": {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "normalized_features": [x for x in selected_meta["input_feature_order"] if x.endswith("_logit")]}, "missing_evidence_policy": "Logits are zero-filled only with an explicit availability mask. No branch availability implies a hard inference error.", "enabled_branches": selected_meta["enabled_branches"], "accepted_branch_checkpoints": {name: {"path": str(path), "sha256": c.BRANCH_HASHES[name]} for name, path in (("visual", args.visual_checkpoint), ("audio", args.audio_checkpoint), ("dense_av", args.av_checkpoint)) if name in selected_meta["enabled_branches"]}, "av_decision": {"decision": "KEEP_OPTIONAL", "audit_records_sha256": "0badb115ad09806d5712055bfad599c3d5b7951bb615e9d5cec424b0f2fa9cd0", "feature_transform": "av_inconsistency_logit = -mean(sync_logits)"}, "manifest_sha256": {"validation": digest(args.validation_manifest)}, "training_seed": args.seed, "selected_epoch": selected_meta["metrics"]["epoch"], "dev_metrics": selected_meta["metrics"], "variant_comparison": variants, "selection_reason": reason, "git": git_state()}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.checkpoint.parent, delete=False) as h:
        temp = Path(h.name); torch.save(payload, h); h.flush(); os.fsync(h.fileno())
    os.replace(temp, args.checkpoint)
    try: torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc: raise RuntimeError(f"Final checkpoint failed safe reload: {exc}") from exc
    report.update({"variants": variants, "selected_variant": selected, "selection_reason": reason, "final_checkpoint": str(args.checkpoint), "final_checkpoint_sha256": c.sha256(args.checkpoint)})
    atomic_json(args.report_output, report, c); print(json.dumps(c.json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__": main()
