"""Train dense audio-mouth synchronization from cached 4 FPS features.

Only REAL videos are used. Aligned two-second sequences are positives and the
same audio paired with mouth sequences shifted by 500 or 1000 ms are negatives.
The frozen visual and audio anti-spoofing backbones are never loaded here.
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
DEFAULT_DENSE_ROOT = PROJECT_ROOT / "data" / "interim" / "av_sync_features" / "dense_v1"
DEFAULT_VISUAL_ROOT = PROJECT_ROOT / "data" / "interim" / "visual_embeddings" / "convnext_tiny_v1"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "train.csv"
DEFAULT_VALIDATION_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "splits" / "validation.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "av_sync_dense"
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "interim" / "training_logs" / "av_sync_dense"
DENSE_SCHEMA_VERSION = "av_sync_dense_v1"
VISUAL_SCHEMA_VERSION = "visual_embedding_v1"
VISUAL_BACKBONE = "torchvision/convnext_tiny/IMAGENET1K_V1"
AUDIO_DESCRIPTOR = "LOG_MEL_MEAN_STD_RMS_ZCR_V1"
AUDIO_DIM = 82
MOUTH_DIM = 768
DEFAULT_SEGMENT_FRAMES = 8
DEFAULT_STRIDE_FRAMES = 4
DEFAULT_SHIFT_FRAMES = (2, 4)
PAIRING_STRATEGY = "REAL_ALIGNED_VS_WITHIN_VIDEO_500MS_1000MS_SHIFTS_V1"


@dataclass(frozen=True)
class Source:
    sample_id: str
    dense_path: Path
    visual_path: Path
    frame_count: int


@dataclass(frozen=True)
class Pair:
    sample_id: str
    audio: Any
    mouth: Any
    mouth_delta: Any
    label: int
    offset_frames: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES)
    parser.add_argument("--stride-frames", type=int, default=DEFAULT_STRIDE_FRAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-sources", type=int, default=None)
    parser.add_argument("--max-validation-sources", type=int, default=None)
    return parser


def require_runtime() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Activate the project environment with NumPy and PyTorch installed") from exc
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


def load_source_arrays(dense: Any, visual: Any, np: Any, expected_sample_id: str | None = None) -> tuple[tuple[Any, Any, Any] | None, str | None]:
    dense_required = {
        "schema_version", "sample_id", "av_modality", "audio_descriptor",
        "visual_schema_version", "visual_backbone", "timestamps_sec",
        "mouth_frame_indices", "audio_features",
    }
    visual_required = {"schema_version", "sample_id", "backbone", "mouth_frame_indices", "mouth_embeddings"}
    missing_dense = dense_required - set(dense.files if hasattr(dense, "files") else dense)
    missing_visual = visual_required - set(visual.files if hasattr(visual, "files") else visual)
    if missing_dense:
        return None, f"dense_missing_keys:{','.join(sorted(missing_dense))}"
    if missing_visual:
        return None, f"visual_missing_keys:{','.join(sorted(missing_visual))}"
    if str(dense["schema_version"].item()) != DENSE_SCHEMA_VERSION:
        return None, "unsupported_dense_schema"
    if str(visual["schema_version"].item()) != VISUAL_SCHEMA_VERSION:
        return None, "unsupported_visual_schema"
    if str(dense["visual_schema_version"].item()) != VISUAL_SCHEMA_VERSION:
        return None, "dense_visual_schema_mismatch"
    if str(dense["visual_backbone"].item()) != VISUAL_BACKBONE or str(visual["backbone"].item()) != VISUAL_BACKBONE:
        return None, "visual_backbone_mismatch"
    if str(dense["audio_descriptor"].item()) != AUDIO_DESCRIPTOR:
        return None, "audio_descriptor_mismatch"
    dense_id, visual_id = str(dense["sample_id"].item()), str(visual["sample_id"].item())
    if dense_id != visual_id or (expected_sample_id is not None and dense_id != expected_sample_id):
        return None, "sample_id_mismatch"
    modality = str(dense["av_modality"].item())
    if modality == "NOT_APPLICABLE":
        return None, "AV_NOT_APPLICABLE"
    if modality != "AVAILABLE":
        return None, f"invalid_av_modality:{modality}"
    audio = np.asarray(dense["audio_features"], dtype=np.float32)
    dense_indices = np.asarray(dense["mouth_frame_indices"], dtype=np.int64)
    visual_indices = np.asarray(visual["mouth_frame_indices"], dtype=np.int64)
    visual_mouth = np.asarray(visual["mouth_embeddings"], dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != AUDIO_DIM or len(audio) != len(dense_indices):
        return None, f"invalid_audio_shape:{audio.shape}"
    if visual_mouth.ndim != 2 or visual_mouth.shape[1] != MOUTH_DIM or len(visual_mouth) != len(visual_indices):
        return None, f"invalid_mouth_shape:{visual_mouth.shape}"
    position_by_frame = {int(frame_index): position for position, frame_index in enumerate(visual_indices)}
    try:
        mouth_positions = np.asarray([position_by_frame[int(index)] for index in dense_indices], dtype=np.int64)
    except KeyError:
        return None, "dense_mouth_index_missing_from_visual"
    mouth = visual_mouth[mouth_positions]
    timestamps = np.asarray(dense["timestamps_sec"], dtype=np.float32)
    if len(timestamps) != len(audio):
        return None, "timestamp_length_mismatch"
    if len(audio) and not (np.isfinite(audio).all() and np.isfinite(mouth).all() and np.isfinite(timestamps).all()):
        return None, "NON_FINITE_FEATURES"
    return (audio, mouth, timestamps), None


def inspect_source(dense_path: Path, visual_path: Path, np: Any, sample_id: str) -> tuple[int | None, str | None]:
    try:
        with np.load(dense_path, allow_pickle=False) as dense, np.load(visual_path, allow_pickle=False) as visual:
            arrays, reason = load_source_arrays(dense, visual, np, sample_id)
    except (EOFError, OSError, ValueError, KeyError) as exc:
        return None, f"unreadable_feature:{type(exc).__name__}:{exc}"
    return (len(arrays[0]), None) if arrays is not None else (None, reason)


def build_sources(manifest_path: Path, dense_root: Path, visual_root: Path, limit: int | None, minimum_frames: int, np: Any) -> tuple[list[Source], list[dict[str, str]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "semantic_class"}
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"Manifest must contain {sorted(required)}")
    rows = sorted((row for row in rows if row["semantic_class"] == "REAL"), key=lambda row: row["sample_id"])
    if limit is not None:
        if limit < 1:
            raise ValueError("Source limits must be positive")
        rows = rows[:limit]
    sources, excluded = [], []
    for row in rows:
        sample_id = row["sample_id"]
        dense_path, visual_path = dense_root / f"{sample_id}.npz", visual_root / f"{sample_id}.npz"
        missing = [name for name, path in (("DENSE", dense_path), ("VISUAL", visual_path)) if not path.exists()]
        if missing:
            excluded.append({"sample_id": sample_id, "reason": f"MISSING_{'_AND_'.join(missing)}_FEATURE_FILE"})
            continue
        count, reason = inspect_source(dense_path, visual_path, np, sample_id)
        if reason is not None or count is None:
            excluded.append({"sample_id": sample_id, "reason": reason or "INVALID_FEATURE"})
        elif count < minimum_frames:
            excluded.append({"sample_id": sample_id, "reason": f"INSUFFICIENT_DENSE_FRAMES:{count}<{minimum_frames}"})
        else:
            sources.append(Source(sample_id, dense_path, visual_path, count))
    return sources, excluded


def load_arrays(source: Source, np: Any) -> tuple[Any, Any, Any]:
    with np.load(source.dense_path, allow_pickle=False) as dense, np.load(source.visual_path, allow_pickle=False) as visual:
        arrays, reason = load_source_arrays(dense, visual, np, source.sample_id)
    if arrays is None:
        raise RuntimeError(f"Feature changed after preflight: {source.sample_id}: {reason}")
    return arrays


def contiguous_runs(timestamps: Any, np: Any, maximum_gap_seconds: float = 0.375) -> list[tuple[int, int]]:
    if len(timestamps) == 0:
        return []
    breaks = np.flatnonzero(np.diff(timestamps) > maximum_gap_seconds) + 1
    boundaries = [0, *breaks.tolist(), len(timestamps)]
    return [(boundaries[index], boundaries[index + 1]) for index in range(len(boundaries) - 1)]


def deterministic_choice(values: list[int], key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def build_pairs(source: Source, np: Any, segment_frames: int, stride_frames: int, seed: int) -> list[Pair]:
    audio, mouth, timestamps = load_arrays(source, np)
    mouth_delta = np.zeros_like(mouth)
    if len(mouth) > 1:
        mouth_delta[1:] = mouth[1:] - mouth[:-1]
    pairs: list[Pair] = []
    for run_start, run_end in contiguous_runs(timestamps, np):
        mouth_delta[run_start] = 0.0
        run_length = run_end - run_start
        if run_length < segment_frames + min(DEFAULT_SHIFT_FRAMES):
            continue
        for start in range(run_start, run_end - segment_frames + 1, stride_frames):
            valid_offsets = [
                offset for magnitude in DEFAULT_SHIFT_FRAMES for offset in (-magnitude, magnitude)
                if run_start <= start + offset and start + offset + segment_frames <= run_end
            ]
            if not valid_offsets:
                continue
            offset = deterministic_choice(valid_offsets, f"{source.sample_id}:{start}", seed)
            audio_segment = audio[start:start + segment_frames]
            pairs.append(Pair(
                source.sample_id, audio_segment, mouth[start:start + segment_frames],
                mouth_delta[start:start + segment_frames], 1, 0,
            ))
            shifted_start = start + offset
            pairs.append(Pair(
                source.sample_id, audio_segment, mouth[shifted_start:shifted_start + segment_frames],
                mouth_delta[shifted_start:shifted_start + segment_frames], 0, offset,
            ))
    return pairs


def build_all_pairs(sources: list[Source], np: Any, segment_frames: int, stride_frames: int, seed: int) -> tuple[list[Pair], list[dict[str, str]]]:
    pairs, excluded = [], []
    for source in sources:
        source_pairs = build_pairs(source, np, segment_frames, stride_frames, seed)
        if source_pairs:
            pairs.extend(source_pairs)
        else:
            excluded.append({"sample_id": source.sample_id, "reason": "NO_CONTIGUOUS_SHIFTABLE_SEGMENTS"})
    return pairs, excluded


def fit_audio_normalizer(pairs: list[Pair], np: Any) -> tuple[Any, Any]:
    aligned = np.concatenate([pair.audio for pair in pairs if pair.label == 1], axis=0).astype(np.float64)
    mean = aligned.mean(axis=0)
    std = np.maximum(aligned.std(axis=0), 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def build_components(torch: Any) -> tuple[Any, Any]:
    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, pairs: list[Pair], audio_mean: Any, audio_std: Any, np: Any):
            self.pairs, self.mean, self.std, self.np = pairs, audio_mean, audio_std, np

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any, int, str]:
            pair = self.pairs[index]
            audio = (pair.audio - self.mean) / self.std
            return (
                torch.from_numpy(audio.astype(self.np.float32, copy=False)),
                torch.from_numpy(pair.mouth.astype(self.np.float32, copy=False)),
                torch.from_numpy(pair.mouth_delta.astype(self.np.float32, copy=False)),
                torch.tensor(pair.label, dtype=torch.float32), pair.offset_frames, pair.sample_id,
            )

    class DenseAVSyncModel(torch.nn.Module):
        def __init__(self, projection_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.audio_projection = torch.nn.Sequential(
                torch.nn.Linear(AUDIO_DIM, projection_dim), torch.nn.GELU(), torch.nn.Dropout(dropout),
            )
            self.mouth_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(MOUTH_DIM * 2), torch.nn.Linear(MOUTH_DIM * 2, projection_dim),
                torch.nn.GELU(), torch.nn.Dropout(dropout),
            )
            heads = 4 if projection_dim % 4 == 0 else 1
            self.audio_to_mouth = torch.nn.MultiheadAttention(projection_dim, heads, dropout=dropout, batch_first=True)
            self.mouth_to_audio = torch.nn.MultiheadAttention(projection_dim, heads, dropout=dropout, batch_first=True)
            self.audio_norm = torch.nn.LayerNorm(projection_dim)
            self.mouth_norm = torch.nn.LayerNorm(projection_dim)
            self.temporal = torch.nn.Sequential(
                torch.nn.Conv1d(projection_dim * 4, hidden_dim, kernel_size=3, padding=1),
                torch.nn.GELU(), torch.nn.Dropout(dropout),
                torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1), torch.nn.GELU(),
            )
            self.classifier = torch.nn.Sequential(torch.nn.LayerNorm(hidden_dim), torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, 1))

        def forward(self, audio: Any, mouth: Any, mouth_delta: Any) -> Any:
            audio_hidden = self.audio_projection(audio)
            mouth_hidden = self.mouth_projection(torch.cat((mouth, mouth_delta), dim=-1))
            audio_cross, _ = self.audio_to_mouth(audio_hidden, mouth_hidden, mouth_hidden, need_weights=False)
            mouth_cross, _ = self.mouth_to_audio(mouth_hidden, audio_hidden, audio_hidden, need_weights=False)
            audio_hidden = self.audio_norm(audio_hidden + audio_cross)
            mouth_hidden = self.mouth_norm(mouth_hidden + mouth_cross)
            joint = torch.cat((audio_hidden, mouth_hidden, torch.abs(audio_hidden - mouth_hidden), audio_hidden * mouth_hidden), dim=-1)
            temporal = self.temporal(joint.transpose(1, 2)).mean(dim=-1)
            return self.classifier(temporal).squeeze(-1)

    return PairDataset, DenseAVSyncModel


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
    labels, probabilities = np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)
    positives, negatives = int(labels.sum()), len(labels) - int(labels.sum())
    if not positives or not negatives:
        raise RuntimeError("Validation pairs must contain both classes")
    ranks = rankdata_average(probabilities, np)
    roc_auc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels, sorted_probabilities = labels[order], probabilities[order]
    cumulative, average_precision, start = np.cumsum(sorted_labels), 0.0, 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        group_positives = int(sorted_labels[start:end].sum())
        if group_positives:
            average_precision += float(cumulative[end - 1] / end) * group_positives / positives
        start = end
    best_balanced_accuracy, best_threshold = -1.0, 0.5
    for threshold in np.unique(probabilities):
        predictions = probabilities >= threshold
        score = (float(predictions[labels == 1].mean()) + float((~predictions[labels == 0]).mean())) / 2
        if score > best_balanced_accuracy:
            best_balanced_accuracy, best_threshold = score, float(threshold)
    return {
        "roc_auc": float(roc_auc), "average_precision": float(average_precision),
        "balanced_accuracy": best_balanced_accuracy, "validation_threshold": best_threshold,
    }


def evaluate(model: Any, loader: Any, device: str, runtime: dict[str, Any]) -> dict[str, float]:
    torch, np = runtime["torch"], runtime["np"]
    model.eval()
    logits, labels, offsets = [], [], []
    with torch.inference_mode():
        for audio, mouth, mouth_delta, batch_labels, batch_offsets, _ in loader:
            logits.append(model(audio.to(device), mouth.to(device), mouth_delta.to(device)).cpu())
            labels.append(batch_labels)
            offsets.append(batch_offsets)
    logits_array = torch.cat(logits).numpy()
    labels_array = torch.cat(labels).numpy()
    offsets_array = torch.cat(offsets).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -80.0, 80.0)))
    metrics = binary_metrics(labels_array, probabilities, np)
    metrics["mean_aligned_probability"] = float(probabilities[offsets_array == 0].mean())
    for magnitude in DEFAULT_SHIFT_FRAMES:
        selected = probabilities[np.abs(offsets_array) == magnitude]
        metrics[f"mean_shift_{magnitude * 250}ms_probability"] = float(selected.mean()) if len(selected) else float("nan")
    return metrics


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
    if min(args.epochs, args.patience, args.batch_size, args.projection_dim, args.hidden_dim, args.segment_frames, args.stride_frames) < 1:
        raise ValueError("Training and sequence parameters must be positive")
    runtime = require_runtime()
    torch, np = runtime["torch"], runtime["np"]
    device = resolve_device(torch, args.device)
    set_seed(args.seed, torch)
    minimum_frames = args.segment_frames + min(DEFAULT_SHIFT_FRAMES)
    train_sources, train_preflight_excluded = build_sources(
        args.train_manifest, args.dense_root, args.visual_root, args.max_train_sources, minimum_frames, np,
    )
    validation_sources, validation_preflight_excluded = build_sources(
        args.validation_manifest, args.dense_root, args.visual_root, args.max_validation_sources, minimum_frames, np,
    )
    train_pairs, train_segment_excluded = build_all_pairs(train_sources, np, args.segment_frames, args.stride_frames, args.seed)
    validation_pairs, validation_segment_excluded = build_all_pairs(validation_sources, np, args.segment_frames, args.stride_frames, args.seed)
    train_excluded = train_preflight_excluded + train_segment_excluded
    validation_excluded = validation_preflight_excluded + validation_segment_excluded
    if not train_pairs or not validation_pairs:
        raise RuntimeError("No usable pairs. Build complete train and validation dense AV features first.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_jsonl(args.log_dir / f"{run_id}_train_excluded.jsonl", train_excluded)
    write_jsonl(args.log_dir / f"{run_id}_validation_excluded.jsonl", validation_excluded)
    audio_mean, audio_std = fit_audio_normalizer(train_pairs, np)
    Dataset, Model = build_components(torch)
    train_dataset = Dataset(train_pairs, audio_mean, audio_std, np)
    validation_dataset = Dataset(validation_pairs, audio_mean, audio_std, np)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.startswith("cuda"),
    )
    model = Model(args.projection_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, stale_epochs = float("-inf"), 0, 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.checkpoint_dir / f"av_sync_dense_{run_id}.pt"
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        losses = []
        for audio, mouth, mouth_delta, labels, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(audio.to(device), mouth.to(device), mouth_delta.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                "model_config": {
                    "audio_dim": AUDIO_DIM, "mouth_dim": MOUTH_DIM,
                    "projection_dim": args.projection_dim, "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout, "segment_frames": args.segment_frames,
                    "stride_frames": args.stride_frames,
                },
                "audio_normalizer_mean": torch.from_numpy(audio_mean.copy()),
                "audio_normalizer_std": torch.from_numpy(audio_std.copy()),
                "best_validation_metrics": metrics,
                "train_manifest_sha256": manifest_hash(args.train_manifest),
                "validation_manifest_sha256": manifest_hash(args.validation_manifest),
                "dense_schema_version": DENSE_SCHEMA_VERSION,
                "visual_schema_version": VISUAL_SCHEMA_VERSION,
                "visual_backbone": VISUAL_BACKBONE, "audio_descriptor": AUDIO_DESCRIPTOR,
                "pairing_strategy": PAIRING_STRATEGY,
                "label_semantics": {"0": "TEMPORALLY_SHIFTED", "1": "TIMESTAMP_ALIGNED"},
                "output_semantics": {"sync_probability": "sigmoid(logit)", "av_inconsistency_score": "1-sync_probability"},
                "seed": args.seed,
            }, checkpoint_path, torch)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    train_source_ids = {pair.sample_id for pair in train_pairs}
    validation_source_ids = {pair.sample_id for pair in validation_pairs}
    report = {
        "run_id": run_id, "device": device, "pairing_strategy": PAIRING_STRATEGY,
        "segment_frames": args.segment_frames, "stride_frames": args.stride_frames,
        "segment_seconds": args.segment_frames / 4.0,
        "shift_milliseconds": [500, 1000],
        "train_real_sources": len(train_source_ids), "validation_real_sources": len(validation_source_ids),
        "train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs),
        "train_excluded": len(train_excluded), "validation_excluded": len(validation_excluded),
        "best_epoch": best_epoch, "best_roc_auc": best_auc,
        "checkpoint_path": str(checkpoint_path), "history": history,
    }
    atomic_json_save(report, args.log_dir / f"{run_id}_summary.json")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
