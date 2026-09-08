"""Shared leakage-safe helpers for the Pramaan-X V2 adaptation scripts.

This module intentionally uses only the standard library and NumPy.  It keeps
the protocol boundary independent from model loading so it can be unit-tested
without a GPU, checkpoints, or media files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


HINDI_LANGUAGE = "hindi"
ROLES = ("adaptation_train", "adaptation_dev", "adaptation_holdout")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_hindi_mlaad(row: dict[str, str]) -> bool:
    """The downloader records the canonical MLAAD Hindi language as ``hindi``."""
    return row.get("dataset", "").strip().casefold() == "mlaad" and row.get("language", "").strip().casefold() == HINDI_LANGUAGE


def assert_hindi_only(rows: Iterable[dict[str, str]]) -> None:
    bad = [row.get("sample_id", "<missing>") for row in rows if row.get("dataset", "").strip().casefold() == "mlaad" and not is_hindi_mlaad(row)]
    if bad:
        raise RuntimeError(f"Non-Hindi MLAAD rows entered V2 data: {bad[:5]}")


def stable_group_key(row: dict[str, str]) -> tuple[str, str]:
    """Prefer a source-level identifier; never claim unavailable identity metadata."""
    for field in ("identity", "original_id"):
        value = row.get(field, "").strip()
        if value and value.casefold() not in {"unknown", "none", "null", "n/a"}:
            return field, value
    return "sample_id_fallback", row["sample_id"]


def deterministic_assign(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    """Assign complete source groups to 60/20/20 roles using a stable hash.

    The balance is by group count rather than samples.  This is intentional:
    keeping a source group intact is more important than exact sample ratios.
    """
    if not rows:
        raise RuntimeError("No MLAAD Hindi rows are available for the V2 protocol.")
    assert_hindi_only(rows)
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(stable_group_key(row), []).append(row)
    if len(groups) < 3:
        raise RuntimeError("Need at least three independent grouping keys for train/dev/holdout.")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).hexdigest())
    train_end = max(1, round(len(ordered) * 0.60))
    dev_end = max(train_end + 1, round(len(ordered) * 0.80))
    dev_end = min(dev_end, len(ordered) - 1)
    by_group: dict[tuple[str, str], str] = {}
    for index, key in enumerate(ordered):
        by_group[key] = "adaptation_train" if index < train_end else "adaptation_dev" if index < dev_end else "adaptation_holdout"
    result: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda value: value["sample_id"]):
        kind, value = stable_group_key(row)
        result.append({
            "sample_id": row["sample_id"], "dataset": row["dataset"], "language": row["language"],
            "audio_label": row["audio_label"], "modality": row["modality"], "generator_tool": row.get("generator_tool", ""),
            "source_group_field": kind, "source_group": value, "role": by_group[(kind, value)],
        })
    validate_assignments(result)
    return result


def validate_assignments(rows: list[dict[str, str]]) -> None:
    if not rows or {row["role"] for row in rows} != set(ROLES):
        raise RuntimeError("Protocol must contain non-empty adaptation train, dev, and holdout roles.")
    assert_hindi_only(rows)
    group_roles: dict[tuple[str, str], set[str]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        if row["sample_id"] in seen_ids:
            raise RuntimeError(f"Duplicate protocol sample_id: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        group_roles.setdefault((row["source_group_field"], row["source_group"]), set()).add(row["role"])
    leaked = [group for group, roles in group_roles.items() if len(roles) != 1]
    if leaked:
        raise RuntimeError(f"Source group crosses protocol roles: {leaked[:3]}")


def quantiles(scores: Any, np: Any) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("Scores must be non-empty finite values.")
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "p05": float(np.quantile(values, .05)), "p25": float(np.quantile(values, .25)), "p50": float(np.quantile(values, .50)), "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def binary_metrics(labels: Any, scores: Any, threshold: float, np: Any) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if set(labels.tolist()) - {0, 1} or not len(labels):
        raise RuntimeError("Binary metrics require non-empty labels in {0, 1}.")
    result: dict[str, Any] = {"sample_count": int(len(labels)), "positive_count": int(labels.sum()), "negative_count": int(len(labels) - labels.sum()), "threshold": float(threshold), "score_distribution": quantiles(scores, np)}
    predicted = scores >= threshold
    tp, tn = int(((predicted == 1) & (labels == 1)).sum()), int(((predicted == 0) & (labels == 0)).sum())
    fp, fn = int(((predicted == 1) & (labels == 0)).sum()), int(((predicted == 0) & (labels == 1)).sum())
    result.update({"tpr": tp / max(1, tp + fn), "tnr": tn / max(1, tn + fp), "fpr": fp / max(1, fp + tn), "detection_rate": float(predicted[labels == 1].mean()) if int(labels.sum()) else None})
    if not labels.sum() or labels.sum() == len(labels):
        result.update({"roc_auc": None, "average_precision": None, "balanced_accuracy": None, "metric_note": "Single-class data: ranking and balanced metrics are unavailable."})
        return result
    ranks = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2
        cursor = end
    positives = int(labels.sum()); negatives = len(labels) - positives
    auc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    sorted_labels = labels[np.argsort(-scores, kind="mergesort")]
    ap = float((np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1))[sorted_labels == 1].sum() / positives)
    result.update({"roc_auc": float(auc), "average_precision": ap, "balanced_accuracy": (result["tpr"] + result["tnr"]) / 2, "metric_note": None})
    return result


def acceptance_decision(fakeav: dict[str, Any], itw: dict[str, Any], hindi_holdout: dict[str, Any], v1_itw_auc: float = .946, min_fac_auc: float = .98, min_fac_ba: float = .97, max_itw_auc_drop: float = .02) -> dict[str, Any]:
    reasons: list[str] = []
    if fakeav.get("roc_auc") is None or fakeav["roc_auc"] < min_fac_auc: reasons.append("FakeAVCeleb validation AUC gate failed")
    if fakeav.get("balanced_accuracy") is None or fakeav["balanced_accuracy"] < min_fac_ba: reasons.append("FakeAVCeleb validation balanced-accuracy gate failed")
    if itw.get("roc_auc") is None or itw["roc_auc"] < v1_itw_auc - max_itw_auc_drop: reasons.append("In-the-Wild AUC protection gate failed")
    if hindi_holdout.get("detection_rate") is None: reasons.append("Hindi holdout is not fake-labelled data")
    return {"decision": "V1_AUDIO_RETAINED" if reasons else "V2_AUDIO_ACCEPTED", "reasons": reasons, "gates": {"min_fakeav_auc": min_fac_auc, "min_fakeav_balanced_accuracy": min_fac_ba, "v1_itw_auc": v1_itw_auc, "max_itw_auc_drop": max_itw_auc_drop}}
