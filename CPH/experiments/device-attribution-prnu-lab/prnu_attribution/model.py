from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

import numpy as np

from .dataset import ImageRecord, discover_dataset, summarize_records
from .metadata import feature_vector, read_metadata
from .prnu import DEFAULT_SIZE, correlation, fingerprint, residual_from_image


MODEL_JSON = "model.json"
FINGERPRINTS_NPZ = "prnu_fingerprints.npz"

PLATFORM_PATTERNS = {
    "whatsapp": [
        r"\bwhatsapp\b",
        r"\bimg-\d{8}-wa\d+\b",
        r"\bwa\d{4,}\b",
        r"whatsapp[ _-]image",
    ],
    "instagram": [
        r"\binstagram\b",
        r"instagram[_-]",
        r"\binsta\b",
        r"\big[_-]",
        r"\binstagram[ _-]image\b",
    ],
    "facebook": [
        r"\bfacebook\b",
        r"facebook[_-]",
        r"\bfb_img\b",
        r"\bmessenger\b",
        r"\bfbstaging\b",
    ],
    "reddit": [
        r"\breddit\b",
        r"reddit[_-]",
        r"\brdt[_-]",
        r"\bi\.redd\.it\b",
        r"\bredditsave\b",
    ],
    "telegram": [
        r"\btelegram\b",
        r"\bphoto_\d{4}-\d{2}-\d{2}\b",
    ],
    "x": [
        r"\btwitter\b",
        r"\btweet\b",
        r"\bx\.com\b",
    ],
}


@dataclass
class TrainedModel:
    manifest: dict[str, Any]
    fingerprints: dict[str, np.ndarray]


def train(dataset_root: str | Path, model_dir: str | Path, prnu_size: int = DEFAULT_SIZE) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    records = discover_dataset(dataset_root)
    warnings: list[str] = []
    if not records:
        warnings.append(f"No images found under {dataset_root}. Add device folders and run training again.")

    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for rec in records:
        try:
            vector, meta = feature_vector(rec.path)
            rows.append({"record": rec, "vector": vector, "metadata": meta})
            features.append(vector)
        except Exception as exc:
            warnings.append(f"Feature extraction failed for {rec.path}: {type(exc).__name__}: {exc}")

    feature_mean = np.zeros(139, dtype=np.float32)
    feature_std = np.ones(139, dtype=np.float32)
    if features:
        matrix = np.stack(features, axis=0)
        feature_mean = matrix.mean(axis=0)
        feature_std = matrix.std(axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

    device_centroids = _centroids(rows, "device_id", feature_mean, feature_std)
    platform_centroids = _centroids(rows, "platform", feature_mean, feature_std)
    metadata_profiles = _metadata_profiles(rows)

    fp_arrays: dict[str, np.ndarray] = {}
    fp_index: dict[str, str] = {}
    template_index: list[dict[str, Any]] = []
    for device_id in sorted({rec.device_id for rec in records}):
        refs = [
            rec.path
            for rec in records
            if rec.device_id == device_id and rec.platform == "original" and rec.split == "reference"
        ]
        if len(refs) < 3:
            refs = [
                rec.path
                for rec in records
                if rec.device_id == device_id and rec.platform == "original"
            ]
            if 0 < len(refs) < 3:
                warnings.append(f"{device_id}: only {len(refs)} original images available for PRNU; use 20-50 for stronger matching.")
        if not refs:
            warnings.append(f"{device_id}: no original reference images for PRNU fingerprint.")
            continue
        fp, fp_errors = fingerprint(refs, size=prnu_size)
        warnings.extend(fp_errors[:10])
        if fp is not None:
            key = _unique_key(fp_arrays, f"ref_{_safe_key(device_id)}")
            fp_arrays[key] = fp
            fp_index[device_id] = key

        for platform in sorted({rec.platform for rec in records if rec.device_id == device_id}):
            platform_paths = [
                rec.path
                for rec in records
                if rec.device_id == device_id and rec.platform == platform
            ]
            min_images = 3 if platform == "original" else 2
            if len(platform_paths) < min_images:
                continue
            template_fp, template_errors = fingerprint(platform_paths, size=prnu_size)
            warnings.extend(template_errors[:10])
            if template_fp is None:
                continue
            key = _unique_key(fp_arrays, f"tpl_{_safe_key(device_id)}_{_safe_key(platform)}")
            fp_arrays[key] = template_fp
            template_index.append(
                {
                    "device_id": device_id,
                    "platform": platform,
                    "key": key,
                    "image_count": len(platform_paths),
                    "source": "platform_residual_template",
                }
            )

    if fp_arrays:
        np.savez_compressed(model_dir / FINGERPRINTS_NPZ, **fp_arrays)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "summary": summarize_records(records),
        "prnu": {
            "size": prnu_size,
            "fingerprint_file": FINGERPRINTS_NPZ if fp_arrays else None,
            "fingerprints": fp_index,
            "platform_templates": template_index,
            "method": "center-crop grayscale residual minus two-pass 5x5 blur, averaged per reference device; platform templates preserve device matching after app recompression",
        },
        "feature_model": {
            "feature_count": int(feature_mean.shape[0]),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "device_centroids": device_centroids,
            "platform_centroids": platform_centroids,
            "metadata_profiles": metadata_profiles,
        },
        "warnings": warnings,
    }
    (model_dir / MODEL_JSON).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_model(model_dir: str | Path) -> TrainedModel:
    model_dir = Path(model_dir)
    manifest_path = model_dir / MODEL_JSON
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model not trained yet: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprints: dict[str, np.ndarray] = {}
    fp_file = manifest.get("prnu", {}).get("fingerprint_file")
    if fp_file and (model_dir / fp_file).exists():
        npz = np.load(model_dir / fp_file)
        for key in npz.files:
            fingerprints[key] = np.asarray(npz[key], dtype=np.float32)
    return TrainedModel(manifest=manifest, fingerprints=fingerprints)


def predict(
    image_path: str | Path,
    model_dir: str | Path,
    mode: str = "both",
    allow_demo_model: bool = False,
) -> dict[str, Any]:
    image_path = Path(image_path).resolve()
    trained = load_model(model_dir)
    manifest = trained.manifest
    mode = mode if mode in {"device", "platform", "both"} else "both"

    vector, meta = feature_vector(image_path)
    model_info = manifest["feature_model"]
    mean = np.asarray(model_info["feature_mean"], dtype=np.float32)
    std = np.asarray(model_info["feature_std"], dtype=np.float32)
    z = (vector - mean) / std

    dataset_root = Path(manifest.get("dataset_root") or ".").resolve()
    trained_model_allowed = (
        not _is_demo_dataset(dataset_root)
        or allow_demo_model
        or _is_relative_to(image_path, dataset_root)
    )

    device_feature_rankings = _centroid_rankings(z, model_info.get("device_centroids", {}))
    platform_rankings = _centroid_rankings(z, model_info.get("platform_centroids", {}))
    preferred_platform = platform_rankings[0]["label"] if platform_rankings else None
    prnu_rankings = (
        _prnu_rankings(image_path, trained, preferred_platform=preferred_platform)
        if trained_model_allowed
        else []
    )

    if mode in {"device", "both"}:
        device = _combine_device_rankings(
            prnu_rankings,
            device_feature_rankings if trained_model_allowed else [],
            meta,
            model_info.get("metadata_profiles", {}) if trained_model_allowed else {},
        )
    else:
        device = _not_requested("device")

    if mode in {"platform", "both"}:
        platform = _platform_result(platform_rankings, meta, trained_model_allowed=trained_model_allowed)
    else:
        platform = _not_requested("platform")

    result = {
        "case": {
            "image": str(image_path),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "model_created_at": manifest.get("created_at"),
            "dataset_root": manifest.get("dataset_root"),
            "mode": mode,
            "trained_model_allowed": trained_model_allowed,
            "allow_demo_model": allow_demo_model,
        },
        "device_attribution": device,
        "platform_attribution": platform,
        "metadata": meta,
        "rankings": {
            "prnu": prnu_rankings,
            "device_features": device_feature_rankings,
            "platform_features": platform_rankings,
        },
        "warnings": _prediction_warnings(
            prnu_rankings,
            platform_rankings if trained_model_allowed else [],
            meta,
            mode=mode,
        ),
    }
    if not trained_model_allowed:
        result["warnings"].insert(
            0,
            "The trained model is a synthetic demo model, so learned PRNU/platform classes were not used for this external upload.",
        )
    result["channels"] = {
        "device": device,
        "platform": platform,
    }
    return result


def predict_platform_trail(image_paths: list[str | Path], model_dir: str | Path) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("At least one image is required for platform trail analysis.")

    analyses = [predict(path, model_dir, mode="platform") for path in image_paths]
    items = []
    for idx, analysis in enumerate(analyses):
        meta = analysis.get("metadata", {})
        platform = analysis.get("channels", {}).get("platform", {})
        date_hint = _date_hint(meta)
        items.append(
            {
                "input_order": idx + 1,
                "analysis_index": idx,
                "file_name": meta.get("file_name"),
                "platform": platform.get("prediction"),
                "display_name": platform.get("display_name"),
                "confidence": platform.get("confidence", 0.0),
                "method": platform.get("primary_method"),
                "date_hint": date_hint,
                "signature": platform.get("signature"),
                "metadata_state": platform.get("metadata_state"),
                "evidence": platform.get("evidence", []),
            }
        )

    dated = [item for item in items if item["date_hint"]]
    if len(dated) >= 2:
        ordered = sorted(items, key=lambda item: (item["date_hint"] is None, item["date_hint"] or "", item["input_order"]))
        basis = "metadata_or_filename_timestamp"
    else:
        ordered = items
        basis = "upload_order"

    sequence = []
    for item in ordered:
        label = item.get("display_name") or _humanize_label(item.get("platform") or "unknown")
        if not sequence or sequence[-1] != label:
            sequence.append(label)

    last = ordered[-1] if ordered else items[-1]
    last_label = last.get("platform")
    trail_text = " -> ".join(sequence) if sequence else "Unknown"
    confidence_values = [float(item.get("confidence") or 0.0) for item in ordered if item.get("platform")]
    trail_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0

    result = analyses[int(last.get("analysis_index", len(analyses) - 1))]
    result["case"]["mode"] = "platform_trail"
    result["platform_trail"] = {
        "sequence": sequence,
        "last_platform": last_label,
        "last_platform_display": last.get("display_name"),
        "confidence": trail_confidence,
        "basis": basis,
        "items": ordered,
        "limitation": "A full cross-platform history cannot be proven from one image alone; this trail is reconstructed from the uploaded copies and their timestamps/signatures.",
    }
    result["channels"]["platform"] = {
        "prediction": last_label,
        "display_name": last.get("display_name") or _humanize_label(last_label or "unknown"),
        "confidence": max(float(last.get("confidence") or 0.0), trail_confidence),
        "primary_method": "platform_trail_reconstruction",
        "scope": "last_platform_from_uploaded_trail",
        "signature": last.get("signature"),
        "metadata_state": last.get("metadata_state"),
        "rankings": ordered,
        "evidence": [
            f"Reconstructed platform trail: {trail_text}.",
            f"Sequence basis: {basis.replace('_', ' ')}.",
            "Trail mode uses multiple uploaded copies of the same content; a single file usually only supports last-platform attribution.",
        ],
    }
    result["platform_attribution"] = result["channels"]["platform"]
    return result


def _centroids(rows: list[dict[str, Any]], field: str, mean: np.ndarray, std: np.ndarray) -> dict[str, list[float]]:
    grouped: dict[str, list[np.ndarray]] = {}
    for row in rows:
        rec: ImageRecord = row["record"]
        label = getattr(rec, field)
        grouped.setdefault(label, []).append((row["vector"] - mean) / std)
    return {label: np.mean(np.stack(vectors, axis=0), axis=0).astype(float).tolist() for label, vectors in grouped.items()}


def _metadata_profiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_device: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rec: ImageRecord = row["record"]
        if rec.platform == "original":
            by_device.setdefault(rec.device_id, []).append(row["metadata"])
    profiles: dict[str, dict[str, Any]] = {}
    for device_id, items in by_device.items():
        profiles[device_id] = {
            "make": _mode([m.get("make") for m in items]),
            "model": _mode([m.get("model") for m in items]),
            "software": _mode([m.get("software") for m in items]),
            "resolution": _mode([f"{m.get('width')}x{m.get('height')}" for m in items if m.get("width") and m.get("height")]),
        }
    return profiles


def _centroid_rankings(z: np.ndarray, centroids: dict[str, list[float]], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for label, centroid in centroids.items():
        c = np.asarray(centroid, dtype=np.float32)
        distance = float(np.linalg.norm(z - c))
        score = 1.0 / (1.0 + distance)
        rows.append({"label": label, "score": score, "distance": distance})
    total = sum(r["score"] for r in rows) or 1.0
    for row in rows:
        row["confidence"] = float(row["score"] / total)
    rows.sort(key=lambda item: item["confidence"], reverse=True)
    return rows[:limit]


def _prnu_rankings(image_path: Path, trained: TrainedModel, preferred_platform: str | None = None) -> list[dict[str, Any]]:
    fp_index = trained.manifest.get("prnu", {}).get("fingerprints", {})
    platform_templates = trained.manifest.get("prnu", {}).get("platform_templates", [])
    if not (fp_index or platform_templates) or not trained.fingerprints:
        return []
    try:
        query = residual_from_image(image_path, size=int(trained.manifest.get("prnu", {}).get("size") or DEFAULT_SIZE))
    except Exception:
        return []

    matches_by_device: dict[str, list[dict[str, Any]]] = {}
    for device_id, key in fp_index.items():
        fp = trained.fingerprints.get(key)
        if fp is None:
            continue
        corr = correlation(query, fp)
        matches_by_device.setdefault(device_id, []).append(
            {
                "template": "original_reference",
                "platform": "original",
                "correlation": corr,
                "score": corr,
                "image_count": None,
            }
        )

    for template in platform_templates:
        key = template.get("key")
        device_id = template.get("device_id")
        platform = template.get("platform")
        fp = trained.fingerprints.get(key)
        if not key or not device_id or fp is None:
            continue
        corr = correlation(query, fp)
        platform_bonus = 0.015 if preferred_platform and platform == preferred_platform else 0.0
        source_weight = 0.96 if platform == "original" else 0.9
        matches_by_device.setdefault(device_id, []).append(
            {
                "template": f"{device_id}/{platform}",
                "platform": platform,
                "correlation": corr,
                "score": corr * source_weight + platform_bonus,
                "image_count": template.get("image_count"),
            }
        )

    rows = []
    for device_id, matches in matches_by_device.items():
        matches.sort(key=lambda item: item["score"], reverse=True)
        best = matches[0]
        rows.append(
            {
                "label": device_id,
                "correlation": best["correlation"],
                "score": best["score"],
                "best_template": best["template"],
                "best_template_platform": best["platform"],
                "template_matches": matches[:4],
            }
        )
    if not rows:
        return []
    values = np.asarray([r["score"] for r in rows], dtype=np.float32)
    shifted = values - values.min()
    if float(shifted.max()) < 1e-9:
        confidences = np.ones_like(values) / len(values)
    else:
        confidences = shifted / shifted.sum() if float(shifted.sum()) > 0 else np.ones_like(values) / len(values)
    for row, conf in zip(rows, confidences):
        row["confidence"] = float(conf)
    rows.sort(key=lambda item: (item["confidence"], item["score"], item["correlation"]), reverse=True)
    return rows


def _combine_device_rankings(
    prnu_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence: list[str] = []
    metadata_label = _metadata_device_label(meta)
    profile_matches = _metadata_profile_matches(meta, profiles)
    trusted_prnu = _trusted_prnu_rows(prnu_rows)

    if trusted_prnu:
        top = trusted_prnu[0]
        template = top.get("best_template") or "original_reference"
        evidence.append(f"PRNU residual is closest to {top['label']} using {template}, correlation {top['correlation']:.5f}.")
        if metadata_label:
            evidence.append(f"Camera metadata reports {metadata_label}.")
        prediction = top["label"]
        profile = profiles.get(prediction, {})
        confidence = _prnu_confidence(top, trusted_prnu)
        rankings = [{"label": row["label"], "confidence": row["confidence"]} for row in trusted_prnu[:8]]
        return {
            "prediction": prediction,
            "known_device_match": prediction,
            "camera_model": metadata_label,
            "display_name": _device_display_name(prediction, profile),
            "confidence": confidence,
            "primary_method": "prnu",
            "metadata_profile": profile,
            "rankings": rankings,
            "evidence": evidence,
        }

    if metadata_label:
        evidence.append(f"Camera metadata reports {metadata_label}.")
        if profile_matches:
            best = profile_matches[0]
            prediction = best["label"]
            profile = profiles.get(prediction, {})
            evidence.append(f"Metadata matches trained device profile {prediction}.")
            return {
                "prediction": prediction,
                "known_device_match": prediction,
                "camera_model": metadata_label,
                "display_name": _device_display_name(prediction, profile),
                "confidence": best["confidence"],
                "primary_method": "metadata",
                "metadata_profile": profile,
                "rankings": profile_matches,
                "evidence": evidence,
            }
        return {
            "prediction": metadata_label,
            "known_device_match": None,
            "camera_model": metadata_label,
            "display_name": metadata_label,
            "confidence": 0.68,
            "primary_method": "metadata",
            "metadata_profile": {},
            "rankings": [],
            "evidence": evidence,
        }

    if feature_rows and feature_rows[0].get("confidence", 0.0) >= 0.65:
        top_feature = feature_rows[0]
        prediction = top_feature["label"]
        profile = profiles.get(prediction, {})
        evidence.append(f"Low-level file features are closest to {prediction}, but no metadata or reliable PRNU was available.")
        return {
            "prediction": prediction,
            "known_device_match": prediction,
            "camera_model": None,
            "display_name": _device_display_name(prediction, profile),
            "confidence": min(float(top_feature["confidence"]), 0.55),
            "primary_method": "weak_file_features",
            "metadata_profile": profile,
            "rankings": feature_rows[:8],
            "evidence": evidence,
        }

    evidence.append("No reliable PRNU match or camera metadata was available for device attribution.")
    return {
        "prediction": None,
        "known_device_match": None,
        "camera_model": None,
        "display_name": "Unknown device",
        "confidence": 0.0,
        "primary_method": "inconclusive",
        "metadata_profile": {},
        "rankings": [],
        "evidence": evidence,
    }


def _platform_result(rows: list[dict[str, Any]], meta: dict[str, Any], trained_model_allowed: bool = True) -> dict[str, Any]:
    evidence = []
    compression = meta.get("compression") or {}
    votes: dict[str, float] = {}
    vote_evidence: dict[str, list[str]] = {}

    if compression.get("metadata_state") == "stripped_or_absent":
        evidence.append("EXIF metadata is absent/stripped, common after social-platform recompression.")
    if compression.get("has_jpeg_quantization"):
        evidence.append(
            f"JPEG quantization signature {compression.get('quant_hash')} with mean {compression.get('quant_mean')} was used for platform matching."
        )

    if rows and trained_model_allowed:
        top = rows[0]
        strength = min(max(float(top.get("confidence", 0.0)), 0.0), 1.0)
        if strength >= 0.34:
            _add_vote(
                votes,
                vote_evidence,
                top["label"],
                0.18 + strength * 0.55,
                f"Trained JPEG/compression signature is closest to {top['label']} ({strength:.0%}).",
            )

    for hint in _platform_hints(meta):
        _add_vote(votes, vote_evidence, hint["label"], hint["score"], hint["evidence"])

    for hint in _dimension_platform_hints(meta):
        _add_vote(votes, vote_evidence, hint["label"], hint["score"], hint["evidence"])

    if meta.get("exif_present") and meta.get("make") and meta.get("model"):
        _add_vote(
            votes,
            vote_evidence,
            "original",
            0.32,
            "Camera make/model metadata is still present, which is more consistent with an original/direct file than a social download.",
        )

    if not votes:
        return {
            "prediction": None,
            "display_name": "Unknown platform",
            "confidence": 0.0,
            "primary_method": "inconclusive",
            "scope": "immediate_source_platform",
            "rankings": [],
            "evidence": evidence or ["No reliable platform signature or filename/software hint was available."],
        }

    ranked_votes = _rank_votes(votes)
    top_vote = ranked_votes[0]
    if top_vote["confidence"] < 0.42 and top_vote["score"] < 0.55:
        return {
            "prediction": None,
            "display_name": "Unknown platform",
            "confidence": 0.0,
            "primary_method": "low_confidence_signal_fusion",
            "scope": "immediate_source_platform",
            "signature": compression.get("quant_hash"),
            "metadata_state": compression.get("metadata_state"),
            "rankings": ranked_votes,
            "evidence": evidence + ["Available platform signals were too weak to make a responsible claim."],
        }

    label = top_vote["label"]
    platform_evidence = evidence + vote_evidence.get(label, [])
    if len(ranked_votes) > 1:
        runner_up = ranked_votes[1]
        platform_evidence.append(f"Next closest platform candidate is {runner_up['label']} ({runner_up['confidence']:.0%}).")

    return {
        "prediction": label,
        "display_name": _humanize_label(label),
        "confidence": top_vote["confidence"],
        "primary_method": "platform_signal_fusion",
        "scope": "immediate_source_platform",
        "signature": compression.get("quant_hash"),
        "metadata_state": compression.get("metadata_state"),
        "rankings": ranked_votes,
        "evidence": platform_evidence,
    }


def _prediction_warnings(
    prnu_rows: list[dict[str, Any]],
    platform_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    mode: str = "both",
) -> list[str]:
    warnings = []
    if mode in {"device", "both"} and not prnu_rows:
        warnings.append("PRNU unavailable or unusable for this image; exact physical-device attribution is weaker.")
    elif mode in {"device", "both"} and len(prnu_rows) > 1 and prnu_rows[0]["confidence"] - prnu_rows[1]["confidence"] < 0.15:
        warnings.append("Top PRNU candidates are close; treat device attribution as tentative.")
    if mode in {"platform", "both"} and platform_rows and platform_rows[0]["confidence"] < 0.5:
        warnings.append("Platform confidence is low; compression signature is not distinctive enough.")
    if meta.get("error"):
        warnings.append(f"Image metadata extraction had an error: {meta['error']}")
    return warnings


def _not_requested(channel: str) -> dict[str, Any]:
    label = "Device attribution not requested" if channel == "device" else "Platform attribution not requested"
    return {
        "prediction": None,
        "display_name": label,
        "confidence": 0.0,
        "primary_method": "not_requested",
        "rankings": [],
        "evidence": [label],
    }


def _trusted_prnu_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    top = rows[0]
    second_score = rows[1].get("score", rows[1].get("correlation", 0.0)) if len(rows) > 1 else -1.0
    top_score = top.get("score", top.get("correlation", 0.0))
    top_corr = top.get("correlation", 0.0)
    margin = float(top_score) - float(second_score)
    if float(top_corr) < 0.03 or margin < 0.01:
        return []
    return rows


def _prnu_confidence(top: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    corr = max(0.0, float(top.get("correlation", 0.0)))
    relative = float(top.get("confidence", 0.0))
    confidence = 0.55 + min(corr, 0.35) * 0.8 + relative * 0.12
    if len(rows) > 1:
        margin = float(top.get("score", corr)) - float(rows[1].get("score", rows[1].get("correlation", 0.0)))
        confidence += min(max(margin, 0.0), 0.25) * 0.2
    return float(min(confidence, 0.95))


def _metadata_device_label(meta: dict[str, Any]) -> str | None:
    make = _clean_meta(meta.get("make"))
    model = _clean_meta(meta.get("model"))
    if make and model and make.lower() not in model.lower():
        return f"{make} {model}"
    return model or make


def _metadata_profile_matches(meta: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    make = (_clean_meta(meta.get("make")) or "").lower()
    model = (_clean_meta(meta.get("model")) or "").lower()
    if not make and not model:
        return []
    rows = []
    for device_id, profile in profiles.items():
        score = 0.0
        profile_make = str(profile.get("make") or "").lower()
        profile_model = str(profile.get("model") or "").lower()
        if model and profile_model and (model == profile_model or model in profile_model or profile_model in model):
            score += 0.75
        if make and profile_make and (make == profile_make or make in profile_make or profile_make in make):
            score += 0.2
        if score:
            rows.append({"label": device_id, "confidence": min(score, 0.85)})
    rows.sort(key=lambda item: item["confidence"], reverse=True)
    return rows


def _platform_hint(meta: dict[str, Any]) -> dict[str, Any] | None:
    haystack = " ".join(
        str(meta.get(key) or "")
        for key in ["path", "file_name", "software", "format"]
    ).lower()
    checks = [
        ("whatsapp", ["whatsapp image", "whatsapp", "wa_"], 0.72),
        ("instagram", ["instagram", "insta", "ig_"], 0.7),
        ("facebook", ["fb_img", "facebook", "messenger"], 0.7),
        ("reddit", ["reddit", "rdt_"], 0.68),
        ("telegram", ["telegram"], 0.68),
    ]
    for label, needles, confidence in checks:
        if any(needle in haystack for needle in needles):
            return {"label": label, "confidence": confidence}
    return None


def _platform_hints(meta: dict[str, Any]) -> list[dict[str, Any]]:
    haystack = " ".join(
        str(meta.get(key) or "")
        for key in ["path", "file_name", "software", "format"]
    ).lower()
    hints = []
    for label, patterns in PLATFORM_PATTERNS.items():
        matched = []
        for pattern in patterns:
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                matched.append(pattern)
        if not matched:
            continue
        score = 0.78 if label in {"whatsapp", "facebook"} else 0.72
        if meta.get("software") and re.search("|".join(patterns), str(meta.get("software")), flags=re.IGNORECASE):
            score += 0.08
        hints.append(
            {
                "label": label,
                "score": min(score, 0.9),
                "evidence": f"Filename/path/software contains a {label} download pattern.",
                "matched_patterns": matched,
            }
        )
    return hints


def _dimension_platform_hints(meta: dict[str, Any]) -> list[dict[str, Any]]:
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if width <= 0 or height <= 0:
        return []
    long_edge = max(width, height)
    short_edge = min(width, height)
    ratio = long_edge / max(short_edge, 1)
    hints = []

    if long_edge in {1080, 1350, 1440} or width == 1080 or height == 1080:
        hints.append(
            {
                "label": "instagram",
                "score": 0.22,
                "evidence": f"Image dimensions {width}x{height} match common Instagram-served size constraints.",
            }
        )
    if long_edge in {720, 960, 1200, 2048}:
        hints.append(
            {
                "label": "facebook",
                "score": 0.18,
                "evidence": f"Image dimensions {width}x{height} match common Facebook/Messenger-served size constraints.",
            }
        )
    if long_edge <= 1280 and ratio < 2.2:
        hints.append(
            {
                "label": "reddit",
                "score": 0.1,
                "evidence": f"Image dimensions {width}x{height} are compatible with a Reddit-hosted download, but this is a weak signal.",
            }
        )
    return hints


def _add_vote(
    votes: dict[str, float],
    evidence: dict[str, list[str]],
    label: str,
    score: float,
    reason: str,
) -> None:
    votes[label] = votes.get(label, 0.0) + float(score)
    evidence.setdefault(label, []).append(reason)


def _rank_votes(votes: dict[str, float]) -> list[dict[str, Any]]:
    rows = [{"label": label, "score": score} for label, score in votes.items()]
    rows.sort(key=lambda item: item["score"], reverse=True)
    total = sum(max(0.0, row["score"]) for row in rows) + 0.15
    for row in rows:
        row["confidence"] = float(max(0.0, row["score"]) / total) if total else 0.0
        row["display_name"] = _humanize_label(row["label"])
    return rows


def _date_hint(meta: dict[str, Any]) -> str | None:
    value = _clean_meta(meta.get("datetime"))
    if value:
        normalized = value.replace(":", "-", 2)
        return normalized

    name = re.sub(r"^[a-f0-9]{32}[_-]", "", str(meta.get("file_name") or ""), flags=re.IGNORECASE)
    patterns = [
        r"(?P<date>\d{4}-\d{2}-\d{2})[^\d]+(?P<hour>\d{1,2})[.\-_](?P<minute>\d{2})[.\-_](?P<second>\d{2})",
        r"(?:img|screenshot|photo|pxl|dsc|fb_img|mmexport|signal)[_-]?(?P<date>\d{8})[_-]?(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})",
        r"(?:img|screenshot|photo|pxl|dsc|fb_img|mmexport|signal)[_-]?(?P<date>\d{8})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        date = match.group("date")
        if len(date) == 8:
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        hour = match.groupdict().get("hour") or "00"
        minute = match.groupdict().get("minute") or "00"
        second = match.groupdict().get("second") or "00"
        return f"{date} {int(hour):02d}:{int(minute):02d}:{int(second):02d}"
    return None


def _clean_meta(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _is_demo_dataset(path: Path) -> bool:
    return path.name.lower().startswith("sample_dataset")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mode(values: list[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        if value is None or value == "":
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]


def _safe_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    return key.strip("_") or "device"


def _unique_key(existing: dict[str, np.ndarray], desired: str) -> str:
    key = desired
    idx = 2
    while key in existing:
        key = f"{desired}_{idx}"
        idx += 1
    return key


def _device_display_name(device_id: str | None, profile: dict[str, Any]) -> str | None:
    if not device_id:
        return None
    model_hint = profile.get("model")
    make_hint = profile.get("make")
    human_label = _humanize_label(device_id)
    if model_hint and str(model_hint).lower() not in human_label.lower():
        return f"{human_label} ({model_hint})"
    if make_hint and str(make_hint).lower() not in human_label.lower():
        return f"{human_label} ({make_hint})"
    return human_label


def _humanize_label(value: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", value).strip()
    if not cleaned:
        return value
    words = []
    for word in cleaned.split():
        if len(word) == 1:
            words.append(word.upper())
        elif word.lower() in {"prnu", "exif", "jpeg", "oneplus"}:
            words.append("OnePlus" if word.lower() == "oneplus" else word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)
