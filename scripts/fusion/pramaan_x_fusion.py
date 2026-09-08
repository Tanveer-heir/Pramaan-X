"""Shared, cache-only Pramaan-X fusion helpers.

The module deliberately imports the accepted branch trainers at runtime.  This
keeps cache validation and branch architectures tied to the code that produced
the accepted checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


CLASS_NAMES = ("REAL", "VISUAL_MANIPULATION", "AUDIO_MANIPULATION", "AUDIO_VISUAL_MANIPULATION")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
FORMAT_VERSION = "pramaan_x_fusion_v1"
BRANCH_HASHES = {
    "visual": "ac348279792a064c24ef53f4c083e0d6bdaca7b5d801bf39d21806a5c5e87f3f",
    "audio": "67ea124dfa3ad71133898c548ca371be2fbe58633a92420f19d5a322375fe8cb",
    "dense_av": "cd93a68bc168cec4853f6c280174d7fe16b3b328b5ae568112bc1d5b237b45a9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import accepted branch trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_trainers(project_root: Path) -> dict[str, Any]:
    training = project_root / "scripts" / "training"
    return {
        "visual": load_module(training / "07_train_visual_temporal.py", "pramaan_visual_trainer"),
        "audio": load_module(training / "14_train_audio_classifier.py", "pramaan_audio_trainer"),
        "dense_av": load_module(training / "19_train_dense_av_sync.py", "pramaan_dense_av_trainer"),
    }


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def require_checkpoint(path: Path, expected_digest: str, torch: Any) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Accepted checkpoint not found: {path}")
    actual = sha256(path)
    if actual != expected_digest:
        raise RuntimeError(f"Checkpoint digest mismatch for {path}: expected {expected_digest}, got {actual}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Safe checkpoint load failed for {path}: {exc}") from exc
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


@dataclass
class BranchRunner:
    models: dict[str, Any]
    checkpoints: dict[str, dict[str, Any]]
    trainers: dict[str, Any]
    np: Any
    torch: Any
    device: str

    @classmethod
    def create(cls, project_root: Path, checkpoint_paths: dict[str, Path], enabled: list[str], device: str) -> "BranchRunner":
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("NumPy and PyTorch are required for cache prediction") from exc
        trainers = load_trainers(project_root)
        resolved = resolve_device(torch, device)
        models, checkpoints = {}, {}
        for branch in enabled:
            checkpoint = require_checkpoint(checkpoint_paths[branch], BRANCH_HASHES[branch], torch)
            config = checkpoint.get("model_config", {})
            if branch == "visual":
                _, _, Model = trainers[branch].build_components(torch)
                model = Model(int(config["hidden_dim"]), float(config["dropout"]))
                if checkpoint.get("feature_schema_version") != trainers[branch].SCHEMA_VERSION:
                    raise RuntimeError("Visual feature schema does not match accepted checkpoint")
            elif branch == "audio":
                _, _, Model = trainers[branch].build_components(torch)
                model = Model(int(config["hidden_dim"]), float(config["dropout"]))
                if checkpoint.get("feature_schema_version") != trainers[branch].FEATURE_SCHEMA_VERSION:
                    raise RuntimeError("Audio feature schema does not match accepted checkpoint")
            elif branch == "dense_av":
                _, Model = trainers[branch].build_components(torch)
                model = Model(int(config["projection_dim"]), int(config["hidden_dim"]), float(config["dropout"]))
                if checkpoint.get("dense_schema_version") != trainers[branch].DENSE_SCHEMA_VERSION:
                    raise RuntimeError("Dense AV feature schema does not match accepted checkpoint")
            else:
                raise RuntimeError(f"Unsupported branch: {branch}")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            models[branch], checkpoints[branch] = model.to(resolved).eval(), checkpoint
        return cls(models, checkpoints, trainers, np, torch, resolved)

    def visual_logit(self, visual_path: Path) -> tuple[float | None, str | None]:
        try:
            with self.np.load(visual_path, allow_pickle=False) as payload:
                sequence, reason = self.trainers["visual"].shared_frame_sequence(payload, self.np)
            if sequence is None:
                return None, reason or "INVALID_VISUAL_FEATURE"
            tensor = self.torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            lengths = self.torch.tensor([len(sequence)], dtype=self.torch.long, device=self.device)
            with self.torch.inference_mode():
                return float(self.models["visual"](tensor, lengths).item()), None
        except (OSError, ValueError, KeyError) as exc:
            return None, f"UNREADABLE_VISUAL_FEATURE:{type(exc).__name__}"

    def audio_logit(self, audio_path: Path) -> tuple[float | None, str | None]:
        try:
            with self.np.load(audio_path, allow_pickle=False) as payload:
                sequence, reason = self.trainers["audio"].load_sequence(payload, self.np)
            if sequence is None:
                return None, reason or "INVALID_AUDIO_FEATURE"
            tensor = self.torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            lengths = self.torch.tensor([len(sequence)], dtype=self.torch.long, device=self.device)
            with self.torch.inference_mode():
                return float(self.models["audio"](tensor, lengths).item()), None
        except (OSError, ValueError, KeyError) as exc:
            return None, f"UNREADABLE_AUDIO_FEATURE:{type(exc).__name__}"

    def av_inconsistency_timeline(self, dense_path: Path, visual_path: Path) -> tuple[float | None, str | None, list[dict[str, float | int]]]:
        """Score dense AV once and retain its real timestamp coverage per segment."""
        trainer, checkpoint = self.trainers["dense_av"], self.checkpoints["dense_av"]
        try:
            with self.np.load(dense_path, allow_pickle=False) as dense, self.np.load(visual_path, allow_pickle=False) as visual:
                arrays, reason = trainer.load_source_arrays(dense, visual, self.np)
            if arrays is None:
                return None, reason or "INVALID_DENSE_AV_FEATURE", []
            audio, mouth, timestamps = arrays
            delta = self.np.zeros_like(mouth)
            if len(mouth) > 1:
                delta[1:] = mouth[1:] - mouth[:-1]
            segments: list[tuple[Any, Any, Any, int, int]] = []
            frames, stride = int(checkpoint["model_config"]["segment_frames"]), int(checkpoint["model_config"]["stride_frames"])
            for run_start, run_end in trainer.contiguous_runs(timestamps, self.np):
                delta[run_start] = 0.0
                for start in range(run_start, run_end - frames + 1, stride):
                    segments.append((audio[start:start + frames], mouth[start:start + frames], delta[start:start + frames], start, start + frames - 1))
            if not segments:
                return None, "NO_DENSE_AV_SEGMENTS", []
            mean = checkpoint["audio_normalizer_mean"].detach().cpu().numpy()
            std = checkpoint["audio_normalizer_std"].detach().cpu().numpy()
            values: list[float] = []
            for begin in range(0, len(segments), 64):
                batch = segments[begin:begin + 64]
                a = self.torch.from_numpy(self.np.stack([(x[0] - mean) / std for x in batch])).to(self.device)
                m = self.torch.from_numpy(self.np.stack([x[1] for x in batch])).to(self.device)
                d = self.torch.from_numpy(self.np.stack([x[2] for x in batch])).to(self.device)
                with self.torch.inference_mode():
                    values.extend(float(value) for value in self.models["dense_av"](a, m, d).detach().cpu().numpy().tolist())
            records = [
                {
                    "segment_index": index,
                    "start_sec": float(timestamps[start]),
                    "end_sec": float(timestamps[end]),
                    "center_sec": float((timestamps[start] + timestamps[end]) / 2.0),
                    "raw_model_logit": value,
                    "av_inconsistency_logit": -value,
                }
                for index, (value, (_, _, _, start, end)) in enumerate(zip(values, segments, strict=True))
            ]
            return -float(self.np.mean(values)), None, records
        except (OSError, ValueError, KeyError) as exc:
            return None, f"UNREADABLE_DENSE_AV_FEATURE:{type(exc).__name__}", []

    def av_inconsistency_logit(self, dense_path: Path, visual_path: Path) -> tuple[float | None, str | None, int]:
        value, reason, records = self.av_inconsistency_timeline(dense_path, visual_path)
        return value, reason, len(records)


def score_evidence(runner: BranchRunner, paths: dict[str, Path | None], enabled: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {"branch_reasons": {}, "coverage": {}}
    if "visual" in enabled and paths.get("visual") is not None:
        value, reason = runner.visual_logit(paths["visual"])
        values["visual_logit"], values["visual_available"] = value, value is not None
        values["branch_reasons"]["visual"] = reason
    else:
        values.update({"visual_logit": None, "visual_available": False})
        values["branch_reasons"]["visual"] = "DISABLED_OR_NOT_SUPPLIED"
    if "audio" in enabled and paths.get("audio") is not None:
        value, reason = runner.audio_logit(paths["audio"])
        values["audio_logit"], values["audio_available"] = value, value is not None
        values["branch_reasons"]["audio"] = reason
    else:
        values.update({"audio_logit": None, "audio_available": False})
        values["branch_reasons"]["audio"] = "DISABLED_OR_NOT_SUPPLIED"
    if "dense_av" in enabled and paths.get("dense_av") is not None and paths.get("visual") is not None:
        value, reason, records = runner.av_inconsistency_timeline(paths["dense_av"], paths["visual"])
        values["av_inconsistency_logit"], values["av_available"] = value, value is not None
        values["branch_reasons"]["dense_av"], values["coverage"]["dense_av_segments"] = reason, len(records)
        values["temporal_evidence"] = {"dense_av": {
            "available": value is not None,
            "evidence_type": "audio_mouth_correspondence",
            "aggregate": {"av_inconsistency_logit": value, "segment_count": len(records)},
            "segments": records,
            "interpretation_note": "Temporal values measure model sensitivity to audio-mouth correspondence and are not calibrated manipulation probabilities.",
        }}
    else:
        values.update({"av_inconsistency_logit": None, "av_available": False})
        values["branch_reasons"]["dense_av"] = "DISABLED_OR_NOT_SUPPLIED"
        values["coverage"]["dense_av_segments"] = 0
        values["temporal_evidence"] = {"dense_av": {"available": False, "evidence_type": "audio_mouth_correspondence", "aggregate": {"av_inconsistency_logit": None, "segment_count": 0}, "segments": [], "interpretation_note": "Dense AV evidence was unavailable; missing evidence is not evidence that media is real."}}
    if not any(values[key] for key in ("visual_available", "audio_available", "av_available")):
        raise RuntimeError("No usable evidence is available for fusion prediction")
    return values


def feature_order(enabled: list[str]) -> list[str]:
    result = []
    for branch, logit in (("visual", "visual_logit"), ("audio", "audio_logit"), ("dense_av", "av_inconsistency_logit")):
        if branch in enabled:
            result.extend((logit, f"{branch}_available" if branch != "dense_av" else "av_available"))
    return result


def vectorize(row: dict[str, Any], order: list[str]) -> list[float]:
    return [float(row.get(name) or 0.0) for name in order]


def normalize_features(matrix: Any, order: list[str], mean: Any | None = None, std: Any | None = None) -> tuple[Any, Any, Any]:
    import numpy as np
    logit_indices = [index for index, name in enumerate(order) if name.endswith("_logit")]
    if mean is None:
        mean, std = np.zeros(matrix.shape[1], dtype=np.float32), np.ones(matrix.shape[1], dtype=np.float32)
        if logit_indices:
            mean[logit_indices] = matrix[:, logit_indices].mean(axis=0)
            std[logit_indices] = np.maximum(matrix[:, logit_indices].std(axis=0), 1e-4)
    output = matrix.astype(np.float32, copy=True)
    if logit_indices:
        output[:, logit_indices] = (output[:, logit_indices] - mean[logit_indices]) / std[logit_indices]
    return output, mean.astype(np.float32), std.astype(np.float32)


def selection_reason(base: dict[str, Any], av: dict[str, Any]) -> tuple[str, str]:
    improvement = av["macro_f1"] - base["macro_f1"]
    protected = all(
        av["per_class"][name]["f1"] + 1e-9 >= base["per_class"][name]["f1"]
        for name in ("REAL", "AUDIO_MANIPULATION")
    )
    if improvement + 1e-9 >= 0.02 and protected:
        return "visual_audio_av", "AV improved macro-F1 by at least 0.02 without reducing protected-class F1"
    return "visual_audio", "AV did not meet the predeclared macro-F1 and protected-class F1 rule"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)): return value.item()
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def counterfactual_modality_analysis(
    evidence: dict[str, Any],
    enabled: list[str],
    score: Any,
) -> dict[str, Any]:
    """Leave one originally available branch out without rerunning extractors."""
    branch_to_fields = {
        "visual": ("visual_logit", "visual_available"),
        "audio": ("audio_logit", "audio_available"),
        "dense_av": ("av_inconsistency_logit", "av_available"),
    }
    full = score(evidence)
    interventions: dict[str, Any] = {}
    changed, stable = [], []
    for branch in enabled:
        logit_key, mask_key = branch_to_fields[branch]
        if not evidence.get(mask_key):
            continue
        masked = dict(evidence)
        masked[logit_key], masked[mask_key] = None, False
        result = score(masked)
        result["removed_evidence"] = branch
        result["class_changed"] = result["selected_class"] != full["selected_class"]
        interventions[f"without_{branch}"] = result
        (changed if result["class_changed"] else stable).append(branch)
    return {
        "method": "leave_one_available_modality_out",
        "full_evidence": full,
        "interventions": interventions,
        "summary": {
            "full_selected_class": full["selected_class"],
            "class_changed_without": changed,
            "class_stable_without": stable,
        },
        "interpretation_note": "Masked-evidence sensitivity is not formal causal attribution, forensic proof, or calibrated feature importance.",
    }
