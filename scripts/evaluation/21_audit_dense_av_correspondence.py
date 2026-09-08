"""Audit whether the accepted dense AV checkpoint is sensitive to audio-mouth pairing.

This is an audit-only entry point.  It replays historical construction unchanged,
then evaluates matched fixed offsets and exactly balanced reciprocal crossover
controls.  It never reads calibration/test manifests or raw media and never
changes caches, checkpoints, or training semantics.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAINING_PATH = ROOT / "scripts" / "training" / "19_train_dense_av_sync.py"
DENSE_SCHEMA = "av_sync_dense_v1"
VISUAL_SCHEMA = "visual_embedding_v1"
AUDIO_DESCRIPTOR = "LOG_MEL_MEAN_STD_RMS_ZCR_V1"
AUDIO_AUDIT_SCHEMA = "dense_av_correspondence_audit_v2"
AUDIO_DIM, MOUTH_DIM = 82, 768
NOMINAL_STEP, OFFSET_TOLERANCE = .25, .05
DEFAULT_CHECKPOINT = ROOT / "checkpoints/av_sync_dense/av_sync_dense_20260905T055537Z.pt"
DEFAULT_DENSE = ROOT / "data/interim/av_sync_features/dense_v1"
DEFAULT_VISUAL = ROOT / "data/interim/visual_embeddings/convnext_tiny_v1"
DEFAULT_TRAIN = ROOT / "data/metadata/splits/train.csv"
DEFAULT_VALIDATION = ROOT / "data/metadata/splits/validation.csv"
DEFAULT_OUT = ROOT / "data/interim/audits/dense_av_correspondence"

@dataclass(frozen=True)
class Source:
    sample_id: str
    identity: str
    component_id: str
    split: str
    dense_path: Path
    visual_path: Path

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Explicit accepted dense AV checkpoint. No newest-file fallback exists.")
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--dense-root", type=Path, required=True)
    p.add_argument("--visual-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mode", choices=("preflight", "smoke", "complete", "recompute"), default="preflight")
    p.add_argument("--records", type=Path, help="JSONL records for --mode recompute")
    p.add_argument("--report", type=Path, help="Original report for exact --mode recompute; defaults to the records sibling")
    p.add_argument("--seed", type=int, default=9173, help="Audit/bootstrap seed, distinct from checkpoint pairing seed")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--smoke-source-limit", type=int, default=2)
    p.add_argument("--smoke-split", choices=("train", "validation", "both"), default="train",
                   help="Smoke inference scope only; manifests are still verified without reading their caches.")
    p.add_argument("--bootstrap-repetitions", type=int, default=2000)
    p.add_argument("--device", default="auto")
    return p

def load_training() -> Any:
    spec = importlib.util.spec_from_file_location("dense_av_training_for_correspondence_audit", TRAINING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training implementation: {TRAINING_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def json_safe(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Refusing non-finite JSON value")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as h:
            temp = Path(h.name)
            json.dump(json_safe(payload), h, sort_keys=True, indent=2)
            h.write("\n"); h.flush(); os.fsync(h.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)

def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as h:
            temp = Path(h.name)
            for row in rows:
                h.write(json.dumps(json_safe(row), sort_keys=True, separators=(",", ":")) + "\n")
            h.flush(); os.fsync(h.fileno())
        os.replace(temp, path)
        return sha256(path)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)

def rows_for_manifest(path: Path, expected_split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    required = {"sample_id", "split", "semantic_class", "identity", "component_id", "linked_identities"}
    if not rows or required - set(rows[0]):
        raise RuntimeError(f"{path} missing manifest columns: {sorted(required - set(rows[0]) if rows else required)}")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"{path} has duplicate sample IDs")
    if any(r["split"] != expected_split for r in rows):
        raise RuntimeError(f"{path} does not contain only declared {expected_split} rows")
    return rows

def linked(value: str) -> set[str]:
    return {part.strip() for part in str(value).split(";") if part.strip()}

def manifest_audit(train: list[dict[str, str]], validation: list[dict[str, str]]) -> dict[str, Any]:
    t_ids, v_ids = {r["sample_id"] for r in train}, {r["sample_id"] for r in validation}
    t_identity = {r["identity"] for r in train}
    v_identity = {r["identity"] for r in validation}
    t_component = {str(r["component_id"]) for r in train}
    v_component = {str(r["component_id"]) for r in validation}
    t_links = set().union(*(linked(r["linked_identities"]) for r in train))
    v_links = set().union(*(linked(r["linked_identities"]) for r in validation))
    errors = []
    for name, common in (("sample_id", t_ids & v_ids), ("identity", t_identity & v_identity),
                         ("component_id", t_component & v_component), ("linked_identity", t_links & v_links),
                         ("train_identity_vs_validation_links", t_identity & v_links),
                         ("validation_identity_vs_train_links", v_identity & t_links)):
        if common: errors.append({"kind": name, "examples": sorted(common)[:20], "count": len(common)})
    # Check each declared component has one identity/link closure in the consumed manifests.
    component_sets: dict[str, set[str]] = {}
    for row in train + validation:
        component_sets.setdefault(str(row["component_id"]), set()).update({row["identity"], *linked(row["linked_identities"])})
    return {"train_rows": len(train), "validation_rows": len(validation),
            "train_components": sorted(t_component), "validation_components": sorted(v_component),
            "overlap_errors": errors, "component_identity_link_sets": {k: sorted(v) for k, v in component_sets.items()}}

def sources(rows: list[dict[str, str]], dense_root: Path, visual_root: Path, limit: int | None) -> list[Source]:
    real = sorted((r for r in rows if r["semantic_class"] == "REAL"), key=lambda r: r["sample_id"])
    if limit is not None: real = real[:limit]
    return [Source(r["sample_id"], r["identity"], str(r["component_id"]), r["split"],
                   dense_root / f'{r["sample_id"]}.npz', visual_root / f'{r["sample_id"]}.npz') for r in real]

def timestamp_runs(timestamps: Any, np: Any) -> list[tuple[int, int]]:
    if len(timestamps) == 0: return []
    if not np.isfinite(timestamps).all(): raise RuntimeError("non-finite timestamps")
    difference = np.diff(timestamps)
    if (difference <= 0).any(): raise RuntimeError("duplicate_or_decreasing_timestamps")
    starts = [0, *(np.flatnonzero(difference > .375) + 1).tolist()]
    return [(start, end) for start, end in zip(starts, [*starts[1:], len(timestamps)])]

def source_runs(cache: dict[str, Any], np: Any) -> list[tuple[int, int]]:
    """Split on timestamp discontinuities and unavailable cached mouth-frame indices."""
    timestamps, indices = cache["timestamps"], cache["indices"]
    timestamp_runs(timestamps, np)
    if len(indices) and (np.diff(indices) <= 0).any():
        raise RuntimeError("duplicate_or_decreasing_mouth_indices")
    breaks = np.flatnonzero((np.diff(timestamps) > .375) | (np.diff(indices) > 1)) + 1
    boundaries = [0, *breaks.tolist(), len(timestamps)]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

def cache_source(source: Source, np: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not source.dense_path.exists() or not source.visual_path.exists():
        return None, "missing_dense_or_visual_file"
    try:
        with np.load(source.dense_path, allow_pickle=False) as dense, np.load(source.visual_path, allow_pickle=False) as visual:
            need_dense = {"schema_version","sample_id","av_modality","timestamps_sec","mouth_frame_indices",
                          "audio_features","sample_rate","audio_context_samples","source_audio_samples",
                          "audio_descriptor","visual_schema_version","visual_backbone"}
            need_visual = {"schema_version","sample_id","backbone","timestamps_sec","mouth_frame_indices","mouth_embeddings"}
            if need_dense-set(dense.files) or need_visual-set(visual.files): return None, "required_cache_keys_missing"
            get = lambda d,k: d[k].item() if getattr(d[k], "ndim", 0) == 0 else d[k]
            if str(get(dense,"schema_version")) != DENSE_SCHEMA or str(get(visual,"schema_version")) != VISUAL_SCHEMA: return None, "schema_mismatch"
            if str(get(dense,"sample_id")) != source.sample_id or str(get(visual,"sample_id")) != source.sample_id: return None, "sample_id_mismatch"
            if str(get(dense,"av_modality")) != "AVAILABLE": return None, f"modality:{get(dense,'av_modality')}"
            if str(get(dense,"audio_descriptor")) != AUDIO_DESCRIPTOR: return None, "audio_descriptor_mismatch"
            if str(get(dense,"visual_schema_version")) != VISUAL_SCHEMA: return None, "dense_visual_schema_mismatch"
            audio = np.asarray(dense["audio_features"], dtype=np.float32)
            di = np.asarray(dense["mouth_frame_indices"], dtype=np.int64)
            dt = np.asarray(dense["timestamps_sec"], dtype=np.float64)
            vi = np.asarray(visual["mouth_frame_indices"], dtype=np.int64)
            vt = np.asarray(visual["timestamps_sec"], dtype=np.float64)
            vm = np.asarray(visual["mouth_embeddings"], dtype=np.float32)
            if audio.ndim != 2 or audio.shape[1] != AUDIO_DIM or len(audio) != len(di) or len(dt) != len(di): return None, "dense_shape_mismatch"
            if vm.ndim != 2 or vm.shape[1] != MOUTH_DIM or len(vm) != len(vi): return None, "visual_shape_mismatch"
            if len(di) and (np.diff(di) <= 0).any(): return None, "mouth_indices_not_strictly_ordered"
            positions = {int(frame): i for i, frame in enumerate(vi)}
            if any(int(i) not in positions for i in di): return None, "dense_index_missing_from_visual"
            mouth = vm[[positions[int(i)] for i in di]]
            corresponding = vt[di]
            if not np.allclose(dt, corresponding, rtol=0, atol=1e-5): return None, "dense_visual_timestamp_mismatch"
            timestamp_runs(dt, np)
            if not (np.isfinite(audio).all() and np.isfinite(mouth).all()): return None, "non_finite_features"
            sample_rate, context, samples = int(get(dense,"sample_rate")), int(get(dense,"audio_context_samples")), int(get(dense,"source_audio_samples"))
            if sample_rate != 16000 or context != 4000 or samples < 0: return None, "audio_context_contract_mismatch"
            return {"source":source,"audio":audio,"mouth":mouth,"timestamps":dt,"indices":di,
                    "sample_rate":sample_rate,"context":context,"samples":samples,
                    "dense_sha256":sha256(source.dense_path),"visual_sha256":sha256(source.visual_path)}, None
    except Exception as e:
        return None, f"unreadable_or_invalid_cache:{type(e).__name__}:{e}"

def padding(timestamp: float, cache: dict[str, Any]) -> dict[str, int]:
    centre = round(timestamp * cache["sample_rate"])
    start = centre - cache["context"] // 2
    end = start + cache["context"]
    left, right = max(0, -start), max(0, end-cache["samples"])
    return {"centre_sample":centre,"left_padding":left,"right_padding":right,
            "valid_samples":max(0, cache["context"]-left-right)}

def timing_ok(cache: dict[str, Any], start: int, offset: int, length: int, np: Any) -> bool:
    t = cache["timestamps"]
    base = np.diff(t[start:start+length])
    moved = np.diff(t[start+offset:start+offset+length])
    disp = t[start+offset:start+offset+length]-t[start:start+length]
    expected = offset*NOMINAL_STEP
    return bool(np.all(np.abs(base-NOMINAL_STEP)<=OFFSET_TOLERANCE) and
                np.all(np.abs(moved-NOMINAL_STEP)<=OFFSET_TOLERANCE) and
                np.all(np.abs(disp-expected)<=OFFSET_TOLERANCE))

def deltas(cache: dict[str, Any], np: Any) -> Any:
    mouth = cache["mouth"]; output = np.zeros_like(mouth)
    if len(mouth)>1: output[1:] = mouth[1:]-mouth[:-1]
    for start, _ in source_runs(cache, np): output[start] = 0
    return output

def historical_deltas(cache: dict[str, Any], np: Any) -> Any:
    """Preserve the accepted trainer's timestamp-only run and delta semantics."""
    mouth=cache["mouth"]; output=np.zeros_like(mouth)
    if len(mouth)>1: output[1:]=mouth[1:]-mouth[:-1]
    for start,_ in timestamp_runs(cache["timestamps"],np): output[start]=0
    return output

def anchors(cache: dict[str, Any], segment: int, stride: int, strict: bool, np: Any) -> list[dict[str, Any]]:
    answer=[]; d=deltas(cache,np); offsets=(-4,-2,0,2,4)
    for run_start, run_end in source_runs(cache,np):
        for a in range(run_start, run_end-segment+1, stride):
            if any(a+o<run_start or a+o+segment>run_end for o in offsets): continue
            if any(not timing_ok(cache,a,o,segment,np) for o in offsets): continue
            if strict:
                # Predecessor requirement and all endpoint audio contexts entirely observed.
                if not (a-4 > run_start and a+4+segment <= run_end): continue
                if any(padding(float(cache["timestamps"][i]),cache)["valid_samples"] != cache["context"]
                       for i in range(a-4,a+4+segment)): continue
            answer.append({"anchor_id":f'{cache["source"].sample_id}:{run_start}:{a}',"run_start":run_start,
                           "run_end":run_end,"start":a,"deltas":d})
    return answer

def record_metadata(cache: dict[str, Any], audio_start: int, mouth_start: int,
                    segment: int, run_start: int, run_end: int, stride: int) -> dict[str, Any]:
    audio_positions = list(range(audio_start, audio_start + segment))
    mouth_positions = list(range(mouth_start, mouth_start + segment))
    contexts = [padding(float(cache["timestamps"][i]), cache) for i in audio_positions]
    return {
        "split": cache["source"].split,
        "audio_start": audio_start, "mouth_start": mouth_start,
        "run_start": run_start, "run_end": run_end,
        "audio_frame_indices": [int(cache["indices"][i]) for i in audio_positions],
        "mouth_frame_indices": [int(cache["indices"][i]) for i in mouth_positions],
        "audio_timestamps_sec": [float(cache["timestamps"][i]) for i in audio_positions],
        "mouth_timestamps_sec": [float(cache["timestamps"][i]) for i in mouth_positions],
        "audio_contexts": contexts,
        "audio_padding_samples": int(sum(c["left_padding"] + c["right_padding"] for c in contexts)),
        "run_start_delta_marker": bool(mouth_start == run_start),
        "start_mod_stride": int(mouth_start % stride),
        "run_position": float((mouth_start - run_start) / max(1, run_end - run_start - segment)),
    }

def sigmoid(logits: Any, np: Any) -> Any:
    return 1/(1+np.exp(-np.clip(logits,-80,80)))

def model_from_checkpoint(checkpoint: dict[str,Any], training:Any, torch:Any, device:str) -> tuple[Any, Any, Any, dict[str,Any]]:
    required={"model_state_dict","model_config","audio_normalizer_mean","audio_normalizer_std","seed",
              "label_semantics","dense_schema_version","visual_schema_version","audio_descriptor",
              "train_manifest_sha256","validation_manifest_sha256","best_validation_metrics"}
    missing=required-set(checkpoint)
    if missing: raise RuntimeError(f"checkpoint missing keys: {sorted(missing)}")
    c=checkpoint["model_config"]
    if int(c.get("audio_dim",-1))!=AUDIO_DIM or int(c.get("mouth_dim",-1))!=MOUTH_DIM: raise RuntimeError("checkpoint model dimensions incompatible")
    if checkpoint["label_semantics"] != {"0":"TEMPORALLY_SHIFTED","1":"TIMESTAMP_ALIGNED"}: raise RuntimeError("unexpected label direction")
    if checkpoint["dense_schema_version"]!=DENSE_SCHEMA or checkpoint["visual_schema_version"]!=VISUAL_SCHEMA or checkpoint["audio_descriptor"]!=AUDIO_DESCRIPTOR: raise RuntimeError("checkpoint artifact contract incompatible")
    mean=checkpoint["audio_normalizer_mean"].detach().cpu().numpy()
    std=checkpoint["audio_normalizer_std"].detach().cpu().numpy()
    if mean.shape!=(AUDIO_DIM,) or std.shape!=(AUDIO_DIM,) or not (np_isfinite(mean) and np_isfinite(std) and (std>0).all()): raise RuntimeError("invalid checkpoint audio normalizer")
    _, Model=training.build_components(torch)
    model=Model(int(c["projection_dim"]),int(c["hidden_dim"]),float(c["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.to(device).eval()
    return model,mean,std,c

def np_isfinite(array:Any)->bool:
    try:
        import numpy as np
        return bool(np.isfinite(array).all())
    except (ImportError,TypeError,ValueError):
        return False

def score(model:Any, examples:list[dict[str,Any]], mean:Any,std:Any,torch:Any,np:Any,device:str,batch:int)->None:
    for begin in range(0,len(examples),batch):
        group=examples[begin:begin+batch]
        audio=np.stack([(x["audio"]-mean)/std for x in group]).astype(np.float32)
        mouth=np.stack([x["mouth"] for x in group]).astype(np.float32)
        delta=np.stack([x["delta"] for x in group]).astype(np.float32)
        with torch.inference_mode():
            logits=model(torch.from_numpy(audio).to(device),torch.from_numpy(mouth).to(device),torch.from_numpy(delta).to(device)).detach().cpu().numpy()
        if not np.isfinite(logits).all(): raise RuntimeError("non-finite model output")
        for item, logit in zip(group,logits):
            item["logit"]=float(logit); item["score"]=float(sigmoid(logit,np))

def fixed_metrics(labels:Any,scores:Any,threshold:float,np:Any)->dict[str,Any]:
    if len(labels)==0 or len(set(np.asarray(labels).tolist()))<2: return {"value":None,"reason":"empty_or_one_class"}
    labels=np.asarray(labels); scores=np.asarray(scores); ranks=np.empty(len(scores)); order=np.argsort(scores,kind="mergesort"); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and scores[order[j]]==scores[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    pos=int(labels.sum()); neg=len(labels)-pos
    auc=(ranks[labels==1].sum()-pos*(pos+1)/2)/(pos*neg)
    ordering=float((scores[labels==1,None]>scores[None,labels==0]).mean()+.5*(scores[labels==1,None]==scores[None,labels==0]).mean())
    descending=np.argsort(-scores,kind="mergesort"); sorted_labels=labels[descending]; sorted_scores=scores[descending]
    cumulative=np.cumsum(sorted_labels); ap=0.0; i=0
    while i<len(sorted_labels):
        j=i+1
        while j<len(sorted_labels) and sorted_scores[j]==sorted_scores[i]: j+=1
        positives=int(sorted_labels[i:j].sum())
        if positives: ap+=float(cumulative[j-1]/j)*positives/pos
        i=j
    pred=scores>=threshold
    return {"roc_auc":float(auc),"average_precision":float(ap),"balanced_accuracy_fixed":float((pred[labels==1].mean()+(~pred[labels==0]).mean())/2),"ordering_fraction":ordering,"threshold":threshold}

def aggregates(records:list[dict[str,Any]], threshold:float,np:Any)->dict[str,Any]:
    data={}
    for cohort in sorted({r["cohort"] for r in records}):
        group=[r for r in records if r["cohort"]==cohort]
        labels=np.asarray([r["label"] for r in group]); scores=np.asarray([r["score"] for r in group])
        data[cohort]={"count":len(group),**fixed_metrics(labels,scores,threshold,np)}
    return data

def historical_metrics(records:list[dict[str,Any]],np:Any)->dict[str,Any]:
    if not records: return {"value":None,"reason":"empty"}
    labels=np.asarray([r["label"] for r in records]); scores=np.asarray([r["score"] for r in records])
    result=fixed_metrics(labels,scores,.5,np); best=-1.0; selected=None
    for threshold in np.unique(scores):
        predictions=scores>=threshold
        value=float((predictions[labels==1].mean()+(~predictions[labels==0]).mean())/2)
        if value>best: best,selected=value,float(threshold)
    result["balanced_accuracy"]=best; result["validation_threshold"]=selected
    result.pop("balanced_accuracy_fixed",None); result.pop("threshold",None); result.pop("ordering_fraction",None)
    return result

def attach_aligned(records:list[dict[str,Any]])->None:
    aligned={(r["cohort"],r["anchor_id"]):r for r in records
             if r["cohort"].startswith("matched") and r["label"]==1}
    for row in records:
        key=(row["cohort"],row.get("paired_anchor_id"))
        if key in aligned:
            row["aligned_score"]=aligned[key]["score"]; row["aligned_logit"]=aligned[key]["logit"]

def matched_balanced_metrics(records:list[dict[str,Any]],threshold:float,np:Any)->dict[str,Any]:
    result={}
    for cohort in ("matched_boundary","matched_strict"):
        positives={(r["anchor_id"]):r for r in records if r["cohort"]==cohort and r["label"]==1}
        for offset in (-4,-2,2,4):
            negatives=[r for r in records if r["cohort"]==cohort and r["label"]==0 and r["offset_frames"]==offset]
            paired=[positives[r["paired_anchor_id"]] for r in negatives if r["paired_anchor_id"] in positives]
            group=[*paired,*negatives]
            result[f"{cohort}:{offset}"]={"anchors":len(negatives),
                **fixed_metrics(np.asarray([r["label"] for r in group]),np.asarray([r["score"] for r in group]),threshold,np)}
    return result

def paired_summary(records:list[dict[str,Any]], np:Any)->dict[str,Any]:
    attach_aligned(records)
    result={}
    for cohort in ("matched_boundary","matched_strict"):
        for offset in (-4,-2,2,4):
            a=[r for r in records if r["cohort"]==cohort and r.get("offset_frames")==offset and r["label"]==0]
            diff=[r["aligned_score"]-r["score"] for r in a]
            logit=[r["aligned_logit"]-r["logit"] for r in a]
            result[f"{cohort}:{offset}"]={"anchors":len(a),
                "mean_aligned_minus_shifted_score":float(np.mean(diff)) if diff else None,
                "mean_aligned_minus_shifted_logit":float(np.mean(logit)) if logit else None,
                "ordering_fraction":float(np.mean([d>0 for d in diff])+0.5*np.mean([d==0 for d in diff])) if diff else None,
                "measured_displacement_seconds": [float(np.mean(r["actual_offsets_sec"])) for r in a] if a else []}
    return result

def clustered_interval(units:list[dict[str,Any]],np:Any,reps:int,seed:int)->dict[str,Any]:
    identities=sorted({r["identity"] for r in units})
    if len(identities)<2: return {"value":None,"reason":"insufficient_identity_clusters","clusters":len(identities)}
    by={i:[r for r in units if r["identity"]==i] for i in identities}; rng=np.random.default_rng(seed); values=[]
    for _ in range(reps):
        sample=rng.choice(identities,size=len(identities),replace=True)
        effects=[]
        for identity in sample:
            source_values=[]
            for source in sorted({r["source"] for r in by[identity]}):
                source_values.append(float(np.mean([r["effect"] for r in by[identity] if r["source"]==source])))
            effects.append(float(np.mean(source_values)))
        values.append(float(np.mean(effects)))
    return {"clusters":len(identities),"repetitions":reps,"lower_95":float(np.quantile(values,.025)),
            "median":float(np.quantile(values,.5)),"upper_95":float(np.quantile(values,.975))}

def bootstrap_identity(records:list[dict[str,Any]], np:Any, reps:int,seed:int)->dict[str,Any]:
    attach_aligned(records); output={}
    for cohort in ("matched_boundary","matched_strict"):
        for offset in (-4,-2,2,4):
            rows=[r for r in records if r["cohort"]==cohort and r["label"]==0 and r["offset_frames"]==offset]
            for kind in ("logit","score","ordering"):
                units=[]
                for r in rows:
                    effect=(r["aligned_logit"]-r["logit"] if kind=="logit" else
                            r["aligned_score"]-r["score"] if kind=="score" else
                            1.0 if r["aligned_score"]>r["score"] else .5 if r["aligned_score"]==r["score"] else 0.0)
                    units.append({"identity":r["identity"],"source":r["source"],"effect":effect})
                output[f"{cohort}:{offset}:{kind}"]=clustered_interval(units,np,reps,seed)
        for magnitude in (2,4):
            by_anchor={}
            for r in records:
                if r["cohort"]==cohort and r["label"]==0 and abs(r["offset_frames"])==magnitude:
                    by_anchor.setdefault(r["paired_anchor_id"],{})[r["offset_frames"]]=r
            units=[]
            for pair in by_anchor.values():
                if {-magnitude,magnitude}<=set(pair):
                    negative,positive=pair[-magnitude],pair[magnitude]
                    units.append({"identity":positive["identity"],"source":positive["source"],
                                  "effect":positive["logit"]-negative["logit"]})
            output[f"{cohort}:direction:{magnitude}:logit"]=clustered_interval(units,np,reps,seed)
    return output


def crossover_interactions(records:list[dict[str,Any]], np:Any)->dict[str,Any]:
    blocks:dict[str,list[dict[str,Any]]]={}
    for row in records:
        if row["cohort"]=="crossover": blocks.setdefault(row["block_id"],[]).append(row)
    values=[]; score_values=[]; ordering=[]; units=[]
    for block, rows in blocks.items():
        by={row["role"]:row for row in rows}
        if set(by)!={"aa","bb","ab","ba"}: raise RuntimeError(f"invalid crossover block: {block}")
        interaction=(by["aa"]["logit"]+by["bb"]["logit"]-by["ab"]["logit"]-by["ba"]["logit"])/2
        values.append(float(interaction))
        units.append({"identity":rows[0]["identity"],"source":rows[0]["source"],"effect":float(interaction)})
        score_values.append(float((by["aa"]["score"]+by["bb"]["score"]-by["ab"]["score"]-by["ba"]["score"])/2))
        ordering.append(float(((by["aa"]["score"]>by["ab"]["score"])+(by["bb"]["score"]>by["ba"]["score"]))/2))
    if not values: return {"value":None,"reason":"no_eligible_crossover_blocks","blocks":0}
    return {"blocks":len(values),"mean_logit_interaction":float(np.mean(values)),
            "mean_score_contrast":float(np.mean(score_values)),
            "within_block_ordering_fraction":float(np.mean(ordering)),
            "identity_effects":units,
            "definition":"[z(Aa,Ma)+z(Ab,Mb)-z(Aa,Mb)-z(Ab,Ma)]/2"}

def build_cohort(cache:dict[str,Any], cohort:str, segment:int,stride:int,np:Any)->list[dict[str,Any]]:
    out=[]
    for row in anchors(cache,segment,stride,cohort=="matched_strict",np):
        a=row["start"]
        aligned={"audio":cache["audio"][a:a+segment],"mouth":cache["mouth"][a:a+segment],"delta":row["deltas"][a:a+segment],
                 "label":1,"cohort":cohort,"source":cache["source"].sample_id,"identity":cache["source"].identity,
                 "component_id":cache["source"].component_id,"anchor_id":row["anchor_id"],"offset_frames":0,
                 **record_metadata(cache,a,a,segment,row["run_start"],row["run_end"],stride)}
        out.append(aligned)
        for offset in (-4,-2,2,4):
            b=a+offset
            actual=(cache["timestamps"][b:b+segment]-cache["timestamps"][a:a+segment]).tolist()
            out.append({**aligned,"mouth":cache["mouth"][b:b+segment],"delta":row["deltas"][b:b+segment],"label":0,
                        "offset_frames":offset,"actual_offsets_sec":[float(x) for x in actual],"paired_anchor_id":row["anchor_id"],
                        **record_metadata(cache,a,b,segment,row["run_start"],row["run_end"],stride)})
    return out

def crossover(cache:dict[str,Any], segment:int,np:Any,stride:int=4)->list[dict[str,Any]]:
    """Build unique reciprocal blocks from the exact strict matched-anchor population."""
    d=deltas(cache,np); associations:dict[tuple[Any,...],dict[str,Any]]={}
    for anchor in anchors(cache,segment,stride,True,np):
        a=anchor["start"]
        for signed_offset in (-4,-2,2,4):
            b=a+signed_offset; left,right=sorted((a,b))
            key=(cache["source"].sample_id,anchor["run_start"],left,right)
            item=associations.setdefault(key,{"anchor_ids":[],"signed_offsets":set(),"run_end":anchor["run_end"]})
            item["anchor_ids"].append(anchor["anchor_id"]); item["signed_offsets"].add(signed_offset)
    result=[]
    for key,item in sorted(associations.items()):
        _,run_start,a,b=key; block_id=":".join(map(str,key))
        base={"cohort":"crossover","source":cache["source"].sample_id,"identity":cache["source"].identity,
              "component_id":cache["source"].component_id,"block_id":block_id,"a":a,"b":b,
              "offset_frames":b-a,"matched_anchor_ids":sorted(set(item["anchor_ids"])),
              "matched_signed_offsets":sorted(item["signed_offsets"])}
        for label,audio_start,mouth_start,role in ((1,a,a,"aa"),(1,b,b,"bb"),(0,a,b,"ab"),(0,b,a,"ba")):
            result.append({**base,"label":label,"role":role,"audio":cache["audio"][audio_start:audio_start+segment],
                           "mouth":cache["mouth"][mouth_start:mouth_start+segment],"delta":d[mouth_start:mouth_start+segment],
                           **record_metadata(cache,audio_start,mouth_start,segment,run_start,item["run_end"],stride)})
    return result

def compact(record:dict[str,Any])->dict[str,Any]:
    keep=("cohort","split","source","identity","component_id","anchor_id","paired_anchor_id","block_id","role","label",
          "offset_frames","actual_offsets_sec","logit","score","audio_start","mouth_start","run_start","run_end",
          "audio_frame_indices","mouth_frame_indices","audio_timestamps_sec","mouth_timestamps_sec","audio_contexts",
          "audio_padding_samples","run_start_delta_marker","start_mod_stride","run_position","matched_anchor_ids","matched_signed_offsets")
    return {key:record[key] for key in keep if key in record}

def legacy(cache:dict[str,Any], segment:int,stride:int,pairing_seed:int,np:Any)->list[dict[str,Any]]:
    d=historical_deltas(cache,np); output=[]
    for rs,re in timestamp_runs(cache["timestamps"],np):
        for a in range(rs,re-segment+1,stride):
            valid=[o for m in (2,4) for o in (-m,m) if rs<=a+o and a+o+segment<=re]
            if not valid: continue
            offset=hash_choice(valid,f"{cache['source'].sample_id}:{a}",pairing_seed)
            base={"cohort":"legacy_replay","source":cache["source"].sample_id,"identity":cache["source"].identity,"component_id":cache["source"].component_id,
                  "anchor_id":f"{cache['source'].sample_id}:{rs}:{a}","audio":cache["audio"][a:a+segment],"label":1,"offset_frames":0,
                  **record_metadata(cache,a,a,segment,rs,re,stride)}
            output.append({**base,"mouth":cache["mouth"][a:a+segment],"delta":d[a:a+segment]})
            b=a+offset
            output.append({**base,"mouth":cache["mouth"][b:b+segment],"delta":d[b:b+segment],"label":0,"offset_frames":offset,
                           "actual_offsets_sec":[float(x) for x in cache["timestamps"][b:b+segment]-cache["timestamps"][a:a+segment]],
                           **record_metadata(cache,a,b,segment,rs,re,stride)})
    return output

def hash_choice(values:list[int],key:str,seed:int)->int:
    digest=hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return values[int.from_bytes(digest[:8],"big")%len(values)]

def census(records:list[dict[str,Any]], np:Any)->dict[str,Any]:
    output={}
    for label in (0,1):
        r=[x for x in records if x["label"]==label]
        positions=[x["run_position"] for x in r]; padding_counts=[x["audio_padding_samples"] for x in r]
        output[str(label)]={"count":len(r),
            "offset_counts":{str(o):sum(x["offset_frames"]==o for x in r) for o in (-4,-2,0,2,4)},
            "start_mod_stride_counts":{str(v):sum(x["start_mod_stride"]==v for x in r) for v in sorted({x["start_mod_stride"] for x in r})},
            "run_start_delta_markers":sum(x["run_start_delta_marker"] for x in r),
            "records_with_audio_padding":sum(v>0 for v in padding_counts),
            "mean_audio_padding_samples":float(np.mean(padding_counts)) if r else None,
            "mean_run_position":float(np.mean(positions)) if r else None}
    return output

def git_state()->dict[str,Any]:
    try:
        sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
        dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip())
        return {"sha":sha,"dirty":dirty}
    except (FileNotFoundError,subprocess.CalledProcessError):
        return {"sha":None,"dirty":None,"reason":"git_state_unavailable"}

def summarize(records:list[dict[str,Any]],threshold:float,np:Any,reps:int,seed:int,
              saved_metrics:dict[str,Any]|None,mode:str)->dict[str,Any]:
    attach_aligned(records)
    legacy={split:[r for r in records if r["cohort"]=="legacy_replay" and r.get("split")==split]
            for split in ("train","validation")}
    legacy_metrics={split:historical_metrics(rows,np) for split,rows in legacy.items()}
    reproduction={"available":False,"reason":"validation_legacy_records_unavailable"}
    if legacy["validation"] and saved_metrics:
        comparable=("roc_auc","average_precision","balanced_accuracy")
        deltas={key:abs(float(legacy_metrics["validation"][key])-float(saved_metrics[key])) for key in comparable}
        reproduction={"available":True,"absolute_deltas":deltas,"tolerance":1e-6,
                      "passed":all(value<=1e-6 for value in deltas.values())}
    cross=crossover_interactions(records,np)
    cross_units=cross.pop("identity_effects",[]) if "identity_effects" in cross else []
    cohorts={name:[r for r in records if r["cohort"]==name]
             for name in ("legacy_replay","matched_boundary","matched_strict","crossover")}
    retention={name:{"records":len(rows),"sources":len({r["source"] for r in rows}),
                     "identities":len({r["identity"] for r in rows})} for name,rows in cohorts.items()}
    strict_offsets=[r for r in cohorts["matched_strict"] if r["label"]==0]
    signs={offset:[r["aligned_logit"]-r["logit"] for r in strict_offsets if r["offset_frames"]==offset]
           for offset in (-4,-2,2,4)}
    if mode!="complete": scientific={"status":"INCONCLUSIVE_SMOKE_ONLY","reason":"Smoke cohorts cannot support a branch decision."}
    elif not strict_offsets or cross.get("blocks",0)==0:
        scientific={"status":"INCONCLUSIVE","reason":"No eligible strict matched or crossover population."}
    elif all(values and float(np.mean(values))>0 for values in signs.values()) and cross.get("mean_logit_interaction",0)>0:
        scientific={"status":"EVIDENCE_SUPPORTS_CORRESPONDENCE_SENSITIVITY",
                    "reason":"All signed strict-offset means and the reciprocal crossover logit interaction are positive."}
    elif cross.get("mean_logit_interaction",0)<=0:
        scientific={"status":"EVIDENCE_SUPPORTS_SHORTCUT_CONCERN",
                    "reason":"Balanced reciprocal crossover lacks a positive mean logit interaction."}
    else: scientific={"status":"INCONCLUSIVE","reason":"Matched directions and reciprocal interaction do not form a consistent pattern."}
    return {
        "legacy_replay":{"construction_census":census(cohorts["legacy_replay"],np),
                         "metrics_by_split":legacy_metrics,"saved_metric_reproduction":reproduction},
        "matched":{"balanced_metrics_by_offset":matched_balanced_metrics(records,threshold,np),
                   "paired_effects":paired_summary(records,np)},
        "crossover":{"metrics":aggregates(cohorts["crossover"],threshold,np),"interaction":cross,
                     "interaction_bootstrap":clustered_interval(cross_units,np,reps,seed),
                     "balance_by_construction":"Each exact endpoint audio and mouth-plus-delta sequence occurs once aligned and once crossed."},
        "clustered_bootstrap":bootstrap_identity(records,np,reps,seed),
        "retention":retention,"scientific_interpretation":scientific}

def preflight(args:Any,training:Any,np:Any,torch:Any)->tuple[dict[str,Any],dict[str,Any],list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
    train,valid=rows_for_manifest(args.train_manifest,"train"),rows_for_manifest(args.validation_manifest,"validation")
    separation=manifest_audit(train,valid)
    if separation["overlap_errors"]: raise RuntimeError(f"manifest overlap: {separation['overlap_errors']}")
    try: ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    except Exception as exc: raise RuntimeError(f"safe checkpoint load failed: {exc}") from exc
    if sha256(args.train_manifest)!=ckpt.get("train_manifest_sha256") or sha256(args.validation_manifest)!=ckpt.get("validation_manifest_sha256"): raise RuntimeError("checkpoint manifest hashes mismatch")
    device=training.resolve_device(torch,args.device)
    model,mean,std,config=model_from_checkpoint(ckpt,training,torch,device)
    train_limit=args.smoke_source_limit if args.mode=="smoke" and args.smoke_split in {"train","both"} else None
    validation_limit=args.smoke_source_limit if args.mode=="smoke" and args.smoke_split in {"validation","both"} else None
    ts=sources(train,args.dense_root,args.visual_root,train_limit) if args.mode!="smoke" or args.smoke_split in {"train","both"} else []
    vs=sources(valid,args.dense_root,args.visual_root,validation_limit) if args.mode!="smoke" or args.smoke_split in {"validation","both"} else []
    valid_caches=[]; exclusions=[]
    for split,group in (("train",ts),("validation",vs)):
        for source in group:
            cache,reason=cache_source(source,np)
            if reason: exclusions.append({"split":split,"sample_id":source.sample_id,"reason":reason})
            else: valid_caches.append((split,cache))
    report={"checkpoint_sha256":sha256(args.checkpoint),"checkpoint_path":str(args.checkpoint),
            "checkpoint_model_config":config,"checkpoint_pairing_seed":int(ckpt["seed"]),
            "checkpoint_validation_threshold":float(ckpt["best_validation_metrics"]["validation_threshold"]),
            "checkpoint_saved_validation_metrics":json_safe(ckpt["best_validation_metrics"]),
            "checkpoint_pairing_strategy":ckpt.get("pairing_strategy"),
            "manifest_sha256":{"train":sha256(args.train_manifest),"validation":sha256(args.validation_manifest)},
            "manifest_audit":separation,"device_requested":args.device,"device_actual":device,
            "feature_files":[{"sample_id":c["source"].sample_id,"dense_sha256":c["dense_sha256"],"visual_sha256":c["visual_sha256"]} for _,c in valid_caches],
            "exclusions":exclusions,"known_provenance_limitation":"Present-day cache hashes do not establish historical raw-media provenance. The checkpoint does not bind every historical feature file or context configuration."}
    return report,ckpt,[c for s,c in valid_caches if s=="train"],[c for s,c in valid_caches if s=="validation"],{"model":model,"mean":mean,"std":std,"device":device,"config":config}

def run(args:Any)->dict[str,Any]:
    training=load_training()
    if args.mode=="recompute":
        try: import numpy as np
        except ImportError as exc: raise RuntimeError("NumPy is required for report recomputation") from exc
        if args.records is None: raise ValueError("--records required for recompute")
        records=[json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
        report_path=args.report or args.records.with_name("report.json")
        if not report_path.exists(): raise FileNotFoundError("Original report required for exact recomputation")
        original=json.loads(report_path.read_text(encoding="utf-8"))
        summary=summarize(records,float(original["checkpoint_validation_threshold"]),np,
                          int(original["statistical_config"]["bootstrap_repetitions"]),
                          int(original["audit_seed"]),original.get("checkpoint_saved_validation_metrics"),
                          str(original["mode"]))
        comparable=("legacy_replay","matched","crossover","clustered_bootstrap","retention","scientific_interpretation")
        matches=all(json_safe(summary[key])==original.get(key) for key in comparable)
        return {"schema_version":AUDIO_AUDIT_SCHEMA,"run_id":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "utc_timestamp":datetime.now(timezone.utc).isoformat(),"mode":"recompute",
                "execution_status":"COMPLETE_REPORT_ONLY","records_sha256":sha256(args.records),
                "source_report":str(report_path),"aggregate_reproduction_passed":matches,**summary}
    runtime=training.require_runtime(); np,torch=runtime["np"],runtime["torch"]
    if args.batch_size<1 or args.bootstrap_repetitions<1: raise ValueError("positive batch and bootstrap settings required")
    report,ckpt,train,val,state=preflight(args,training,np,torch)
    report.update({"schema_version":AUDIO_AUDIT_SCHEMA,"run_id":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),"utc_timestamp":datetime.now(timezone.utc).isoformat(),
                   "audit_seed":args.seed,"mode":args.mode,"execution_status":"COMPLETE_PREFLIGHT","code":git_state(),
                   "statistical_config":{"bootstrap_repetitions":args.bootstrap_repetitions,"cluster":"identity","source_aggregation":"within_identity_first"},
                   "temporal_contract":{"offset_sec_definition":"mouth_timestamp_minus_audio_timestamp","nominal_step_sec":NOMINAL_STEP,
                     "tolerance_sec":OFFSET_TOLERANCE,"segment_centres_span_sec":(int(state["config"]["segment_frames"])-1)*NOMINAL_STEP,
                     "nominal_complete_context_support_sec":int(state["config"]["segment_frames"])*NOMINAL_STEP},
                   "execution_status":"COMPLETE_PREFLIGHT","software":{"python":platform.python_version(),"torch":getattr(torch,"__version__","unknown"),"numpy":getattr(np,"__version__","unknown")}})
    if args.mode=="preflight": return report
    all_records=[]
    for cache in train+val:
        all_records.extend(legacy(cache,int(state["config"]["segment_frames"]),int(state["config"]["stride_frames"]),int(ckpt["seed"]),np))
        all_records.extend(build_cohort(cache,"matched_boundary",int(state["config"]["segment_frames"]),int(state["config"]["stride_frames"]),np))
        all_records.extend(build_cohort(cache,"matched_strict",int(state["config"]["segment_frames"]),int(state["config"]["stride_frames"]),np))
        all_records.extend(crossover(cache,int(state["config"]["segment_frames"]),np,int(state["config"]["stride_frames"])))
    if not all_records:
        report["execution_status"]="FAILED_NO_USABLE_RECORDS"; return report
    if state["device"].startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    inference_started=perf_counter()
    score(state["model"],all_records,state["mean"],state["std"],torch,np,state["device"],args.batch_size)
    report["inference"]={"batch_size":args.batch_size,"records":len(all_records),
        "seconds":perf_counter()-inference_started,
        "cuda_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if state["device"].startswith("cuda") else None}
    attach_aligned(all_records)
    compact_records=[compact(r) for r in all_records]
    report["execution_status"]="SMOKE_ONLY" if args.mode=="smoke" else "COMPLETE"
    report["selection_counts"]={"train_sources":len(train),"validation_sources":len(val),"records":len(compact_records)}
    report.update(summarize(all_records,report["checkpoint_validation_threshold"],np,args.bootstrap_repetitions,args.seed,
                            report["checkpoint_saved_validation_metrics"],args.mode))
    report["limitations"]=["Validation is one connected identity component; identity resampling does not establish unseen-component uncertainty.",
        "Linked identities may retain dependence. Validation has been repeatedly used for selection, so intervals do not remove selection bias.",
        "This audit cannot prove phonetic synchronization. Overlapping sequences, silence, repeated articulation, native AV offsets, and joint nuisance relationships remain.",
        "No calibration, test, fusion, or production acceptance conclusion follows."]
    report["_records"]=compact_records
    return report

def main()->None:
    args=parser().parse_args()
    outcome=run(args); records=outcome.pop("_records",[])
    run_dir=args.output_dir/outcome["run_id"]; run_dir.mkdir(parents=True,exist_ok=False)
    records_path=run_dir/"records.jsonl"; record_hash=atomic_jsonl(records_path,records)
    outcome["records_path"]=str(records_path); outcome["records_sha256"]=record_hash
    report_path=run_dir/"report.json"; atomic_json(report_path,outcome)
    completion={"schema_version":AUDIO_AUDIT_SCHEMA,"report":"report.json","records":"records.jsonl",
                "sha256":{"report":sha256(report_path),"records":record_hash},"execution_status":outcome["execution_status"]}
    atomic_json(run_dir/"completion.json",completion)
    print(json.dumps(json_safe({**outcome,"report_path":str(report_path),"completion_path":str(run_dir/"completion.json")}),indent=2,sort_keys=True))
    if outcome["execution_status"].startswith("FAILED"): raise SystemExit(1)
if __name__=="__main__": main()
