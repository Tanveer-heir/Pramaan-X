"""Create and audit the deterministic Hindi-only Pramaan-X V2 protocol."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from v2_common import HINDI_LANGUAGE, atomic_csv, atomic_json, deterministic_assign, read_csv, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/metadata/modern_challenge_manifest.csv")
    p.add_argument("--audio-root", type=Path, default=ROOT / "data/interim/modern_challenge/audio_embeddings/xlsr_sls_v1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/interim/v2_protocol")
    p.add_argument("--seed", type=int, default=20260907)
    return p


def main() -> None:
    args = parser().parse_args()
    rows = read_csv(args.manifest)
    needed = {"sample_id", "dataset", "language", "audio_label", "modality", "identity", "original_id"}
    if not rows or needed - set(rows[0]):
        raise RuntimeError(f"Manifest missing required fields: {sorted(needed - set(rows[0]) if rows else needed)}")
    mlaad = [row for row in rows if row["dataset"].strip().casefold() == "mlaad"]
    observed_languages = sorted({row.get("language", "").strip().casefold() for row in mlaad})
    if HINDI_LANGUAGE not in observed_languages:
        raise RuntimeError(f"Canonical Hindi value {HINDI_LANGUAGE!r} was not found; observed MLAAD values: {observed_languages}")
    hindi = [row for row in mlaad if row.get("language", "").strip().casefold() == HINDI_LANGUAGE]
    if any(row.get("audio_label") != "1" for row in hindi):
        raise RuntimeError("This V2 protocol assumes MLAAD Hindi fake-only rows; inspect labels before proceeding.")
    assignments = deterministic_assign(hindi, args.seed)
    for row in assignments:
        row["audio_cache_status"] = "usable" if (args.audio_root / f"{row['sample_id']}.npz").is_file() else "missing"
    fields = list(assignments[0])
    csv_path, json_path = args.output_dir / "mlaad_hindi_protocol.csv", args.output_dir / "mlaad_hindi_protocol_audit.json"
    atomic_csv(csv_path, assignments, fields)
    role_counts = Counter(row["role"] for row in assignments)
    group_counts = Counter((row["role"], row["source_group_field"]) for row in assignments)
    audit = {"schema_version": "pramaan_x_v2_protocol_v1", "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest), "canonical_hindi_value": HINDI_LANGUAGE, "observed_mlaad_languages": observed_languages, "hindi_total_rows": len(assignments), "usable_audio_cache_rows": sum(row["audio_cache_status"] == "usable" for row in assignments), "label_counts": dict(Counter(row["audio_label"] for row in assignments)), "modality_counts": dict(Counter(row["modality"] for row in assignments)), "generator_counts": dict(Counter(row["generator_tool"] or "unknown" for row in assignments)), "role_counts": dict(role_counts), "role_group_field_counts": {f"{role}:{field}": count for (role, field), count in group_counts.items()}, "assignments_csv": str(csv_path), "assignments_sha256": sha256_file(csv_path), "limitations": ["MLAAD Hindi is fake-only, so its holdout cannot supply AUC, specificity, or balanced accuracy.", "The V1 external baseline was observed before this split was frozen; this holdout is not a pristine never-seen benchmark.", "In-the-Wild remains diagnostic only and is not included in adaptation training.", "FakeAVCeleb calibration and test are not read by this script."], "exclusions": {"non_hindi_mlaad_rows": len(mlaad) - len(hindi)}}
    atomic_json(json_path, audit)
    print(f"Protocol: {csv_path}\nAudit: {json_path}\nHindi rows: {len(assignments)}; usable audio caches: {audit['usable_audio_cache_rows']}")


if __name__ == "__main__":
    main()
