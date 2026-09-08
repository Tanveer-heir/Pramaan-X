"""Compare explicit V1/V2 audio reports and make the predeclared audio decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2_common import acceptance_decision, atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1-report", type=Path, required=True); p.add_argument("--v2-report", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "data/interim/v2_evaluations/v1_v2_audio_comparison.json")
    return p


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle: return json.load(handle)


def metrics(report: dict, name: str) -> dict:
    value = report.get("evaluations", {}).get(name)
    if not isinstance(value, dict): raise RuntimeError(f"Missing required evaluation: {name}")
    return value


def main() -> None:
    args = parser().parse_args(); v1, v2 = load(args.v1_report), load(args.v2_report)
    prohibited = ("MLAAD_english", "MLAAD_English", "MLAAD_overall", "MLAAD")
    for report in (v1, v2):
        names = set(report.get("evaluations", {}))
        if names.intersection(prohibited): raise RuntimeError("Final V2 comparison must not include non-Hindi or aggregate MLAAD results")
    fac, itw, holdout = metrics(v2, "FakeAVCeleb_validation"), metrics(v2, "In-the-Wild_diagnostic"), metrics(v2, "MLAAD_Hindi_adaptation_holdout")
    decision = acceptance_decision(fac, itw, holdout)
    payload = {"schema_version": "pramaan_x_v1_v2_audio_comparison_v1", "v1_report": {"path": str(args.v1_report), "sha256": sha256_file(args.v1_report)}, "v2_report": {"path": str(args.v2_report), "sha256": sha256_file(args.v2_report)}, "comparisons": {"FakeAVCeleb_validation": {"v1": metrics(v1, "FakeAVCeleb_validation"), "v2": fac}, "In-the-Wild_diagnostic": {"v1": metrics(v1, "In-the-Wild_diagnostic"), "v2": itw}, "MLAAD_Hindi_adaptation_dev": {"v1": metrics(v1, "MLAAD_Hindi_adaptation_dev"), "v2": metrics(v2, "MLAAD_Hindi_adaptation_dev")}, "MLAAD_Hindi_adaptation_holdout": {"v1": metrics(v1, "MLAAD_Hindi_adaptation_holdout"), "v2": holdout}}, "decision": decision, "next_step": "Run Fusion V2 only with the explicit accepted V2 audio checkpoint." if decision["decision"] == "V2_AUDIO_ACCEPTED" else "Retain V1 audio and do not run Fusion V2.", "limitations": ["This decision does not calibrate probabilities.", "Hindi holdout was split after V1 external results were observed, so it is not a pristine benchmark."]}
    atomic_json(args.output, payload); print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__": main()
