"""Phase 3e: Generate a consolidated baseline report from all evaluation results.

Reads JSON outputs from audio, visual, and fusion evaluations and generates
a markdown report with recommendations on which branches need retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "interim" / "v2_evaluations"
OUTPUT = EVAL_DIR / "baseline_report.md"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("PRAMAAN-X V2: Baseline Report Generator", flush=True)
    print("=" * 60, flush=True)

    audio_path = args.eval_dir / "audio_baseline_external.json"
    visual_path = args.eval_dir / "visual_baseline_external.json"
    fusion_path = args.eval_dir / "fusion_baseline_external.json"

    report_lines: list[str] = []
    report_lines.append("# Pramaan-X V2: External Baseline Evaluation Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append("")

    recommendations: list[str] = []

    # Audio section
    report_lines.append("## Audio Branch Baseline")
    report_lines.append("")
    if audio_path.is_file():
        with audio_path.open() as f:
            audio_data = json.load(f)

        report_lines.append(f"Checkpoint: `{audio_data.get('checkpoint', 'N/A')}`")
        report_lines.append("")
        report_lines.append("| Dataset | Samples | AUC | Balanced Acc | Detection Rate | FPR |")
        report_lines.append("|:---|:---|:---|:---|:---|:---|")

        audio_needs_retrain = False
        for name, result in sorted(audio_data.get("evaluations", {}).items()):
            n = result.get("valid_samples", 0)
            if "metrics" in result:
                m = result["metrics"]
                auc = f"{m['roc_auc']:.3f}"
                ba = f"{m['balanced_accuracy']:.3f}"
                report_lines.append(f"| {name} | {n} | {auc} | {ba} | — | — |")
                if m["roc_auc"] < 0.85:
                    audio_needs_retrain = True
            elif "detection_rate_at_05" in result:
                dr = f"{result['detection_rate_at_05']:.3f}"
                report_lines.append(f"| {name} | {n} | — | — | {dr} | — |")
                if result["detection_rate_at_05"] < 0.5:
                    audio_needs_retrain = True
            elif "false_positive_rate_at_05" in result:
                fpr = f"{result['false_positive_rate_at_05']:.3f}"
                report_lines.append(f"| {name} | {n} | — | — | — | {fpr} |")
                if result["false_positive_rate_at_05"] > 0.3:
                    audio_needs_retrain = True
            else:
                report_lines.append(f"| {name} | {n} | error | error | — | — |")

        report_lines.append("")
        if audio_needs_retrain:
            recommendations.append("**Audio head: RETRAIN RECOMMENDED** — External AUC < 0.85 or detection rate < 50%")
        else:
            recommendations.append("**Audio head: SKIP RETRAINING** — Generalizes adequately to external data")
    else:
        report_lines.append("*Audio evaluation not yet available.*")
        report_lines.append("")

    # Visual section
    report_lines.append("## Visual Branch Baseline")
    report_lines.append("")
    if visual_path.is_file():
        with visual_path.open() as f:
            visual_data = json.load(f)

        report_lines.append(f"Checkpoint: `{visual_data.get('checkpoint', 'N/A')}`")
        report_lines.append("")
        report_lines.append("| Dataset | Samples | AUC | Balanced Acc | Detection Rate | FPR |")
        report_lines.append("|:---|:---|:---|:---|:---|:---|")

        visual_needs_retrain = False
        for name, result in sorted(visual_data.get("evaluations", {}).items()):
            n = result.get("valid_samples", 0)
            if "metrics" in result:
                m = result["metrics"]
                auc = f"{m['roc_auc']:.3f}"
                ba = f"{m['balanced_accuracy']:.3f}"
                report_lines.append(f"| {name} | {n} | {auc} | {ba} | — | — |")
                if m["roc_auc"] < 0.80:
                    visual_needs_retrain = True
            elif "detection_rate_at_05" in result:
                dr = f"{result['detection_rate_at_05']:.3f}"
                report_lines.append(f"| {name} | {n} | — | — | {dr} | — |")
            elif "false_positive_rate_at_05" in result:
                fpr = f"{result['false_positive_rate_at_05']:.3f}"
                report_lines.append(f"| {name} | {n} | — | — | — | {fpr} |")
            else:
                report_lines.append(f"| {name} | {n} | error | error | — | — |")

        report_lines.append("")
        if visual_needs_retrain:
            recommendations.append("**Visual head: RETRAIN RECOMMENDED** — External AUC < 0.80 on video data")
        else:
            recommendations.append("**Visual head: SKIP RETRAINING** — Generalizes adequately")
    else:
        report_lines.append("*Visual evaluation not yet available.*")
        report_lines.append("")

    # Fusion section
    report_lines.append("## Fusion Model Baseline")
    report_lines.append("")
    if fusion_path.is_file():
        with fusion_path.open() as f:
            fusion_data = json.load(f)

        report_lines.append(f"Enabled branches: `{fusion_data.get('enabled_branches', 'N/A')}`")
        report_lines.append("")

        for name, result in sorted(fusion_data.get("evaluations", {}).items()):
            n = result.get("evaluated_samples", 0)
            report_lines.append(f"### {name} (n={n})")
            report_lines.append("")

            if "metrics" in result:
                m = result["metrics"]
                report_lines.append(f"**Accuracy: {m['accuracy']:.3f} | Macro-F1: {m['macro_f1']:.3f}**")
                report_lines.append("")
                report_lines.append("| Class | Precision | Recall | F1 | Support |")
                report_lines.append("|:---|:---|:---|:---|:---|")
                for cls_name, cls_m in m.get("per_class", {}).items():
                    report_lines.append(f"| {cls_name} | {cls_m['precision']:.3f} | {cls_m['recall']:.3f} | {cls_m['f1']:.3f} | {cls_m['support']} |")
                report_lines.append("")

                if "confusion_matrix" in m:
                    report_lines.append("Confusion matrix (rows=actual, cols=predicted):")
                    report_lines.append("```")
                    header = "           " + "  ".join(f"{c[:6]:>6}" for c in ("REAL", "VIS", "AUD", "AV"))
                    report_lines.append(header)
                    for i, row in enumerate(m["confusion_matrix"]):
                        label = ("REAL", "VIS", "AUD", "AV")[i]
                        report_lines.append(f"  {label:>6}  " + "  ".join(f"{v:>6}" for v in row))
                    report_lines.append("```")
                    report_lines.append("")

            if "modality_coverage" in result:
                cov = result["modality_coverage"]
                report_lines.append(f"Modality coverage: visual={cov['visual_available']}, audio={cov['audio_available']}, av={cov['av_available']} / {cov['total']}")
                report_lines.append("")
    else:
        report_lines.append("*Fusion evaluation not yet available.*")
        report_lines.append("")

    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    if recommendations:
        for rec in recommendations:
            report_lines.append(f"- {rec}")
    else:
        report_lines.append("- *Run evaluations first to generate recommendations.*")
    report_lines.append("")

    report_lines.append("## Next Steps")
    report_lines.append("")
    report_lines.append("Based on the above results:")
    report_lines.append("")
    report_lines.append("1. If audio retraining recommended → `venv\\Scripts\\python.exe scripts/v2/10_train_audio_v2.py --device cpu`")
    report_lines.append("2. If visual retraining recommended → `venv\\Scripts\\python.exe scripts/v2/11_train_visual_v2.py --device cpu`")
    report_lines.append("3. Dense AV: **skip** unless a specific sync-failure mode is identified")
    report_lines.append("4. V2 fusion → `venv\\Scripts\\python.exe scripts/v2/12_train_fusion_v2.py --device cpu`")
    report_lines.append("5. Final comparison → `venv\\Scripts\\python.exe scripts/v2/13_compare_v1_v2.py --device cpu`")

    # Write report
    report_text = "\n".join(report_lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    print(f"\n  Report saved to: {args.output}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
