"""Generate a self-contained HTML investigator evidence report.

The report is a presentation layer over an existing prediction JSON. It does
not rerun inference, assign new thresholds, or convert model scores into legal
or calibrated forensic conclusions.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterable
import html
import json
import math
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.explain.render_av_timeline import render_av_timeline
from scripts.explain.render_counterfactual import render_counterfactual_chart, summarize_counterfactual
from scripts.explain.render_image_overlay import render_image_overlay


VIDEO_LIMITATIONS = [
    "Softmax values are uncalibrated model scores, not forensic certainty or authentic/manipulated probabilities.",
    "This result is evidence for investigator review, not an autonomous legal conclusion.",
    "AV correspondence evidence does not establish authenticity. Fully synthetic media may still be synchronized.",
    "Missing evidence is not evidence that the media is authentic.",
    "Counterfactual masking measures decision sensitivity and is not formal causal attribution.",
    "Source tracing, provenance reconstruction, and dissemination attribution are separate subsystems unless explicitly integrated.",
]
IMAGE_LIMITATIONS = [
    "Assessment is produced by a multimodal language model and is not a calibrated forensic probability.",
    "A region box, when present, is an approximate model-indicated region and not forensic localization.",
    "Missing EXIF or metadata does not imply manipulation.",
    "The result should be reviewed with independent forensic and provenance evidence.",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prediction", type=Path, required=True, help="Existing Pramaan-X prediction JSON.")
    result.add_argument("--output", type=Path, required=True, help="HTML report path.")
    result.add_argument("--assets-dir", type=Path, help="Optional directory for generated PNG evidence artifacts.")
    result.add_argument("--image", type=Path, help="Optional source image for a defensible region overlay.")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Prediction JSON contains non-finite number: {value}")


def load_prediction(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Prediction JSON must contain an object")
    return value


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return f"{number:.{digits}f}" if number is not None else "NOT_AVAILABLE"


def _basename(value: Any) -> str:
    if not value:
        return "NOT_AVAILABLE"
    return Path(str(value)).name


def _inline_image(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_html = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table>"


def _list(items: Iterable[Any]) -> str:
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _availability(payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    evidence = payload.get("branch_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    result = []
    for label, key in (("Visual", "visual_available"), ("Audio", "audio_available"), ("Dense AV", "av_available")):
        available = evidence.get(key)
        status = "AVAILABLE" if available is True else "NOT_AVAILABLE"
        reason = evidence.get("branch_reasons", {}).get(key if key != "av_available" else "dense_av") if isinstance(evidence.get("branch_reasons"), dict) else None
        result.append((label, status, reason or ""))
    return result


def _branch_rows(payload: dict[str, Any]) -> list[list[str]]:
    evidence = payload.get("branch_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    branches = (
        ("Visual", "visual_logit", "visual_available", "visual"),
        ("Audio", "audio_logit", "audio_available", "audio"),
        ("Dense AV", "av_inconsistency_logit", "av_available", "dense_av"),
    )
    rows = []
    reasons = evidence.get("branch_reasons") if isinstance(evidence.get("branch_reasons"), dict) else {}
    for label, value_key, availability_key, reason_key in branches:
        available = evidence.get(availability_key) is True
        value = _fmt(evidence.get(value_key)) if available else "NOT_AVAILABLE"
        rows.append([_escape(label), _escape("AVAILABLE" if available else "NOT_AVAILABLE"), _escape(value), _escape(reasons.get(reason_key) or "")])
    return rows


def _provenance_rows(payload: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    raw = payload.get("raw_video")
    if isinstance(raw, dict):
        if raw.get("path"):
            rows.append(["Input filename", _escape(_basename(raw.get("path")))])
        for key in ("sha256", "bytes"):
            if raw.get(key) is not None:
                rows.append([f"Input {key}", _escape(raw[key])])
    image_input = payload.get("input")
    if isinstance(image_input, dict):
        for key in ("filename", "sha256", "format", "width", "height", "size_bytes"):
            if image_input.get(key) is not None:
                rows.append([f"Input {key}", _escape(image_input[key])])
    checkpoint = payload.get("checkpoint_provenance")
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if key == "accepted_branch_checkpoints" and isinstance(value, dict):
                for branch, details in value.items():
                    if isinstance(details, dict):
                        path = _basename(details.get("path")) if details.get("path") else "NOT_AVAILABLE"
                        digest = details.get("sha256") or details.get("digest") or "NOT_AVAILABLE"
                        rows.append([f"{branch} checkpoint", _escape(f"{path}; SHA-256: {digest}")])
                    else:
                        rows.append([f"{branch} checkpoint", _escape(_basename(details))])
            elif key in {"fusion_checkpoint", "checkpoint"}:
                rows.append(["Fusion checkpoint", _escape(_basename(value))])
            elif isinstance(value, (str, int, float, bool)):
                rows.append([key.replace("_", " ").title(), _escape(value)])
    if not rows:
        rows.append(["Provenance", "NOT_AVAILABLE_IN_PREDICTION"])
    return rows


def _image_findings(payload: dict[str, Any]) -> str:
    findings = payload.get("visual_findings")
    if not isinstance(findings, list) or not findings:
        return "<p>No structured visual findings were returned.</p>"
    rows = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        rows.append([
            _escape(finding.get("category", "")),
            _escape(finding.get("severity", "")),
            _escape(finding.get("region", "")),
            _escape(finding.get("finding", "")),
        ])
    return _table(("Category", "Severity", "Region", "Observed explanation"), rows)


def _safe_render(render_fn: Any, payload: dict[str, Any], path: Path) -> tuple[Path | None, str | None]:
    try:
        return render_fn(payload, path), None
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return None, f"Renderer unavailable: {type(exc).__name__}: {exc}"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _html_document(payload: dict[str, Any], assets: dict[str, Path | None], renderer_notes: list[str]) -> str:
    is_image = payload.get("media_type") == "image" or str(payload.get("schema_version", "")).startswith("pramaan_x_image")
    selected = payload.get("selected_class") or payload.get("assessment", {}).get("label") or "NOT_AVAILABLE"
    score_map = payload.get("softmax_probabilities")
    if not isinstance(score_map, dict):
        score_map = payload.get("class_logits") if isinstance(payload.get("class_logits"), dict) else {}
    score_rows = [[_escape(name), _escape(_fmt(value))] for name, value in score_map.items()]
    if not score_rows:
        score_rows = [["NOT_AVAILABLE", "NOT_AVAILABLE"]]
    input_name = "NOT_AVAILABLE"
    if isinstance(payload.get("raw_video"), dict):
        input_name = _basename(payload["raw_video"].get("path"))
    elif isinstance(payload.get("input"), dict):
        input_name = payload["input"].get("filename") or input_name
    availability_rows = [[_escape(name), _escape(status), _escape(reason)] for name, status, reason in _availability(payload)]
    html_sections = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Pramaan-X Investigator Evidence Report</title><style>"
        "body{font-family:Arial,sans-serif;color:#232a34;background:#f4f6f8;margin:0;line-height:1.45}"
        ".page{max-width:1120px;margin:0 auto;padding:34px 24px 60px}.hero{background:#16263f;color:white;padding:28px 32px;border-radius:14px}"
        "h1{margin:0 0 8px;font-size:30px}h2{margin-top:34px;color:#16263f;border-bottom:2px solid #d8e0e8;padding-bottom:8px}"
        "h3{color:#16263f}p.note,.note{background:#fff7e8;border-left:4px solid #d67e28;padding:12px 15px}"
        "table{width:100%;border-collapse:collapse;background:white;margin:14px 0 18px}th,td{padding:10px 12px;border:1px solid #d8e0e8;text-align:left;vertical-align:top}th{background:#e9eef5;color:#16263f}"
        ".verdict{font-size:24px;font-weight:bold}.muted{color:#5b6777}.chart{max-width:100%;height:auto;border:1px solid #d8e0e8;background:white;border-radius:8px}"
        ".badge{display:inline-block;padding:4px 9px;border-radius:999px;background:#e9eef5;font-size:13px}.available{color:#1e7944;font-weight:bold}.unavailable{color:#a13d3d;font-weight:bold}"
        "</style></head><body><main class=\"page\">",
        f"<section class=\"hero\"><h1>Pramaan-X Investigator Evidence Report</h1><div>{_escape(input_name)}</div><div class=\"verdict\">Selected model assessment: {_escape(selected)}</div></section>",
        "<p class=\"note\">This report presents model evidence for investigator review. It is not a forensic conclusion, calibrated certainty statement, or proof of manipulation.</p>",
        "<h2>A. Case summary</h2>",
        f"<p><strong>Media type:</strong> {_escape('image' if is_image else 'video')}<br><strong>Selected model class/label:</strong> {_escape(selected)}<br><strong>Schema:</strong> {_escape(payload.get('schema_version', 'NOT_AVAILABLE'))}</p>",
        "<h2>B. Model score</h2>",
        "<p class=\"muted\">Scores below retain the semantics of the prediction artifact. They are not presented as calibrated authenticity probabilities.</p>",
        _table(("Class or label", "Model score"), score_rows),
        "<h2>C. Evidence availability</h2>",
        _table(("Evidence channel", "Status", "Reason"), availability_rows),
    ]
    if is_image:
        html_sections.extend([
            "<h2>D. Image visual findings</h2>",
            _image_findings(payload),
        ])
        overlay = assets.get("image_overlay")
        if overlay:
            html_sections.extend([
                "<h3>Optional region overlay</h3>",
                f"<p class=\"muted\">Only explicitly normalized model-indicated coordinates are shown. Text-only regions are not converted into boxes.</p><img class=\"chart\" src=\"{_escape(_inline_image(overlay))}\" alt=\"Approximate model-indicated image regions\">",
            ])
        else:
            html_sections.append("<p class=\"muted\">No defensible normalized region coordinates were available, so no overlay was generated.</p>")
    else:
        html_sections.extend([
            "<h2>D. Branch evidence</h2>",
            _table(("Branch", "Availability", "Evidence value", "Reason"), _branch_rows(payload)),
            "<h2>E. Counterfactual evidence sensitivity</h2>",
            f"<p>{_escape(summarize_counterfactual(payload))}</p>",
        ])
        chart = assets.get("counterfactual")
        if chart:
            html_sections.append(f"<img class=\"chart\" src=\"{_escape(_inline_image(chart))}\" alt=\"Counterfactual evidence sensitivity chart\">")
        else:
            html_sections.append("<p class=\"muted\">Counterfactual visualization: NOT_AVAILABLE.</p>")
        html_sections.extend([
            "<h2>F. Temporal dense-AV evidence</h2>",
            "<p class=\"muted\">This is timestamped audio-mouth correspondence evidence. It does not identify a ground-truth manipulation interval.</p>",
        ])
        timeline = assets.get("av_timeline")
        if timeline:
            html_sections.append(f"<img class=\"chart\" src=\"{_escape(_inline_image(timeline))}\" alt=\"Temporal dense AV evidence timeline\">")
        else:
            html_sections.append("<p class=\"muted\">Dense AV temporal evidence: NOT_AVAILABLE.</p>")
    html_sections.extend([
        "<h2>G. Model and input provenance</h2>",
        _table(("Item", "Value"), _provenance_rows(payload)),
        "<h2>H. Limitations</h2>",
        _list(payload.get("limitations") if isinstance(payload.get("limitations"), list) else (IMAGE_LIMITATIONS if is_image else VIDEO_LIMITATIONS)),
    ])
    if renderer_notes:
        html_sections.extend(["<h2>Report-generation notes</h2>", _list(renderer_notes)])
    html_sections.extend(["<p class=\"muted\">Generated locally from the supplied prediction artifact. The report does not rerun inference.</p>", "</main></body></html>"])
    return "".join(html_sections)


def generate_report(prediction_path: Path, output: Path, *, assets_dir: Path | None = None, image_path: Path | None = None) -> dict[str, Any]:
    payload = load_prediction(prediction_path)
    assets_dir = assets_dir or output.parent / f"{output.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    assets: dict[str, Path | None] = {"counterfactual": None, "av_timeline": None, "image_overlay": None}
    is_image = payload.get("media_type") == "image" or str(payload.get("schema_version", "")).startswith("pramaan_x_image")
    if is_image:
        if image_path is not None:
            findings = payload.get("visual_findings") if isinstance(payload.get("visual_findings"), list) else []
            try:
                assets["image_overlay"] = render_image_overlay(image_path, findings, assets_dir / "image_region_overlay.png")
            except (OSError, ValueError) as exc:
                notes.append(f"Image overlay unavailable: {type(exc).__name__}: {exc}")
    else:
        chart, note = _safe_render(render_counterfactual_chart, payload, assets_dir / "counterfactual_sensitivity.png")
        assets["counterfactual"] = chart
        if note:
            notes.append(note)
        timeline, note = _safe_render(render_av_timeline, payload, assets_dir / "dense_av_timeline.png")
        assets["av_timeline"] = timeline
        if note:
            notes.append(note)
    document = _html_document(payload, assets, notes)
    _atomic_write(output, document)
    return {"output": str(output), "assets": {key: str(value) if value else None for key, value in assets.items()}, "notes": notes}


def main() -> None:
    args = parser().parse_args()
    prediction = args.prediction.expanduser().resolve()
    output = args.output.expanduser().resolve()
    image = args.image.expanduser().resolve() if args.image else None
    if image is not None and not image.is_file():
        raise FileNotFoundError(f"Image source not found: {image}")
    result = generate_report(prediction, output, assets_dir=args.assets_dir, image_path=image)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

