"""Render leave-one-available-modality-out evidence sensitivity."""

from __future__ import annotations

from collections.abc import Iterable
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PREFERRED_ORDER = ("without_visual", "without_audio", "without_dense_av")
DISPLAY_NAMES = {
    "without_visual": "Without visual",
    "without_audio": "Without audio",
    "without_dense_av": "Without dense AV",
}
_NAVY = (22, 38, 63)
_BLUE = (55, 113, 190)
_GREEN = (48, 139, 85)
_ORANGE = (214, 126, 40)
_RED = (184, 63, 63)
_GRID = (218, 224, 232)
_TEXT = (35, 42, 52)
_MUTED = (91, 103, 119)
_WHITE = (255, 255, 255)


def _font(size: int) -> ImageFont.ImageFont:
    """Use a bundled-free font so report generation works on minimal systems."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _score(record: dict[str, Any]) -> tuple[float | None, str]:
    for field, label in (
        ("softmax_scores", "Selected-class softmax score (uncalibrated)"),
        ("softmax_probabilities", "Selected-class softmax score (uncalibrated)"),
        ("class_logits", "Selected-class logit"),
    ):
        scores = record.get(field)
        selected = record.get("selected_class")
        if isinstance(scores, dict) and selected in scores:
            value = _number(scores[selected])
            if value is not None:
                return value, label
    return None, "Selected-class score unavailable"


def _rows(section: dict[str, Any]) -> list[dict[str, Any]]:
    full = section.get("full_evidence")
    if not isinstance(full, dict):
        raise ValueError("counterfactual_modality_analysis.full_evidence must be an object")
    rows = [{"key": "full_evidence", "label": "Full evidence", "record": full, "class_changed": False}]
    interventions = section.get("interventions")
    if not isinstance(interventions, dict):
        interventions = {}
    keys = [key for key in PREFERRED_ORDER if key in interventions]
    keys.extend(key for key in interventions if key not in keys)
    full_class = full.get("selected_class")
    for key in keys:
        record = interventions[key]
        if not isinstance(record, dict):
            continue
        changed = record.get("class_changed")
        if not isinstance(changed, bool):
            changed = record.get("selected_class") != full_class
        rows.append({
            "key": key,
            "label": DISPLAY_NAMES.get(key, key.replace("_", " ").title()),
            "record": record,
            "class_changed": changed,
        })
    return rows


def counterfactual_rows(payload_or_section: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized chart rows from a full prediction or its section."""
    section = payload_or_section.get("counterfactual_modality_analysis", payload_or_section)
    if not isinstance(section, dict):
        raise ValueError("counterfactual modality analysis must be an object")
    result = []
    for row in _rows(section):
        value, value_label = _score(row["record"])
        result.append({**row, "score": value, "score_label": value_label, "selected_class": row["record"].get("selected_class")})
    return result


def summarize_counterfactual(payload_or_section: dict[str, Any]) -> str:
    """Create restrained deterministic text, without causal language."""
    section = payload_or_section.get("counterfactual_modality_analysis", payload_or_section)
    if not isinstance(section, dict):
        return "Counterfactual evidence sensitivity was not available."
    summary = section.get("summary")
    changed = summary.get("class_changed_without", []) if isinstance(summary, dict) else []
    stable = summary.get("class_stable_without", []) if isinstance(summary, dict) else []
    changed = [str(item).replace("_", " ") for item in changed] if isinstance(changed, list) else []
    stable = [str(item).replace("_", " ") for item in stable] if isinstance(stable, list) else []
    if not changed and not stable:
        rows = counterfactual_rows(section)
        changed = [row["key"].replace("without_", "").replace("_", " ") for row in rows[1:] if row["class_changed"]]
        stable = [row["key"].replace("without_", "").replace("_", " ") for row in rows[1:] if not row["class_changed"]]
    if changed and stable:
        return f"The selected class changed when {', '.join(changed)} evidence was removed, and remained stable when {', '.join(stable)} evidence was removed."
    if changed:
        return f"The selected class changed when {', '.join(changed)} evidence was removed."
    if stable:
        return f"The selected class remained stable under removal of {', '.join(stable)} evidence."
    return "No valid single-modality counterfactual interventions were available."


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def render_counterfactual_chart(payload: dict[str, Any], output: Path) -> Path | None:
    """Write a PNG showing selected-class scores for each evidence configuration."""
    rows = counterfactual_rows(payload)
    usable = [row for row in rows if row["score"] is not None]
    if not usable:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1320, 760
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _font(30), _font(20), _font(16)
    draw.text((56, 36), "Counterfactual evidence sensitivity", fill=_NAVY, font=title_font)
    draw.text((56, 82), "Masked-evidence comparison, not causal attribution", fill=_MUTED, font=body_font)

    chart_left, chart_top, chart_right, chart_bottom = 110, 160, width - 70, 530
    values = [float(row["score"]) for row in usable]
    score_label = usable[0]["score_label"]
    is_softmax = "softmax" in score_label.lower()
    if is_softmax:
        lower, upper = 0.0, 1.0
    else:
        lower = min(0.0, min(values))
        upper = max(0.0, max(values))
        if math.isclose(lower, upper):
            lower, upper = lower - 1.0, upper + 1.0
    span = upper - lower
    for tick in range(6):
        value = lower + span * tick / 5
        y = chart_bottom - int((value - lower) / span * (chart_bottom - chart_top))
        draw.line((chart_left, y, chart_right, y), fill=_GRID, width=1)
        draw.text((20, y - 10), f"{value:.2f}", fill=_MUTED, font=small_font)
    zero_y = chart_bottom - int((0.0 - lower) / span * (chart_bottom - chart_top)) if lower <= 0 <= upper else chart_bottom
    if chart_top <= zero_y <= chart_bottom:
        draw.line((chart_left, zero_y, chart_right, zero_y), fill=_MUTED, width=2)

    band = (chart_right - chart_left) / len(usable)
    bar_width = min(150, int(band * 0.58))
    for index, row in enumerate(usable):
        value = float(row["score"])
        center = chart_left + band * (index + 0.5)
        x0, x1 = int(center - bar_width / 2), int(center + bar_width / 2)
        y_value = chart_bottom - int((value - lower) / span * (chart_bottom - chart_top))
        y0, y1 = min(zero_y, y_value), max(zero_y, y_value)
        color = _BLUE if index == 0 else (_RED if row["class_changed"] else _GREEN)
        draw.rounded_rectangle((x0, y0, x1, max(y1, y0 + 2)), radius=8, fill=color)
        draw.text((int(center - 28), min(y0, y1) - 30), f"{value:.3f}", fill=_TEXT, font=small_font)
        label_lines = _wrap(row["label"], 16)
        for line_index, line in enumerate(label_lines):
            box = draw.textbbox((0, 0), line, font=small_font)
            draw.text((int(center - (box[2] - box[0]) / 2), chart_bottom + 20 + line_index * 20), line, fill=_TEXT, font=small_font)
        selected = str(row["selected_class"] or "Unavailable")
        class_lines = _wrap(selected, 21)
        for line_index, line in enumerate(class_lines):
            box = draw.textbbox((0, 0), line, font=small_font)
            draw.text((int(center - (box[2] - box[0]) / 2), 570 + line_index * 19), line, fill=_MUTED, font=small_font)
        if index > 0:
            status = "CLASS CHANGED" if row["class_changed"] else "STABLE"
            status_color = _RED if row["class_changed"] else _GREEN
            box = draw.textbbox((0, 0), status, font=small_font)
            draw.text((int(center - (box[2] - box[0]) / 2), 665), status, fill=status_color, font=small_font)

    draw.text((chart_left, 705), score_label, fill=_MUTED, font=small_font)
    draw.text((width - 370, 705), "Blue = full evidence  |  Green = stable  |  Red = changed", fill=_MUTED, font=small_font)
    image.save(output, format="PNG")
    return output


__all__ = ["counterfactual_rows", "render_counterfactual_chart", "summarize_counterfactual"]

