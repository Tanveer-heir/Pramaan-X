"""Render timestamped dense audio-mouth correspondence evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


_NAVY = (22, 38, 63)
_BLUE = (55, 113, 190)
_GRID = (218, 224, 232)
_TEXT = (35, 42, 52)
_MUTED = (91, 103, 119)
_WHITE = (255, 255, 255)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def extract_timeline(payload_or_section: dict[str, Any]) -> tuple[list[dict[str, float | int]], str]:
    """Return valid timestamped segments and the exact evidence field used."""
    section = payload_or_section.get("temporal_evidence", payload_or_section)
    if isinstance(section, dict) and "dense_av" in section:
        section = section["dense_av"]
    if not isinstance(section, dict):
        return [], "Dense AV evidence unavailable"
    raw_segments = section.get("segments")
    if not isinstance(raw_segments, list):
        return [], "Dense AV evidence unavailable"
    result: list[dict[str, float | int]] = []
    field_name: str | None = None
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        start = _number(raw.get("start_sec"))
        end = _number(raw.get("end_sec"))
        center = _number(raw.get("center_sec"))
        if start is None or end is None or end < start:
            continue
        value = _number(raw.get("av_inconsistency_logit"))
        current_field = "AV inconsistency logit"
        if value is None:
            value = _number(raw.get("raw_model_logit"))
            current_field = "Raw dense AV model logit"
        if value is None:
            continue
        if center is None:
            center = (start + end) / 2.0
        if field_name is None:
            field_name = current_field
        if current_field != field_name:
            continue
        result.append({
            "segment_index": int(raw.get("segment_index", len(result))),
            "start_sec": start,
            "end_sec": end,
            "center_sec": center,
            "value": value,
        })
    return result, field_name or "Dense AV evidence unavailable"


def render_av_timeline(payload: dict[str, Any], output: Path) -> Path | None:
    """Write a PNG using the cached segment timestamps, with no fake threshold."""
    segments, value_label = extract_timeline(payload)
    if not segments:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1320, 700
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _font(30), _font(20), _font(16)
    draw.text((56, 34), "Temporal audio-mouth correspondence evidence", fill=_NAVY, font=title_font)
    draw.text((56, 80), "Higher values indicate stronger model evidence for the plotted quantity", fill=_MUTED, font=body_font)
    left, top, right, bottom = 110, 150, width - 70, 510
    x_values = [float(row["center_sec"]) for row in segments]
    y_values = [float(row["value"]) for row in segments]
    x_min, x_max = min(float(row["start_sec"]) for row in segments), max(float(row["end_sec"]) for row in segments)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    y_min, y_max = min(0.0, min(y_values)), max(0.0, max(y_values))
    if math.isclose(y_min, y_max):
        y_min, y_max = y_min - 1.0, y_max + 1.0

    def px(value: float) -> int:
        return left + int((value - x_min) / (x_max - x_min) * (right - left))

    def py(value: float) -> int:
        return bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = py(value)
        draw.line((left, y, right, y), fill=_GRID, width=1)
        draw.text((18, y - 10), f"{value:.2f}", fill=_MUTED, font=small_font)
    for tick in range(6):
        value = x_min + (x_max - x_min) * tick / 5
        x = px(value)
        draw.line((x, top, x, bottom), fill=_GRID, width=1)
        draw.text((x - 18, bottom + 14), f"{value:.1f}", fill=_MUTED, font=small_font)

    points = [(px(float(row["center_sec"])), py(float(row["value"]))) for row in segments]
    if len(points) > 1:
        draw.line(points, fill=_BLUE, width=4, joint="curve")
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=_BLUE)
    draw.line((left, py(0.0), right, py(0.0)), fill=_MUTED, width=2)
    draw.text((left, 625), f"x-axis: time in seconds  |  y-axis: {value_label}", fill=_TEXT, font=small_font)
    draw.text((left, 650), "No forensic threshold is shown. This is not ground-truth manipulation localization.", fill=_MUTED, font=small_font)
    image.save(output, format="PNG")
    return output


__all__ = ["extract_timeline", "render_av_timeline"]

