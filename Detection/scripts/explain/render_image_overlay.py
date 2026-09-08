"""Render optional, explicitly normalized image finding regions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def normalize_region_bbox(value: Any) -> dict[str, float] | None:
    """Validate normalized x/y/width/height coordinates in the image frame."""
    if isinstance(value, dict):
        values = [value.get(key) for key in ("x", "y", "width", "height")]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        values = list(value)
    else:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        return None
    x, y, width, height = (float(item) for item in values)
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        return None
    if min(x, y, width, height) < 0 or width <= 0 or height <= 0:
        return None
    if x + width > 1 or y + height > 1:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def render_image_overlay(image_path: Path, findings: list[dict[str, Any]], output: Path) -> Path | None:
    """Draw only explicitly normalized model-indicated boxes, never text guesses."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    boxes = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        box = normalize_region_bbox(finding.get("region_bbox"))
        if box is not None:
            boxes.append((index + 1, box, str(finding.get("category", "finding"))))
    if not boxes:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    width, height = image.size
    for number, box, category in boxes:
        x0, y0 = int(box["x"] * width), int(box["y"] * height)
        x1, y1 = int((box["x"] + box["width"]) * width), int((box["y"] + box["height"]) * height)
        draw.rectangle((x0, y0, x1, y1), outline=(204, 44, 44), width=max(3, width // 500))
        label = f"{number}: {category}"
        text_box = draw.textbbox((0, 0), label, font=font)
        label_bottom = max(y0, text_box[3] - text_box[1] + 8)
        draw.rectangle((x0, max(0, label_bottom - (text_box[3] - text_box[1] + 8)), x0 + text_box[2] + 10, label_bottom), fill=(204, 44, 44))
        draw.text((x0 + 5, label_bottom - (text_box[3] - text_box[1]) - 3), label, fill=(255, 255, 255), font=font)
    image.save(output, format="PNG")
    return output


__all__ = ["normalize_region_bbox", "render_image_overlay"]

