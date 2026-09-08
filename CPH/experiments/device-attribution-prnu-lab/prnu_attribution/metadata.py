from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import math

import numpy as np
from PIL import ExifTags, Image, ImageOps


TAG_NAMES = ExifTags.TAGS


def read_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    data: dict[str, Any] = {
        "path": str(path),
        "file_name": path.name,
        "file_size_bytes": stat.st_size,
        "sha256": _sha256(path),
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "make": None,
        "model": None,
        "software": None,
        "datetime": None,
        "exif_present": False,
        "exif_tag_count": 0,
        "gps_present": False,
        "jpeg_quantization": None,
        "compression": {},
    }
    try:
        with Image.open(path) as raw:
            data["format"] = raw.format
            quant = getattr(raw, "quantization", None)
            exif = raw.getexif()

            img = ImageOps.exif_transpose(raw)
            data["width"], data["height"] = img.size
            data["mode"] = img.mode

            data["exif_present"] = bool(exif)
            data["exif_tag_count"] = len(exif)
            named = {TAG_NAMES.get(tag, str(tag)): value for tag, value in exif.items()}
            data["make"] = _clean_text(named.get("Make"))
            data["model"] = _clean_text(named.get("Model"))
            data["software"] = _clean_text(named.get("Software"))
            data["datetime"] = _clean_text(named.get("DateTimeOriginal") or named.get("DateTime"))
            data["gps_present"] = "GPSInfo" in named

            if quant:
                tables = []
                for key in sorted(quant):
                    values = list(quant[key])
                    tables.append(values[:64] + [0] * max(0, 64 - len(values)))
                data["jpeg_quantization"] = tables[:4]
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"

    data["compression"] = compression_summary(data)
    return data


def compression_summary(meta: dict[str, Any]) -> dict[str, Any]:
    width = meta.get("width") or 0
    height = meta.get("height") or 0
    pixels = max(1, int(width) * int(height))
    bytes_per_mp = meta.get("file_size_bytes", 0) / (pixels / 1_000_000)
    q_values = _flat_qtables(meta.get("jpeg_quantization"))
    summary = {
        "bytes_per_megapixel": round(float(bytes_per_mp), 2),
        "has_jpeg_quantization": bool(q_values),
        "quant_table_count": len(meta.get("jpeg_quantization") or []),
        "quant_mean": None,
        "quant_std": None,
        "quant_hash": None,
        "metadata_state": "present" if meta.get("exif_present") else "stripped_or_absent",
    }
    if q_values:
        arr = np.asarray(q_values, dtype=np.float32)
        summary["quant_mean"] = round(float(arr.mean()), 4)
        summary["quant_std"] = round(float(arr.std()), 4)
        summary["quant_hash"] = hashlib.sha1(bytes(int(x) % 256 for x in q_values)).hexdigest()[:12]
    return summary


def feature_vector(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    meta = read_metadata(path)
    width = float(meta.get("width") or 0)
    height = float(meta.get("height") or 0)
    pixels = max(1.0, width * height)
    q_values = _flat_qtables(meta.get("jpeg_quantization"))
    q_padded = (q_values[:128] + [0] * 128)[:128]
    q_arr = np.asarray(q_padded, dtype=np.float32) / 255.0
    nonzero_q = [v for v in q_values if v > 0]
    high_freq = nonzero_q[32:] if len(nonzero_q) > 32 else nonzero_q

    scalars = np.asarray(
        [
            math.log1p(float(meta.get("file_size_bytes") or 0)) / 16.0,
            width / 8000.0,
            height / 8000.0,
            (width / height) if height else 0.0,
            math.log1p((meta.get("compression") or {}).get("bytes_per_megapixel") or 0) / 16.0,
            1.0 if meta.get("exif_present") else 0.0,
            float(meta.get("exif_tag_count") or 0) / 80.0,
            float((meta.get("compression") or {}).get("quant_table_count") or 0) / 4.0,
            (float(np.mean(nonzero_q)) / 255.0) if nonzero_q else 0.0,
            (float(np.std(nonzero_q)) / 255.0) if nonzero_q else 0.0,
            (float(np.mean(high_freq)) / 255.0) if high_freq else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([scalars, q_arr]).astype(np.float32), meta


def _flat_qtables(tables: Any) -> list[int]:
    if not tables:
        return []
    flat: list[int] = []
    for table in tables:
        flat.extend(int(v) for v in table[:64])
    return flat


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
