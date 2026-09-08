from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


DEFAULT_SIZE = 512


def residual_from_image(path: str | Path, size: int = DEFAULT_SIZE) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = _center_crop_square(img)
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    blurred = _blur5(_blur5(arr))
    residual = arr - blurred

    mask = (arr > 0.04) & (arr < 0.98)
    residual = np.where(mask, residual, 0.0)
    residual = residual - residual.mean(axis=0, keepdims=True)
    residual = residual - residual.mean(axis=1, keepdims=True)
    residual = residual - residual.mean()
    std = float(residual.std())
    if std > 1e-8:
        residual = residual / std
    return residual.astype(np.float32)


def fingerprint(paths: Iterable[str | Path], size: int = DEFAULT_SIZE) -> tuple[np.ndarray | None, list[str]]:
    residuals = []
    errors: list[str] = []
    for path in paths:
        try:
            residuals.append(residual_from_image(path, size=size))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if not residuals:
        return None, errors
    fp = np.mean(np.stack(residuals, axis=0), axis=0)
    fp = normalize(fp)
    return fp.astype(np.float32), errors


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalize(a)
    bb = normalize(b)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(aa * bb) / denom)


def normalize(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    out -= float(out.mean())
    std = float(out.std())
    if std > 1e-8:
        out /= std
    return out


def _center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def _blur5(arr: np.ndarray) -> np.ndarray:
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    pad = 2

    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.zeros_like(arr, dtype=np.float32)
    for idx, weight in enumerate(kernel):
        tmp += weight * padded[:, idx : idx + arr.shape[1]]

    padded = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for idx, weight in enumerate(kernel):
        out += weight * padded[idx : idx + arr.shape[0], :]
    return out

