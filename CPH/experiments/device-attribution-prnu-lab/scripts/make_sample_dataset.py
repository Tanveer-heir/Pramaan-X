from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic smoke-test dataset.")
    parser.add_argument("--out", default="sample_dataset")
    parser.add_argument("--devices", type=int, default=3)
    parser.add_argument("--refs", type=int, default=12)
    parser.add_argument("--tests", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(42)
    for d in range(args.devices):
        device = f"device_{chr(65 + d)}_synthetic_oneplus"
        sensor = rng.normal(0, 3.5, size=(768, 1024, 1)).astype(np.float32)
        for folder in ["original_reference", "original_test", "reddit", "instagram", "facebook"]:
            (out / device / folder).mkdir(parents=True, exist_ok=True)
        for idx in range(args.refs):
            image = synth_image(rng, sensor, flat=True)
            save_jpeg(out / device / "original_reference" / f"ref_{idx:03d}.jpg", image, quality=96)
        for idx in range(args.tests):
            image = synth_image(rng, sensor, flat=False)
            save_jpeg(out / device / "original_test" / f"test_{idx:03d}.jpg", image, quality=94)
            save_jpeg(out / device / "reddit" / f"reddit_{idx:03d}.jpg", image.resize((960, 720)), quality=86)
            save_jpeg(out / device / "instagram" / f"instagram_{idx:03d}.jpg", image.resize((1080, 810)), quality=78)
            save_jpeg(out / device / "facebook" / f"facebook_{idx:03d}.jpg", image.resize((900, 675)), quality=72)
    print(f"Wrote synthetic dataset to {out.resolve()}")


def synth_image(rng: np.random.Generator, sensor: np.ndarray, flat: bool) -> Image.Image:
    h, w, _ = sensor.shape
    if flat:
        base = np.full((h, w, 3), rng.uniform(135, 210), dtype=np.float32)
        base += rng.normal(0, 4, size=(h, w, 3))
    else:
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        base = 80 + 130 * np.concatenate([x.repeat(h, 0), y.repeat(w, 1), (1 - x).repeat(h, 0)], axis=2)
        for _ in range(8):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            rr = rng.integers(30, 140)
            color = rng.uniform(40, 230, size=(1, 1, 3))
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < rr ** 2
            base[mask] = color
    arr = np.clip(base + sensor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def save_jpeg(path: Path, image: Image.Image, quality: int) -> None:
    image.save(path, "JPEG", quality=quality, optimize=True)


if __name__ == "__main__":
    main()
