from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".dng"}

REFERENCE_HINTS = {"reference", "ref", "original_reference", "reference_original"}
TEST_HINTS = {"test", "query", "original_test", "test_original"}

PLATFORM_ALIASES = {
    "original": "original",
    "originals": "original",
    "native": "original",
    "camera": "original",
    "reference": "original",
    "ref": "original",
    "original_reference": "original",
    "reference_original": "original",
    "original_test": "original",
    "test_original": "original",
    "whatsapp": "whatsapp",
    "wa": "whatsapp",
    "whatsapp_compressed": "whatsapp",
    "reddit": "reddit",
    "reddit_compressed": "reddit",
    "reddit_download": "reddit",
    "reddit_downloaded": "reddit",
    "telegram": "telegram",
    "tg": "telegram",
    "instagram": "instagram",
    "ig": "instagram",
    "instagram_compressed": "instagram",
    "instagram_download": "instagram",
    "instagram_downloaded": "instagram",
    "facebook": "facebook",
    "fb": "facebook",
    "facebook_compressed": "facebook",
    "facebook_download": "facebook",
    "facebook_downloaded": "facebook",
    "youtube": "youtube",
    "yt": "youtube",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    device_id: str
    platform: str
    split: str

    def as_row(self, root: Path) -> dict[str, str]:
        return {
            "path": str(self.path),
            "relative_path": str(self.path.relative_to(root)) if self.path.is_relative_to(root) else str(self.path),
            "device_id": self.device_id,
            "platform": self.platform,
            "split": self.split,
        }


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_platform(name: str) -> str:
    key = _clean_name(name)
    if key in PLATFORM_ALIASES:
        return PLATFORM_ALIASES[key]
    for alias, platform in PLATFORM_ALIASES.items():
        if alias in key:
            return platform
    return key or "unknown"


def infer_split(name: str, platform: str) -> str:
    key = _clean_name(name)
    if platform == "original" and any(hint in key for hint in REFERENCE_HINTS):
        return "reference"
    if any(hint in key for hint in TEST_HINTS):
        return "test"
    if platform == "original":
        return "train"
    return "platform_sample"


def discover_dataset(root: str | Path) -> list[ImageRecord]:
    root = Path(root).resolve()
    labels_csv = root / "labels.csv"
    if labels_csv.exists():
        return _read_labels_csv(root, labels_csv)

    records: list[ImageRecord] = []
    if not root.exists():
        return records

    for device_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        device_id = device_dir.name
        child_dirs = [p for p in sorted(device_dir.iterdir()) if p.is_dir()]

        direct_images = [p for p in sorted(device_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        for path in direct_images:
            records.append(ImageRecord(path=path, device_id=device_id, platform="original", split="train"))

        for bucket in child_dirs:
            platform = infer_platform(bucket.name)
            split = infer_split(bucket.name, platform)
            for path in image_files(bucket):
                records.append(ImageRecord(path=path, device_id=device_id, platform=platform, split=split))

    return records


def summarize_records(records: Iterable[ImageRecord]) -> dict[str, object]:
    rows = list(records)
    by_device: dict[str, int] = {}
    by_platform: dict[str, int] = {}
    by_split: dict[str, int] = {}
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        by_device[row.device_id] = by_device.get(row.device_id, 0) + 1
        by_platform[row.platform] = by_platform.get(row.platform, 0) + 1
        by_split[row.split] = by_split.get(row.split, 0) + 1
        matrix.setdefault(row.device_id, {})
        matrix[row.device_id][row.platform] = matrix[row.device_id].get(row.platform, 0) + 1
    return {
        "total_images": len(rows),
        "devices": by_device,
        "platforms": by_platform,
        "splits": by_split,
        "device_platform_matrix": matrix,
    }


def _read_labels_csv(root: Path, labels_csv: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with labels_csv.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            rel = raw.get("file_path") or raw.get("path") or raw.get("relative_path")
            if not rel:
                continue
            path = (root / rel).resolve()
            if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            device_id = raw.get("device_id") or raw.get("device") or path.parts[-3]
            platform = infer_platform(raw.get("platform") or raw.get("platform_path") or path.parent.name)
            is_reference = (raw.get("is_reference") or "").strip().lower() in {"1", "true", "yes", "y"}
            split = raw.get("split") or ("reference" if is_reference else infer_split(path.parent.name, platform))
            records.append(ImageRecord(path=path, device_id=device_id, platform=platform, split=split))
    return records


def _clean_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
