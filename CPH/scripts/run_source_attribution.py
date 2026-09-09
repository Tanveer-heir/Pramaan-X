"""Run only the CPH source-attribution pipeline for one image or video."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


CPH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CPH_ROOT))

from src.source_attribution.pipeline import SourceAttributionPipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--media", type=Path, required=True, help="Path to an image or video.")
    result.add_argument("--output", type=Path, required=True, help="Destination for OriginTracingEvidence JSON.")
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def run(media_path: Path) -> dict[str, Any]:
    result = await SourceAttributionPipeline().execute(media_path=str(media_path))
    return result.model_dump(mode="json")


def main() -> None:
    args = parser().parse_args()
    media = args.media.expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"Input media not found: {media}")
    payload = asyncio.run(run(media))
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
