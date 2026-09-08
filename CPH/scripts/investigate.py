"""Run the same unified investigation orchestration used by the HTTP API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile


CPH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CPH_ROOT))

from src.gateway.config import GatewayConfig
from src.gateway.media import store_path_once
from src.gateway.orchestrator import UnifiedInvestigationOrchestrator, new_investigation_id


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--media", type=Path, required=True)
    result.add_argument("--case-id")
    result.add_argument("--output", type=Path, required=True)
    return result


def atomic_json(path: Path, payload: dict) -> None:
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


async def run(args: argparse.Namespace) -> dict:
    config = GatewayConfig.from_env()
    investigation_id = new_investigation_id()
    stored = await asyncio.to_thread(
        store_path_once,
        args.media,
        investigation_id,
        config.shared_media_dir,
        config.max_upload_bytes,
    )
    report = await UnifiedInvestigationOrchestrator(config).investigate(stored, args.case_id)
    return report.model_dump(mode="json")


def main() -> None:
    args = parser().parse_args()
    payload = asyncio.run(run(args))
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
