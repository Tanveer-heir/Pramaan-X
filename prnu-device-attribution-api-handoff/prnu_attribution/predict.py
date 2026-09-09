from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from .model import predict


def atomic_json(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the source device for one image.")
    parser.add_argument("--image", required=True, help="Query image path.")
    parser.add_argument("--model-dir", default="models", help="Trained model folder.")
    parser.add_argument("--json", action="store_true", help="Print full JSON only.")
    parser.add_argument("--output", type=Path, help="Optional file path for the full JSON result.")
    args = parser.parse_args()

    result = predict(args.image, args.model_dir, mode="device")
    if args.output is not None:
        atomic_json(args.output, result)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return

    device = result["device_attribution"]
    print(f"Device:  {device.get('display_name') or device.get('prediction')} ({device.get('confidence', 0):.1%}) via {device.get('primary_method')}")
    print()
    print("Evidence:")
    evidence = []
    evidence.extend(device.get("evidence", []))
    for item in evidence:
        print(f"- {item}")
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
