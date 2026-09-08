from __future__ import annotations

import argparse
import json

from .model import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the source device for one image.")
    parser.add_argument("--image", required=True, help="Query image path.")
    parser.add_argument("--model-dir", default="models", help="Trained model folder.")
    parser.add_argument("--json", action="store_true", help="Print full JSON only.")
    args = parser.parse_args()

    result = predict(args.image, args.model_dir, mode="device")
    if args.json:
        print(json.dumps(result, indent=2))
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
