from __future__ import annotations

import argparse
import json

from .model import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PRNU/device/platform attribution model.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root folder.")
    parser.add_argument("--model-dir", default="models", help="Output model folder.")
    parser.add_argument("--prnu-size", type=int, default=512, help="Square residual size used for PRNU.")
    args = parser.parse_args()

    manifest = train(args.dataset, args.model_dir, prnu_size=args.prnu_size)
    print(json.dumps({
        "model_dir": args.model_dir,
        "summary": manifest.get("summary"),
        "warnings": manifest.get("warnings", []),
    }, indent=2))


if __name__ == "__main__":
    main()

