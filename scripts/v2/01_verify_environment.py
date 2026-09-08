"""Phase 1: Verify Pramaan-X environment and accepted baseline integrity.

Checks:
1. All 4 accepted checkpoints exist and match SHA-256
2. MediaPipe face landmarker asset exists
3. XLSR-SLS ONNX model exists and matches SHA-256
4. Fusion checkpoint loads cleanly with weights_only=True
5. Prints environment summary
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CHECKPOINTS = {
    "checkpoints/visual_temporal/visual_temporal_20260904T145626Z.pt":
        "ac348279792a064c24ef53f4c083e0d6bdaca7b5d801bf39d21806a5c5e87f3f",
    "checkpoints/audio_classifier/audio_classifier_20260904T181732Z.pt":
        "67ea124dfa3ad71133898c548ca371be2fbe58633a92420f19d5a322375fe8cb",
    "checkpoints/av_sync_dense/av_sync_dense_20260905T055537Z.pt":
        "cd93a68bc168cec4853f6c280174d7fe16b3b328b5ae568112bc1d5b237b45a9",
    "checkpoints/fusion/pramaan_x_hackathon_final.pth":
        "f6127b5ba37ce195be50d6f42919ab7d7e41aac17cda5bc866d791935a34670c",
}

EXPECTED_ASSETS = {
    "models/mediapipe/face_landmarker.task": None,  # no hash check, just existence
    "models/xlsr_sls/xlsr-sls.onnx": "0aa36298a1a893bfb58a8d721dd30d046157f559da1165f95d2a0b69988c1bbd",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 60, flush=True)
    print("PRAMAAN-X V2: Environment Verification", flush=True)
    print("=" * 60, flush=True)

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Python & OS
    print(f"\n[Environment]")
    print(f"  Python:    {sys.version}")
    print(f"  Platform:  {platform.platform()}")
    print(f"  Cwd:       {Path.cwd()}")
    print(f"  Root:      {PROJECT_ROOT}")

    # 2. PyTorch
    try:
        import torch
        print(f"  PyTorch:   {torch.__version__}")
        print(f"  CUDA:      {torch.cuda.is_available()}")
    except ImportError:
        errors.append("PyTorch not installed")

    # 3. NumPy
    try:
        import numpy as np
        print(f"  NumPy:     {np.__version__}")
    except ImportError:
        errors.append("NumPy not installed")

    # 4. ONNX Runtime
    try:
        import onnxruntime
        print(f"  ORT:       {onnxruntime.__version__}")
    except ImportError:
        warnings.append("onnxruntime not installed (needed for audio extraction)")

    # 5. FFmpeg
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    # Also check project-local tools
    local_ffmpeg = PROJECT_ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if not ffmpeg and local_ffmpeg.is_file():
        ffmpeg = str(local_ffmpeg)
        ffprobe = str(local_ffmpeg.parent / "ffprobe.exe")
    print(f"  FFmpeg:    {ffmpeg or 'NOT FOUND'}")
    print(f"  FFprobe:   {ffprobe or 'NOT FOUND'}")
    if not ffmpeg:
        warnings.append("FFmpeg not on PATH. Run: venv\\Scripts\\python.exe scripts/v2/00_install_ffmpeg.py")

    # 6. Checkpoints
    print(f"\n[Checkpoint Verification]")
    for rel_path, expected_hash in EXPECTED_CHECKPOINTS.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.is_file():
            errors.append(f"MISSING checkpoint: {rel_path}")
            print(f"  [MISSING] {rel_path}")
            continue
        actual = sha256_file(full_path)
        if actual != expected_hash:
            errors.append(f"HASH MISMATCH: {rel_path}\n    expected: {expected_hash}\n    actual:   {actual}")
            print(f"  [HASH MISMATCH] {rel_path}")
        else:
            print(f"  [OK] {rel_path}")

    # 7. Model assets
    print(f"\n[Model Assets]")
    for rel_path, expected_hash in EXPECTED_ASSETS.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.is_file():
            errors.append(f"MISSING asset: {rel_path}")
            print(f"  [MISSING] {rel_path}")
            continue
        if expected_hash:
            actual = sha256_file(full_path)
            if actual != expected_hash:
                errors.append(f"HASH MISMATCH: {rel_path}")
                print(f"  [HASH MISMATCH] {rel_path}")
            else:
                print(f"  [OK] {rel_path} ({full_path.stat().st_size / 1024 / 1024:.0f} MB)")
        else:
            print(f"  [OK] {rel_path} exists ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 8. Fusion checkpoint load test
    print(f"\n[Fusion Checkpoint Load Test]")
    try:
        import torch
        payload = torch.load(
            PROJECT_ROOT / "checkpoints/fusion/pramaan_x_hackathon_final.pth",
            map_location="cpu",
            weights_only=True,
        )
        print(f"  Classes:          {payload['class_names']}")
        print(f"  Enabled branches: {payload['enabled_branches']}")
        print(f"  Feature order:    {payload['input_feature_order']}")
        print(f"  Architecture:     input_dim={payload['architecture']['input_dim']}, hidden_dim={payload['architecture']['hidden_dim']}")
        print(f"  Selected epoch:   {payload.get('selected_epoch', 'unknown')}")
        if "dev_metrics" in payload:
            dm = payload["dev_metrics"]
            print(f"  Dev accuracy:     {dm.get('accuracy', 'N/A')}")
            print(f"  Dev macro-F1:     {dm.get('macro_f1', 'N/A')}")
    except Exception as exc:
        errors.append(f"Fusion checkpoint load failed: {exc}")
        print(f"  [FAIL] Load failed: {exc}")

    # 9. Feature cache summary
    print(f"\n[Feature Cache Summary]")
    cache_roots = {
        "visual": "data/interim/visual_embeddings/convnext_tiny_v1",
        "audio": "data/interim/audio_embeddings/xlsr_sls_v1",
        "dense_av": "data/interim/av_sync_features/dense_v1",
        "blink": "data/interim/blink_features/ear_v1",
    }
    for name, rel in cache_roots.items():
        p = PROJECT_ROOT / rel
        if p.exists():
            count = len(list(p.glob("*.npz")))
            print(f"  {name:10s}: {count:6d} cached features")
        else:
            print(f"  {name:10s}: directory not found")

    # 10. Split summary
    print(f"\n[Data Split Summary]")
    import csv
    for split_name in ("train", "validation", "test", "calibration"):
        split_path = PROJECT_ROOT / f"data/metadata/splits/{split_name}.csv"
        if split_path.is_file():
            with split_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            classes = {}
            for r in rows:
                sc = r.get("semantic_class", "unknown")
                classes[sc] = classes.get(sc, 0) + 1
            print(f"  {split_name:12s}: {len(rows):6d} samples  {dict(sorted(classes.items()))}")
        else:
            print(f"  {split_name:12s}: not found")

    # Summary
    print(f"\n{'=' * 60}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [FAIL] {e}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")
    if not errors:
        print("ALL CHECKS PASSED [OK]")
    else:
        print(f"\n{len(errors)} error(s) must be fixed before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
