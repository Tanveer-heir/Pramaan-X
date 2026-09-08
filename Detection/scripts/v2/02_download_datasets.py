"""Phase 2: Download targeted external dataset subsets for Pramaan-X V2 evaluation.

Downloads:
1. In-the-Wild Audio Deepfake (full, ~3.5 GB) — held out for external testing
2. MLAAD Hindi + English subsets — audio-only for audio branch training/eval
3. AV-Deepfake1M subset (50 clips per class) — complete AV for fusion eval
4. IndicVoices Hindi subset — real Indian speech reference

Creates a unified master manifest at data/metadata/modern_challenge_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHALLENGE_ROOT = PROJECT_ROOT / "data" / "raw" / "modern_challenge"
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "modern_challenge_manifest.csv"
LOG_PATH = PROJECT_ROOT / "data" / "interim" / "v2_logs" / "download_log.json"

MANIFEST_FIELDS = [
    "sample_id", "file_path", "dataset", "original_id", "identity",
    "manipulation_type", "generator_tool", "language", "modality",
    "provenance", "semantic_class", "visual_label", "audio_label",
    "final_label", "split", "held_out",
]

CLASS_MAP = {"REAL": 0, "VISUAL_MANIPULATION": 1, "AUDIO_MANIPULATION": 2, "AUDIO_VISUAL_MANIPULATION": 3}


def ensure_hf_hub():
    """Ensure huggingface_hub is available."""
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_in_the_wild(hf_token: str) -> list[dict[str, str]]:
    """Download In-the-Wild Audio Deepfake dataset."""
    print("\n" + "=" * 60)
    print("[1/4] In-the-Wild Audio Deepfake (mueller91/In-The-Wild)")
    print("=" * 60)

    dest = CHALLENGE_ROOT / "in_the_wild"
    dest.mkdir(parents=True, exist_ok=True)

    hf = ensure_hf_hub()
    try:
        # Try downloading the zip file
        zip_path = hf.hf_hub_download(
            repo_id="mueller91/In-The-Wild",
            filename="release_in_the_wild.zip",
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dest),
        )
        zip_path = Path(zip_path)
        print(f"  Downloaded: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Extract
        print("  Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        print(f"  Extracted to: {dest}")
    except Exception as exc:
        print(f"  ⚠ Zip download failed ({exc}), trying individual file listing...")
        # Fallback: list and download files
        api = hf.HfApi(token=hf_token)
        try:
            files = api.list_repo_files("mueller91/In-The-Wild", repo_type="dataset")
            audio_files = [f for f in files if f.endswith((".wav", ".flac", ".mp3", ".ogg"))]
            print(f"  Found {len(audio_files)} audio files in repo")

            # Download up to 500 files for manageable size
            max_files = min(500, len(audio_files))
            for i, fname in enumerate(audio_files[:max_files]):
                try:
                    hf.hf_hub_download(
                        repo_id="mueller91/In-The-Wild",
                        filename=fname,
                        repo_type="dataset",
                        token=hf_token,
                        local_dir=str(dest),
                    )
                    if (i + 1) % 50 == 0:
                        print(f"    Downloaded {i + 1}/{max_files} files...")
                except Exception as e2:
                    print(f"    Skip {fname}: {e2}")
            print(f"  Downloaded {max_files} files to {dest}")
        except Exception as e3:
            print(f"  ✗ Could not access In-the-Wild dataset: {e3}")
            return []

    # Build manifest entries
    entries = []
    # Look for the meta.csv or label file
    meta_file = None
    for candidate in [dest / "release_in_the_wild" / "meta.csv",
                      dest / "meta.csv",
                      dest / "release_in_the_wild" / "labels.csv"]:
        if candidate.is_file():
            meta_file = candidate
            break

    if meta_file:
        print(f"  Found metadata: {meta_file}")
        with meta_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # In-the-Wild typically has columns: file, label (where label is 'bona-fide' or 'spoof')
                file_col = None
                for col in ("file", "path", "filename", "audio_path"):
                    if col in row:
                        file_col = col
                        break
                if not file_col:
                    # Try first column
                    file_col = list(row.keys())[0]

                label_col = None
                for col in ("label", "class", "target"):
                    if col in row:
                        label_col = col
                        break

                file_val = row.get(file_col, "")
                label_val = row.get(label_col, "unknown").strip().lower() if label_col else "unknown"

                # Find the actual file
                audio_path = None
                for search_root in [dest, dest / "real", dest / "fake", dest / "release_in_the_wild"]:
                    candidate = search_root / file_val
                    if candidate.is_file():
                        audio_path = candidate
                        break

                if audio_path is None:
                    continue

                is_fake = label_val in ("spoof", "fake", "1", "deepfake")
                is_real = label_val in ("bona-fide", "bonafide", "real", "0", "genuine")

                if is_fake:
                    sem_class = "AUDIO_MANIPULATION"
                    audio_label = 1
                elif is_real:
                    sem_class = "REAL"
                    audio_label = 0
                else:
                    continue  # skip ambiguous

                rel_path = audio_path.relative_to(PROJECT_ROOT)
                sample_id = f"itw_{hashlib.md5(str(rel_path).encode()).hexdigest()[:12]}"

                entries.append({
                    "sample_id": sample_id,
                    "file_path": str(rel_path).replace("\\", "/"),
                    "dataset": "In-the-Wild",
                    "original_id": file_val,
                    "identity": "unknown",
                    "manipulation_type": "audio_cloning" if is_fake else "none",
                    "generator_tool": "unknown" if is_fake else "none",
                    "language": "english",
                    "modality": "audio_only",
                    "provenance": "huggingface/mueller91/In-The-Wild",
                    "semantic_class": sem_class,
                    "visual_label": "0",
                    "audio_label": str(audio_label),
                    "final_label": str(CLASS_MAP[sem_class]),
                    "split": "external_test",
                    "held_out": "true",
                })
    else:
        # No metadata found — scan for audio files and label by directory
        print("  No meta.csv found, scanning directories...")
        for audio_file in sorted(dest.rglob("*")):
            if not audio_file.is_file():
                continue
            if audio_file.suffix.lower() not in (".wav", ".flac", ".mp3", ".ogg"):
                continue

            rel_to_dest = str(audio_file.relative_to(dest)).lower()
            is_fake = any(x in rel_to_dest for x in ("fake", "spoof"))
            is_real = any(x in rel_to_dest for x in ("real", "bona", "genuine"))

            if is_fake:
                sem_class = "AUDIO_MANIPULATION"
                audio_label = 1
            elif is_real:
                sem_class = "REAL"
                audio_label = 0
            else:
                continue

            rel_path = audio_file.relative_to(PROJECT_ROOT)
            sample_id = f"itw_{hashlib.md5(str(rel_path).encode()).hexdigest()[:12]}"

            entries.append({
                "sample_id": sample_id,
                "file_path": str(rel_path).replace("\\", "/"),
                "dataset": "In-the-Wild",
                "original_id": audio_file.stem,
                "identity": "unknown",
                "manipulation_type": "audio_cloning" if is_fake else "none",
                "generator_tool": "unknown" if is_fake else "none",
                "language": "english",
                "modality": "audio_only",
                "provenance": "huggingface/mueller91/In-The-Wild",
                "semantic_class": sem_class,
                "visual_label": "0",
                "audio_label": str(audio_label),
                "final_label": str(CLASS_MAP[sem_class]),
                "split": "external_test",
                "held_out": "true",
            })

    print(f"  Total In-the-Wild entries: {len(entries)}")
    real_count = sum(1 for e in entries if e["semantic_class"] == "REAL")
    fake_count = sum(1 for e in entries if e["semantic_class"] == "AUDIO_MANIPULATION")
    print(f"    Real: {real_count}, Fake: {fake_count}")
    return entries


def download_mlaad(hf_token: str) -> list[dict[str, str]]:
    """Download MLAAD Hindi + English audio subsets fast via targeted repo tree."""
    print("\n" + "=" * 60)
    print("[2/4] MLAAD (mueller91/MLAAD) — Hindi + English subsets")
    print("=" * 60)

    dest = CHALLENGE_ROOT / "mlaad"
    dest.mkdir(parents=True, exist_ok=True)

    hf = ensure_hf_hub()
    api = hf.HfApi(token=hf_token)
    entries = []

    for lang_key, lang_name in [("hi", "hindi"), ("en", "english")]:
        lang_dir = dest / lang_name
        lang_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Fetching {lang_name} file list...")
        try:
            tree = list(api.list_repo_tree("mueller91/MLAAD", path_in_repo=f"fake/{lang_key}", repo_type="dataset", recursive=True))
            audio_files = [f.path for f in tree if hasattr(f, "path") and f.path.endswith((".wav", ".flac", ".mp3", ".ogg"))]
            print(f"  {lang_name}: found {len(audio_files)} files")

            max_per_lang = 100
            step = max(1, len(audio_files) // max_per_lang)
            selected_files = audio_files[::step][:max_per_lang]
            print(f"  Downloading {len(selected_files)} {lang_name} files...")

            downloaded = 0
            for fname in selected_files:
                try:
                    local_path = hf.hf_hub_download(
                        repo_id="mueller91/MLAAD",
                        filename=fname,
                        repo_type="dataset",
                        token=hf_token,
                        local_dir=str(lang_dir),
                    )
                    local_path = Path(local_path)
                    rel_path = local_path.relative_to(PROJECT_ROOT)
                    sample_id = f"mlaad_{lang_key}_{hashlib.md5(fname.encode()).hexdigest()[:10]}"

                    tool_name = fname.split("/")[2] if len(fname.split("/")) > 2 else "unknown_tts"
                    entries.append({
                        "sample_id": sample_id,
                        "file_path": str(rel_path).replace("\\", "/"),
                        "dataset": "MLAAD",
                        "original_id": fname,
                        "identity": "unknown",
                        "manipulation_type": "audio_synthesis",
                        "generator_tool": tool_name,
                        "language": lang_name,
                        "modality": "audio_only",
                        "provenance": "huggingface/mueller91/MLAAD",
                        "semantic_class": "AUDIO_MANIPULATION",
                        "visual_label": "0",
                        "audio_label": "1",
                        "final_label": "2",
                        "split": "external_val",
                        "held_out": "false",
                    })
                    downloaded += 1
                    if downloaded % 25 == 0:
                        print(f"    {lang_name}: {downloaded}/{len(selected_files)} downloaded")
                except Exception as exc:
                    if downloaded == 0 and ("403" in str(exc) or "GatedRepo" in type(exc).__name__):
                        print(f"  ⚠ Gated dataset access denied for MLAAD ({exc})")
                        print("  Accept terms at: https://huggingface.co/datasets/mueller91/MLAAD")
                        break

            print(f"  {lang_name}: {downloaded} files downloaded")
        except Exception as exc:
            print(f"  ⚠ Failed to fetch {lang_name}: {exc}")

    print(f"  Total MLAAD entries: {len(entries)}")
    return entries


def download_av_deepfake1m(hf_token: str) -> list[dict[str, str]]:
    """Download AV-Deepfake1M targeted subset (~50 clips per class) via split zip volumes."""
    print("\n" + "=" * 60)
    print("[3/4] AV-Deepfake1M (ControlNet/AV-Deepfake1M)")
    print("=" * 60)

    dest = CHALLENGE_ROOT / "av_deepfake1m"
    dest.mkdir(parents=True, exist_ok=True)

    hf = ensure_hf_hub()
    entries = []

    class_specs = {
        "visual_modified": ("VISUAL_MANIPULATION", "1", "0", "1", "FVRA"),
        "audio_modified": ("AUDIO_MANIPULATION", "0", "1", "2", "RVFA"),
        "audio_visual_modified": ("AUDIO_VISUAL_MANIPULATION", "1", "1", "3", "FVFA"),
        "real": ("REAL", "0", "0", "0", "RVRA"),
    }

    try:
        print("  Downloading val_metadata.json...")
        meta_path = hf.hf_hub_download(
            repo_id="ControlNet/AV-Deepfake1M",
            filename="val_metadata.json",
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dest),
        )
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Metadata loaded: {len(metadata)} samples total")

        print("  Downloading split zip volumes (val.zip.001 through 005 and 020)...")
        v_paths = []
        for v in ["001", "002", "003", "004", "005", "020"]:
            p = hf.hf_hub_download(
                repo_id="ControlNet/AV-Deepfake1M",
                filename=f"val/val.zip.{v}",
                repo_type="dataset",
                token=hf_token,
            )
            v_paths.append(Path(p))

        combined_zip = dest / "val_combined.zip"
        if not combined_zip.is_file() or combined_zip.stat().st_size < 3_000_000_000:
            print("  Concatenating split zip volumes...")
            with combined_zip.open("wb") as out:
                for vp in v_paths:
                    out.write(vp.read_bytes())
            print(f"  Combined zip ready: {combined_zip.stat().st_size / 1024 / 1024:.1f} MB")

        import zipfile
        by_type: dict[str, list[dict[str, Any]]] = {
            "visual_modified": [],
            "audio_modified": [],
            "audio_visual_modified": [],
            "real": [],
        }
        for item in metadata:
            mtype = item.get("modify_type", "real")
            if mtype is None or mtype not in by_type:
                mtype = "real"
            by_type[mtype].append(item)

        with zipfile.ZipFile(combined_zip) as zf:
            available_in_zip = set(zf.namelist())
            max_per_class = 50

            for mtype, (sem_class, vis_lbl, aud_lbl, fin_lbl, cat_code) in class_specs.items():
                cat_dir = dest / cat_code
                cat_dir.mkdir(parents=True, exist_ok=True)
                
                # Filter candidates that exist in the zip volume
                valid_candidates = []
                for item in by_type[mtype]:
                    zip_entry = f"val/{item['file']}"
                    if zip_entry in available_in_zip:
                        valid_candidates.append(item)
                    if len(valid_candidates) >= max_per_class:
                        break

                print(f"  {cat_code} ({sem_class}): extracting {len(valid_candidates)} clips...")
                downloaded = 0
                for item in valid_candidates:
                    vf = item["file"]
                    zip_entry = f"val/{vf}"
                    try:
                        zf.extract(zip_entry, dest)
                        extracted_file = dest / zip_entry
                        rel_path = extracted_file.relative_to(PROJECT_ROOT)
                        sample_id = f"avdf_{cat_code.lower()}_{hashlib.md5(vf.encode()).hexdigest()[:10]}"

                        entries.append({
                            "sample_id": sample_id,
                            "file_path": str(rel_path).replace("\\", "/"),
                            "dataset": "AV-Deepfake1M",
                            "original_id": vf,
                            "identity": item.get("original", "").split("/")[0] if "original" in item else "unknown",
                            "manipulation_type": mtype,
                            "generator_tool": str(item.get("audio_model", "unknown")),
                            "language": "english",
                            "modality": "audio_video",
                            "provenance": "huggingface/ControlNet/AV-Deepfake1M",
                            "semantic_class": sem_class,
                            "visual_label": vis_lbl,
                            "audio_label": aud_lbl,
                            "final_label": fin_lbl,
                            "split": "external_val",
                            "held_out": "false",
                        })
                        downloaded += 1
                    except Exception as exc:
                        pass

                print(f"  {cat_code}: {downloaded} clips extracted")

    except Exception as exc:
        print(f"  ⚠ AV-Deepfake1M extraction failed: {exc}")

    print(f"  Total AV-Deepfake1M entries: {len(entries)}")
    return entries


def download_indicvoices(hf_token: str) -> list[dict[str, str]]:
    """Download IndicVoices Hindi subset from parquet shards."""
    print("\n" + "=" * 60)
    print("[4/4] IndicVoices Hindi (ai4bharat/IndicVoices)")
    print("=" * 60)

    dest = CHALLENGE_ROOT / "indicvoices" / "hindi"
    dest.mkdir(parents=True, exist_ok=True)

    hf = ensure_hf_hub()
    entries = []
    max_clips = 50

    try:
        print("  Downloading IndicVoices Hindi parquet shard...")
        p = hf.hf_hub_download(
            repo_id="ai4bharat/IndicVoices",
            filename="hindi/train-00000-of-00082.parquet",
            repo_type="dataset",
            token=hf_token,
        )
        import pandas as pd
        df = pd.read_parquet(p)
        print(f"  Parquet loaded: {len(df)} samples")

        downloaded = 0
        for idx in range(min(max_clips, len(df))):
            row = df.iloc[idx]
            audio_data = row.get("audio_filepath", row.get("audio", None))
            wav_bytes = None
            
            if isinstance(audio_data, dict):
                wav_bytes = audio_data.get("bytes", None)
            elif isinstance(audio_data, bytes):
                wav_bytes = audio_data

            fname = f"indic_hi_{idx:04d}.wav"
            out_file = dest / fname

            if wav_bytes is not None:
                out_file.write_bytes(wav_bytes)
            else:
                continue

            rel_path = out_file.relative_to(PROJECT_ROOT)
            sample_id = f"indic_hi_{hashlib.md5(fname.encode()).hexdigest()[:10]}"

            entries.append({
                "sample_id": sample_id,
                "file_path": str(rel_path).replace("\\", "/"),
                "dataset": "IndicVoices",
                "original_id": fname,
                "identity": "unknown",
                "manipulation_type": "none",
                "generator_tool": "none",
                "language": "hindi",
                "modality": "audio_only",
                "provenance": "huggingface/ai4bharat/IndicVoices",
                "semantic_class": "REAL",
                "visual_label": "0",
                "audio_label": "0",
                "final_label": "0",
                "split": "external_val",
                "held_out": "false",
            })
            downloaded += 1

        print(f"  Extracted {downloaded} Hindi audio clips to {dest}")

    except Exception as exc:
        print(f"  ⚠ IndicVoices processing skipped: {exc}")

    print(f"  Total IndicVoices entries: {len(entries)}")
    return entries


def write_manifest(entries: list[dict[str, str]], path: Path) -> None:
    """Write the unified master manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in sorted(entries, key=lambda e: e["sample_id"]):
            writer.writerow(entry)
    print(f"\nMaster manifest: {path}")
    print(f"  Total entries: {len(entries)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face API token (defaults to HF_TOKEN env var)")
    parser.add_argument("--skip-itw", action="store_true", help="Skip In-the-Wild download")
    parser.add_argument("--skip-mlaad", action="store_true", help="Skip MLAAD download")
    parser.add_argument("--skip-avdf", action="store_true", help="Skip AV-Deepfake1M download")
    parser.add_argument("--skip-indic", action="store_true", help="Skip IndicVoices download")
    args = parser.parse_args()

    print("=" * 60)
    print("PRAMAAN-X V2: External Dataset Downloader")
    print("=" * 60)
    print(f"  Destination: {CHALLENGE_ROOT}")
    if args.hf_token:
        print(f"  Token:       {args.hf_token[:4]}...{args.hf_token[-4:]}")
    else:
        print("  Token:       None (anonymous / environment)")

    # Login to HF
    hf = ensure_hf_hub()
    hf.login(token=args.hf_token, add_to_git_credential=False)
    print("  HF login: OK")

    all_entries: list[dict[str, str]] = []

    # 1. In-the-Wild
    if not args.skip_itw:
        all_entries.extend(download_in_the_wild(args.hf_token))

    # 2. MLAAD
    if not args.skip_mlaad:
        all_entries.extend(download_mlaad(args.hf_token))

    # 3. AV-Deepfake1M
    if not args.skip_avdf:
        all_entries.extend(download_av_deepfake1m(args.hf_token))

    # 4. IndicVoices
    if not args.skip_indic:
        all_entries.extend(download_indicvoices(args.hf_token))

    # Write manifest
    write_manifest(all_entries, MANIFEST_PATH)

    # Write download log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(all_entries),
        "by_dataset": {},
        "by_class": {},
        "by_modality": {},
        "by_split": {},
    }
    for entry in all_entries:
        for key_field, log_field in [("dataset", "by_dataset"), ("semantic_class", "by_class"),
                                      ("modality", "by_modality"), ("split", "by_split")]:
            val = entry[key_field]
            log[log_field][val] = log[log_field].get(val, 0) + 1

    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"\nDownload log: {LOG_PATH}")

    # Summary
    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total samples:  {len(all_entries)}")
    print(f"  By dataset:     {json.dumps(log['by_dataset'], indent=4)}")
    print(f"  By class:       {json.dumps(log['by_class'], indent=4)}")
    print(f"  By modality:    {json.dumps(log['by_modality'], indent=4)}")
    print(f"  By split:       {json.dumps(log['by_split'], indent=4)}")
    print(f"\nNext step: Run Phase 3 feature extraction:")
    print(f"  venv\\Scripts\\python.exe scripts/v2/03_extract_external_features.py --device cpu")


if __name__ == "__main__":
    main()
