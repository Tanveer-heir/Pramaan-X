from pathlib import Path
import hashlib
import re

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fakeavceleb"
    / "FakeAVCeleb_v1.2"
    / "FakeAVCeleb_v1.2"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "fakeavceleb_manifest.csv"
)


# ---------------------------------------------------------
# FakeAVCeleb directory labels
# ---------------------------------------------------------

CLASS_INFO = {
    "RealVideo-RealAudio": {
        "visual_label": 0,
        "audio_label": 0,
        "final_label": 0,
        "semantic_class": "REAL",
    },
    "FakeVideo-RealAudio": {
        "visual_label": 1,
        "audio_label": 0,
        "final_label": 1,
        "semantic_class": "VISUAL_MANIPULATION",
    },
    "RealVideo-FakeAudio": {
        "visual_label": 0,
        "audio_label": 1,
        "final_label": 2,
        "semantic_class": "AUDIO_MANIPULATION",
    },
    "FakeVideo-FakeAudio": {
        "visual_label": 1,
        "audio_label": 1,
        "final_label": 3,
        "semantic_class": "AUDIO_VISUAL_MANIPULATION",
    },
}


def create_sample_id(relative_path: str) -> str:
    """
    Stable sample ID based on path.

    This means sample IDs remain unchanged even if we rebuild
    the manifest later.
    """
    digest = hashlib.sha1(
        relative_path.encode("utf-8")
    ).hexdigest()[:12]

    return f"fac_{digest}"


def find_identity(relative_parts):
    """
    Find directory such as id00145.
    """
    for part in relative_parts:
        if re.fullmatch(r"id\d+", part):
            return part

    return None


def extract_filename_metadata(filename: str):
    """
    Extract useful hints from FakeAVCeleb filenames.

    Example:
        00043_id01005_wavtolip.mp4
        00043_id01004_867Wlj7Gw68_faceswap.mp4
    """

    stem = Path(filename).stem
    lower = stem.lower()

    # First numerical token often identifies source/content group.
    pieces = stem.split("_")

    content_id = pieces[0] if pieces else None

    # All idXXXXX references in filename
    linked_identities = re.findall(r"id\d+", stem)

    # Manipulation method hint
    if "faceswap" in lower:
        method_hint = "faceswap"

    elif "wav2lip" in lower or "wavtolip" in lower:
        method_hint = "wav2lip"

    elif "fsgan" in lower:
        method_hint = "fsgan"

    elif "rtvc" in lower:
        method_hint = "rtvc"

    elif "sv2tts" in lower:
        method_hint = "sv2tts"

    else:
        method_hint = "unknown"

    return content_id, linked_identities, method_hint


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"FakeAVCeleb root not found:\n{DATA_ROOT}"
        )

    rows = []

    for directory_class, labels in CLASS_INFO.items():

        class_root = DATA_ROOT / directory_class

        if not class_root.exists():
            print(f"WARNING: missing directory: {class_root}")
            continue

        video_paths = sorted(class_root.rglob("*.mp4"))

        print(
            f"{directory_class}: "
            f"{len(video_paths):,} videos"
        )

        for video_path in video_paths:

            relative_to_class = video_path.relative_to(
                class_root
            )

            parts = relative_to_class.parts

            # Expected:
            # race / gender / idXXXXX / video.mp4
            race = (
                parts[0]
                if len(parts) >= 1
                else None
            )

            gender = (
                parts[1]
                if len(parts) >= 2
                else None
            )

            identity = find_identity(parts)

            relative_project_path = (
                video_path
                .relative_to(PROJECT_ROOT)
                .as_posix()
            )

            content_id, linked_ids, method_hint = (
                extract_filename_metadata(
                    video_path.name
                )
            )

            sample_id = create_sample_id(
                relative_project_path
            )

            rows.append(
                {
                    "sample_id": sample_id,

                    "video_path": relative_project_path,

                    "filename": video_path.name,

                    "dataset": "FakeAVCeleb",

                    "directory_class": directory_class,

                    "semantic_class": (
                        labels["semantic_class"]
                    ),

                    "visual_label": (
                        labels["visual_label"]
                    ),

                    "audio_label": (
                        labels["audio_label"]
                    ),

                    "final_label": (
                        labels["final_label"]
                    ),

                    "identity": identity,

                    "race_group": race,

                    "gender_group": gender,

                    "content_id": content_id,

                    "linked_identities": ";".join(
                        linked_ids
                    ),

                    "method_hint": method_hint,
                }
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No MP4 files discovered."
        )

    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------

    if df["sample_id"].duplicated().any():
        duplicates = df[
            df["sample_id"].duplicated(
                keep=False
            )
        ]

        print(duplicates)

        raise RuntimeError(
            "Duplicate sample IDs detected."
        )

    if df["video_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate video paths detected."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FAKEAVCELEB MANIFEST")
    print("=" * 70)

    print(
        f"Total videos: {len(df):,}"
    )

    print(
        f"Unique identities: "
        f"{df['identity'].nunique():,}"
    )

    print()
    print("Class counts:")
    print(
        df["semantic_class"]
        .value_counts()
        .to_string()
    )

    print()
    print("Visual labels:")
    print(
        df["visual_label"]
        .value_counts()
        .to_string()
    )

    print()
    print("Audio labels:")
    print(
        df["audio_label"]
        .value_counts()
        .to_string()
    )

    print()
    print("Method hints:")
    print(
        df["method_hint"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Missing identity values: "
        f"{df['identity'].isna().sum()}"
    )

    print()
    print(
        f"Saved manifest to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()