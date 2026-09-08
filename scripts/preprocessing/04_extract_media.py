from pathlib import Path
import argparse
import subprocess

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "fakeavceleb_manifest_split.csv"
)

AUDIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "audio"
)

FRAME_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "frames"
)


def run(command):
    subprocess.run(
        command,
        check=True,
    )


def extract_audio(
    video_path,
    audio_path,
):
    audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if audio_path.exists():
        return

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),

            "-vn",

            "-ac",
            "1",

            "-ar",
            "16000",

            "-c:a",
            "pcm_s16le",

            str(audio_path),
        ]
    )


def extract_frames(
    video_path,
    frame_dir,
):
    frame_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = list(
        frame_dir.glob(
            "frame_*.jpg"
        )
    )

    if existing:
        return

    output_pattern = (
        frame_dir
        / "frame_%06d.jpg"
    )

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),

            "-vf",
            "fps=4",

            "-q:v",
            "2",

            str(output_pattern),
        ]
    )


def select_samples(
    df,
    split,
    per_class,
):
    part = df[
        df["split"] == split
    ]

    selected = []

    for class_name in sorted(
        part["semantic_class"]
        .unique()
    ):

        group = part[
            part[
                "semantic_class"
            ] == class_name
        ]

        n = min(
            per_class,
            len(group),
        )

        sampled = group.sample(
            n=n,
            random_state=42,
        )

        selected.append(
            sampled
        )

    return pd.concat(
        selected,
        ignore_index=True,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        MANIFEST_PATH
    )

    selected = select_samples(
        df,
        args.split,
        args.per_class,
    )

    print()
    print("=" * 80)
    print("SMOKE TEST SAMPLE")
    print("=" * 80)

    print(
        selected[
            [
                "sample_id",
                "semantic_class",
                "identity",
                "filename",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    for i, row in selected.iterrows():

        sample_id = row[
            "sample_id"
        ]

        video_path = (
            PROJECT_ROOT
            / row["video_path"]
        )

        audio_path = (
            AUDIO_ROOT
            / f"{sample_id}.wav"
        )

        frame_dir = (
            FRAME_ROOT
            / sample_id
        )

        print(
            f"[{i + 1}/{len(selected)}] "
            f"{sample_id}"
        )

        print(
            f"  class: "
            f"{row['semantic_class']}"
        )

        print(
            f"  source: "
            f"{video_path.name}"
        )

        try:

            extract_audio(
                video_path,
                audio_path,
            )

            extract_frames(
                video_path,
                frame_dir,
            )

        except subprocess.CalledProcessError:

            print(
                "  ERROR: FFmpeg failed."
            )

            continue

        frame_count = len(
            list(
                frame_dir.glob(
                    "frame_*.jpg"
                )
            )
        )

        print(
            f"  audio: "
            f"{audio_path}"
        )

        print(
            f"  frames: "
            f"{frame_count}"
        )

        print()

    print(
        "Smoke-test extraction complete."
    )


if __name__ == "__main__":
    main()