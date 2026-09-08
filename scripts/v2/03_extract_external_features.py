"""Phase 3a: Extract features from external challenge datasets.

Runs the existing frozen Pramaan-X feature extractors on external challenge
datasets while keeping their caches completely separate from FakeAVCeleb.

Execution strategy
------------------
Audio-only:
    XLSR-SLS

Audio-video:
    ConvNeXt
    XLSR-SLS
    Dense AV

Important:
- Samples are grouped into manifest batches so a heavy extractor is loaded once
  for several files instead of once per individual file.
- GPU-heavy branches remain sequential.
- Existing cache files are reused.
- Dense AV is CPU-side and therefore receives no --device argument.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "modern_challenge_manifest.csv"
)

VISUAL_EXTRACTOR = (
    PROJECT_ROOT
    / "scripts"
    / "preprocessing"
    / "06_extract_convnext_features.py"
)

AUDIO_EXTRACTOR = (
    PROJECT_ROOT
    / "scripts"
    / "preprocessing"
    / "13_extract_xlsr_sls_features.py"
)

DENSE_EXTRACTOR = (
    PROJECT_ROOT
    / "scripts"
    / "preprocessing"
    / "18_extract_dense_av_sync_features.py"
)


# ============================================================
# EXTERNAL CACHE ROOTS
# ============================================================

VISUAL_OUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "modern_challenge"
    / "visual_embeddings"
    / "convnext_tiny_v1"
)

AUDIO_OUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "modern_challenge"
    / "audio_embeddings"
    / "xlsr_sls_v1"
)

DENSE_OUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "modern_challenge"
    / "av_sync_features"
    / "dense_v1"
)

LOG_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "v2_logs"
)


# ============================================================
# HELPERS
# ============================================================

SUCCESS_STATES = {
    "success",
    "already_cached",
}


def resolve_device(device_arg: str) -> str:
    """Resolve auto device selection."""

    if device_arg != "auto":
        return device_arg

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

    except ImportError:
        pass

    return "cpu"


def chunked(
    rows: list[dict[str, str]],
    batch_size: int,
) -> Iterable[list[dict[str, str]]]:
    """Yield rows in batches."""

    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def media_path(row: dict[str, str]) -> Path:
    """Resolve the media path from the manifest."""

    path = Path(row["file_path"])

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def cache_path(
    root: Path,
    sample_id: str,
) -> Path:
    return root / f"{sample_id}.npz"


def required_branches(
    modality: str,
) -> list[str]:
    """Return required branches for one modality."""

    if modality == "audio_only":
        return ["audio"]

    if modality == "audio_video":
        return [
            "visual",
            "audio",
            "dense_av",
        ]

    raise ValueError(
        f"Unsupported modality: {modality}"
    )


def output_root_for_branch(
    branch: str,
) -> Path:
    if branch == "visual":
        return VISUAL_OUT

    if branch == "audio":
        return AUDIO_OUT

    if branch == "dense_av":
        return DENSE_OUT

    raise ValueError(
        f"Unknown branch: {branch}"
    )


def sample_fully_cached(
    row: dict[str, str],
) -> bool:
    """Check whether all required branch outputs exist."""

    sample_id = row["sample_id"]

    for branch in required_branches(
        row["modality"]
    ):
        path = cache_path(
            output_root_for_branch(branch),
            sample_id,
        )

        if not path.is_file():
            return False

    return True


def validate_manifest(
    rows: list[dict[str, str]],
) -> None:
    """Validate required external manifest fields."""

    required = {
        "sample_id",
        "file_path",
        "modality",
        "dataset",
    }

    if not rows:
        raise RuntimeError(
            "Manifest contains no rows."
        )

    for index, row in enumerate(
        rows,
        start=1,
    ):
        missing = [
            field
            for field in required
            if not row.get(field)
        ]

        if missing:
            raise RuntimeError(
                f"Manifest row {index} missing: "
                f"{', '.join(missing)}"
            )

        if row["modality"] not in {
            "audio_only",
            "audio_video",
        }:
            raise RuntimeError(
                f"Manifest row {index} has unsupported "
                f"modality {row['modality']!r}"
            )


def create_temp_manifest(
    rows: list[dict[str, str]],
) -> str:
    """Create one extractor-compatible manifest for a batch."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
        newline="",
    ) as tmp:

        writer = csv.DictWriter(
            tmp,
            fieldnames=[
                "sample_id",
                "video_path",
                "split",
                "semantic_class",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "video_path": str(
                        media_path(row)
                    ),
                    "split": "external",
                    "semantic_class": "UNKNOWN",
                }
            )

        return tmp.name


def stderr_tail(
    text: str | None,
    max_chars: int = 2000,
) -> str:
    """Return useful tail of subprocess output."""

    if not text:
        return ""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[-max_chars:]


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically publish JSON output."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                payload,
                handle,
                indent=2,
                allow_nan=False,
            )

            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temp_name,
            path,
        )

    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass

        raise


# ============================================================
# BRANCH COMMAND BUILDERS
# ============================================================

def build_visual_command(
    python: str,
    manifest: str,
    device: str,
    visual_batch_size: int,
) -> list[str]:
    """Build command matching 06_extract_convnext_features.py."""

    return [
        python,
        str(VISUAL_EXTRACTOR),

        "--input-mode",
        "video",

        "--manifest",
        manifest,

        "--split",
        "external",

        # IMPORTANT:
        # 06 uses --feature-root, NOT --output-root.
        "--feature-root",
        str(VISUAL_OUT),

        "--device",
        device,

        "--batch-size",
        str(visual_batch_size),
    ]


def build_audio_command(
    python: str,
    manifest: str,
    device: str,
    audio_batch_size: int,
) -> list[str]:
    """Build command matching 13_extract_xlsr_sls_features.py."""

    return [
        python,
        str(AUDIO_EXTRACTOR),

        "--input-mode",
        "video",

        "--manifest",
        manifest,

        "--split",
        "external",

        "--feature-root",
        str(AUDIO_OUT),

        "--device",
        device,

        # Current XLSR extractor explicitly permits only 1 or 2.
        "--batch-size",
        str(audio_batch_size),
    ]


def build_dense_command(
    python: str,
    manifest: str,
) -> list[str]:
    """Build command matching 18_extract_dense_av_sync_features.py."""

    return [
        python,
        str(DENSE_EXTRACTOR),

        "--input-mode",
        "video",

        "--manifest",
        manifest,

        "--split",
        "external",

        "--visual-root",
        str(VISUAL_OUT),

        "--output-root",
        str(DENSE_OUT),

        # NO --device here.
        # Dense AV extraction is CPU-side in the current implementation.
    ]


# ============================================================
# BATCH RUNNER
# ============================================================

def run_branch_batch(
    *,
    branch: str,
    rows: list[dict[str, str]],
    python: str,
    device: str,
    visual_batch_size: int,
    audio_batch_size: int,
    timeout_seconds: int,
) -> dict[str, dict[str, str]]:
    """Run one branch on one batch of samples."""

    if not rows:
        return {}

    output_root = output_root_for_branch(
        branch
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    statuses: dict[
        str,
        dict[str, str],
    ] = {}

    pending_rows: list[
        dict[str, str]
    ] = []

    # --------------------------------------------------------
    # Cache detection
    # --------------------------------------------------------

    for row in rows:
        sample_id = row["sample_id"]

        path = cache_path(
            output_root,
            sample_id,
        )

        if path.is_file():
            statuses[sample_id] = {
                "status": "already_cached",
                "reason": "already_cached",
            }
        else:
            pending_rows.append(row)

    if not pending_rows:
        return statuses

    temp_manifest = create_temp_manifest(
        pending_rows
    )

    try:
        # ----------------------------------------------------
        # Correct branch-specific CLI
        # ----------------------------------------------------

        if branch == "visual":
            command = build_visual_command(
                python,
                temp_manifest,
                device,
                visual_batch_size,
            )

        elif branch == "audio":
            command = build_audio_command(
                python,
                temp_manifest,
                device,
                audio_batch_size,
            )

        elif branch == "dense_av":
            command = build_dense_command(
                python,
                temp_manifest,
            )

        else:
            raise ValueError(
                f"Unknown branch: {branch}"
            )

        # ----------------------------------------------------
        # Run one extractor process for this manifest batch
        # ----------------------------------------------------

        try:
            result = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            process_error = ""

            if result.returncode != 0:
                process_error = (
                    stderr_tail(result.stderr)
                    or stderr_tail(result.stdout)
                    or (
                        f"extractor returned "
                        f"{result.returncode}"
                    )
                )

                print()
                print(
                    f"        {branch} subprocess "
                    f"returned {result.returncode}"
                )

                print(
                    "        Error tail:"
                )

                for line in (
                    process_error
                    .splitlines()[-8:]
                ):
                    print(
                        f"          {line}"
                    )

        except subprocess.TimeoutExpired as exc:
            result = None

            process_error = (
                f"timeout after "
                f"{timeout_seconds} seconds"
            )

            if exc.stderr:
                raw = exc.stderr

                if isinstance(raw, bytes):
                    raw = raw.decode(
                        "utf-8",
                        errors="replace",
                    )

                process_error += (
                    "\n"
                    + stderr_tail(raw)
                )

        except Exception as exc:
            result = None

            process_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        # ----------------------------------------------------
        # Validate expected outputs separately
        # ----------------------------------------------------

        for row in pending_rows:
            sample_id = row["sample_id"]

            artifact = cache_path(
                output_root,
                sample_id,
            )

            if artifact.is_file():
                statuses[sample_id] = {
                    "status": "success",
                    "reason": "success",
                }

            else:
                if process_error:
                    reason = process_error
                elif (
                    result is not None
                    and result.returncode == 0
                ):
                    reason = (
                        "extractor exited successfully "
                        "but expected NPZ was not created"
                    )
                else:
                    reason = (
                        "extractor failed and expected "
                        "NPZ was not created"
                    )

                statuses[sample_id] = {
                    "status": "failed",
                    "reason": reason,
                }

    finally:
        try:
            os.unlink(temp_manifest)
        except OSError:
            pass

    return statuses


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if hasattr(
        sys.stdout,
        "reconfigure",
    ):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

        sys.stderr.reconfigure(
            encoding="utf-8"
        )

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
    )

    parser.add_argument(
        "--modality",
        choices=[
            "audio_only",
            "audio_video",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    # Number of media FILES sent to one extractor process.
    # This can be 10 even though XLSR internal inference batch is 2.
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=10,
    )

    # ConvNeXt internal frame inference batch.
    parser.add_argument(
        "--visual-inference-batch-size",
        type=int,
        default=10,
    )

    # Current XLSR-SLS extractor only accepts 1 or 2.
    parser.add_argument(
        "--audio-inference-batch-size",
        type=int,
        choices=[1, 2],
        default=2,
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
    )

    args = parser.parse_args()

    # ========================================================
    # ARGUMENT VALIDATION
    # ========================================================

    if args.sample_batch_size < 1:
        parser.error(
            "--sample-batch-size must be >= 1"
        )

    if args.visual_inference_batch_size < 1:
        parser.error(
            "--visual-inference-batch-size must be >= 1"
        )

    if (
        args.max_samples is not None
        and args.max_samples < 1
    ):
        parser.error(
            "--max-samples must be >= 1"
        )

    if args.timeout_seconds < 1:
        parser.error(
            "--timeout-seconds must be >= 1"
        )

    # ========================================================
    # PATH VALIDATION
    # ========================================================

    if not args.manifest.is_file():
        print(
            f"ERROR: manifest not found: "
            f"{args.manifest}"
        )
        sys.exit(1)

    for extractor in [
        VISUAL_EXTRACTOR,
        AUDIO_EXTRACTOR,
        DENSE_EXTRACTOR,
    ]:
        if not extractor.is_file():
            print(
                f"ERROR: extractor missing: "
                f"{extractor}"
            )
            sys.exit(1)

    # ========================================================
    # READ MANIFEST
    # ========================================================

    with args.manifest.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    try:
        validate_manifest(rows)
    except RuntimeError as exc:
        print(
            f"ERROR: {exc}"
        )
        sys.exit(1)

    original_count = len(rows)

    # ========================================================
    # FILTER
    # ========================================================

    if args.modality != "all":
        rows = [
            row
            for row in rows
            if row["modality"]
            == args.modality
        ]

    if args.dataset_filter:
        rows = [
            row
            for row in rows
            if row["dataset"]
            == args.dataset_filter
        ]

    if args.max_samples is not None:
        rows = rows[
            :args.max_samples
        ]

    if not rows:
        print(
            "No samples matched filters."
        )
        sys.exit(0)

    device = resolve_device(
        args.device
    )

    python = sys.executable

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)
    print(
        "PRAMAAN-X V2: EXTERNAL FEATURE EXTRACTION"
    )
    print("=" * 70)

    print(
        f"Manifest total:            "
        f"{original_count}"
    )

    print(
        f"Selected:                  "
        f"{len(rows)}"
    )

    print(
        f"Device:                    "
        f"{args.device} -> {device}"
    )

    print(
        f"Media files / subprocess:  "
        f"{args.sample_batch_size}"
    )

    print(
        f"ConvNeXt GPU batch:        "
        f"{args.visual_inference_batch_size}"
    )

    print(
        f"XLSR-SLS GPU batch:        "
        f"{args.audio_inference_batch_size}"
    )

    print()

    # ========================================================
    # INITIAL STATE
    # ========================================================

    states: dict[
        str,
        dict[str, Any],
    ] = {}

    valid_rows: list[
        dict[str, str]
    ] = []

    initially_cached: dict[
        str,
        bool,
    ] = {}

    for row in rows:
        sample_id = row["sample_id"]

        path = media_path(row)

        initially_cached[
            sample_id
        ] = sample_fully_cached(row)

        states[sample_id] = {
            "sample_id": sample_id,
            "dataset": row["dataset"],
            "modality": row["modality"],
            "file_path": str(path),
            "branches": {},
            "source_error": None,
        }

        if not path.is_file():
            states[
                sample_id
            ][
                "source_error"
            ] = (
                f"file_not_found: {path}"
            )

            continue

        valid_rows.append(row)

    # ========================================================
    # PROCESS BATCHES
    # ========================================================

    batches = list(
        chunked(
            valid_rows,
            args.sample_batch_size,
        )
    )

    for batch_number, batch_rows in enumerate(
        batches,
        start=1,
    ):
        print("-" * 70)

        print(
            f"BATCH "
            f"{batch_number}/{len(batches)} "
            f"| files={len(batch_rows)}"
        )

        print("-" * 70)

        av_rows = [
            row
            for row in batch_rows
            if row["modality"]
            == "audio_video"
        ]

        audio_rows = list(
            batch_rows
        )

        # ====================================================
        # VISUAL
        # ====================================================

        if av_rows:
            print(
                f"  Visual:   "
                f"{len(av_rows)} sample(s)"
            )

            statuses = run_branch_batch(
                branch="visual",
                rows=av_rows,
                python=python,
                device=device,
                visual_batch_size=(
                    args.visual_inference_batch_size
                ),
                audio_batch_size=(
                    args.audio_inference_batch_size
                ),
                timeout_seconds=(
                    args.timeout_seconds
                ),
            )

            for sample_id, status in (
                statuses.items()
            ):
                states[
                    sample_id
                ][
                    "branches"
                ][
                    "visual"
                ] = status

            good = sum(
                status["status"]
                in SUCCESS_STATES
                for status
                in statuses.values()
            )

            print(
                f"            "
                f"{good}/{len(av_rows)} done"
            )

        else:
            print(
                "  Visual:   N/A"
            )

        # ====================================================
        # AUDIO
        # ====================================================

        print(
            f"  Audio:    "
            f"{len(audio_rows)} sample(s)"
        )

        statuses = run_branch_batch(
            branch="audio",
            rows=audio_rows,
            python=python,
            device=device,
            visual_batch_size=(
                args.visual_inference_batch_size
            ),
            audio_batch_size=(
                args.audio_inference_batch_size
            ),
            timeout_seconds=(
                args.timeout_seconds
            ),
        )

        for sample_id, status in (
            statuses.items()
        ):
            states[
                sample_id
            ][
                "branches"
            ][
                "audio"
            ] = status

        good = sum(
            status["status"]
            in SUCCESS_STATES
            for status
            in statuses.values()
        )

        print(
            f"            "
            f"{good}/{len(audio_rows)} done"
        )

        # ====================================================
        # DENSE AV
        # ====================================================

        dense_rows: list[
            dict[str, str]
        ] = []

        for row in av_rows:
            sample_id = row["sample_id"]

            visual_file = cache_path(
                VISUAL_OUT,
                sample_id,
            )

            if visual_file.is_file():
                dense_rows.append(row)

            else:
                states[
                    sample_id
                ][
                    "branches"
                ][
                    "dense_av"
                ] = {
                    "status": "failed",
                    "reason": (
                        "dense_av skipped because "
                        "visual cache is unavailable"
                    ),
                }

        if dense_rows:
            print(
                f"  Dense AV: "
                f"{len(dense_rows)} sample(s)"
            )

            statuses = run_branch_batch(
                branch="dense_av",
                rows=dense_rows,
                python=python,
                device=device,
                visual_batch_size=(
                    args.visual_inference_batch_size
                ),
                audio_batch_size=(
                    args.audio_inference_batch_size
                ),
                timeout_seconds=(
                    args.timeout_seconds
                ),
            )

            for sample_id, status in (
                statuses.items()
            ):
                states[
                    sample_id
                ][
                    "branches"
                ][
                    "dense_av"
                ] = status

            good = sum(
                status["status"]
                in SUCCESS_STATES
                for status
                in statuses.values()
            )

            print(
                f"            "
                f"{good}/{len(dense_rows)} done"
            )

        elif av_rows:
            print(
                "  Dense AV: no eligible samples"
            )
        else:
            print(
                "  Dense AV: N/A"
            )

        print()

    # ========================================================
    # FINAL ACCOUNTING
    # ========================================================

    results: dict[str, Any] = {
        "success": 0,
        "fully_cached": 0,
        "failed": 0,
        "errors": [],
        "samples": [],
    }

    for row in rows:
        sample_id = row["sample_id"]

        state = states[
            sample_id
        ]

        source_error = state[
            "source_error"
        ]

        if source_error is not None:
            final_status = "failed"

            results[
                "failed"
            ] += 1

            results[
                "errors"
            ].append(
                {
                    "sample_id": sample_id,
                    "branch": "source",
                    "reason": source_error,
                }
            )

        else:
            required = required_branches(
                row["modality"]
            )

            failed_branches = []

            for branch in required:
                branch_state = (
                    state[
                        "branches"
                    ].get(branch)
                )

                if branch_state is None:
                    failed_branches.append(
                        (
                            branch,
                            "missing branch status",
                        )
                    )

                elif (
                    branch_state[
                        "status"
                    ]
                    not in SUCCESS_STATES
                ):
                    failed_branches.append(
                        (
                            branch,
                            branch_state[
                                "reason"
                            ],
                        )
                    )

            if failed_branches:
                final_status = "failed"

                results[
                    "failed"
                ] += 1

                for (
                    branch,
                    reason,
                ) in failed_branches:
                    results[
                        "errors"
                    ].append(
                        {
                            "sample_id": (
                                sample_id
                            ),
                            "branch": branch,
                            "reason": reason,
                        }
                    )

            elif initially_cached[
                sample_id
            ]:
                final_status = (
                    "fully_cached"
                )

                results[
                    "fully_cached"
                ] += 1

            else:
                final_status = "success"

                results[
                    "success"
                ] += 1

        results[
            "samples"
        ].append(
            {
                **state,
                "status": final_status,
            }
        )

    accounted = (
        results["success"]
        + results["fully_cached"]
        + results["failed"]
    )

    # ========================================================
    # LOG
    # ========================================================

    payload = {
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "manifest": str(
            args.manifest
        ),
        "device": device,
        "sample_batch_size": (
            args.sample_batch_size
        ),
        "visual_inference_batch_size": (
            args.visual_inference_batch_size
        ),
        "audio_inference_batch_size": (
            args.audio_inference_batch_size
        ),
        "selected": len(rows),
        "accounted": accounted,
        "results": results,
    }

    log_path = (
        LOG_DIR
        / "extraction_log.json"
    )

    atomic_write_json(
        log_path,
        payload,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print(
        "EXTRACTION SUMMARY"
    )
    print("=" * 70)

    print(
        f"Selected:             "
        f"{len(rows)}"
    )

    print(
        f"Success, newly done:  "
        f"{results['success']}"
    )

    print(
        f"Fully cached:         "
        f"{results['fully_cached']}"
    )

    print(
        f"Failed:               "
        f"{results['failed']}"
    )

    print(
        f"Accounted:            "
        f"{accounted}"
    )

    print(
        f"Errors:               "
        f"{len(results['errors'])}"
    )

    print(
        f"Log:                  "
        f"{log_path}"
    )

    if results["errors"]:
        print()
        print(
            "First 10 errors:"
        )

        for error in (
            results["errors"][:10]
        ):
            reason = str(
                error["reason"]
            )

            # Show the END of stderr.
            # argparse's useful error is normally at the bottom.
            reason = reason[
                -500:
            ].replace(
                "\n",
                " | ",
            )

            print(
                f"  - "
                f"{error['sample_id']} "
                f"| {error['branch']} "
                f"| {reason}"
            )

    if results["failed"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()