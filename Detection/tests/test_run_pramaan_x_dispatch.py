"""Focused dispatch tests for the public Detection CLI boundary."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inference" / "run_pramaan_x.py"
SPEC = importlib.util.spec_from_file_location("run_pramaan_x", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        fusion_checkpoint=tmp_path / "fusion.pth",
        device="cuda",
        keep_temporary_features=False,
        face_landmarker=None,
    )


def test_image_dispatches_to_native_image_boundary(tmp_path, monkeypatch):
    image = tmp_path / "evidence.jpg"
    output = tmp_path / "result.json"
    image.write_bytes(b"image")
    monkeypatch.setattr(MODULE, "IMAGE_PREDICTOR", tmp_path / "analyse_pramaan_x_image.py")
    MODULE.IMAGE_PREDICTOR.write_text("", encoding="utf-8")

    command = MODULE.command_for_media(args(tmp_path), image, output, MODULE.media_type(image))

    assert MODULE.media_type(image) == "image"
    assert command == [sys.executable, str(MODULE.IMAGE_PREDICTOR), str(image), "--output", str(output)]


def test_video_dispatch_preserves_raw_video_predictor_arguments(tmp_path, monkeypatch):
    video = tmp_path / "evidence.mp4"
    output = tmp_path / "result.json"
    video.write_bytes(b"video")
    fusion = tmp_path / "fusion.pth"
    fusion.write_bytes(b"checkpoint")
    raw_predictor = tmp_path / "predict_pramaan_x_video.py"
    raw_predictor.write_text("", encoding="utf-8")
    monkeypatch.setattr(MODULE, "RAW_VIDEO_PREDICTOR", raw_predictor)
    values = args(tmp_path)

    command = MODULE.command_for_media(values, video, output, MODULE.media_type(video))

    assert MODULE.media_type(video) == "video"
    assert command == [
        sys.executable,
        str(raw_predictor),
        "--video",
        str(video),
        "--fusion-checkpoint",
        str(fusion),
        "--output",
        str(output),
        "--device",
        "cuda",
    ]


def test_unsupported_media_extension_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="Unsupported media type"):
        MODULE.media_type(tmp_path / "evidence.gif")
