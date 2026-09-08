"""Tests for Video Keyframe Extractor."""

import pytest
from src.source_attribution.reverse_search.keyframe_extractor import VideoKeyframeExtractor

def test_is_video():
    extractor = VideoKeyframeExtractor()
    assert extractor.is_video("incident.mp4") is True
    assert extractor.is_video("incident.MOV") is True
    assert extractor.is_video("incident.avi") is True
    assert extractor.is_video("screenshot.png") is False
    assert extractor.is_video("photo.jpg") is False

def test_extract_keyframes_nonexistent():
    extractor = VideoKeyframeExtractor()
    assert extractor.extract_keyframes("non_existent_file.mp4") == []

def test_invalid_video_does_not_fabricate_keyframe(tmp_path):
    extractor = VideoKeyframeExtractor(output_dir=str(tmp_path / "keyframes"))
    # Create an actual dummy video file on disk
    dummy_video = tmp_path / "test_clip.mp4"
    dummy_video.write_bytes(b"dummy video binary content for testing")

    frames = extractor.extract_keyframes(str(dummy_video))
    assert frames == []
    assert list((tmp_path / "keyframes").glob("*.jpg")) == []
