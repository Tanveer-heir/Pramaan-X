"""
Unit tests for Sub-Image Homography & Crop Detection Engine
Chandigarh Police Hackathon - A* Attribution Research Track
"""

import os
import sys
import tempfile
import cv2
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.research.astar_attribution.subimage_matcher import (
    SubImageMatcher,
    MatchResult,
)


@pytest.fixture
def test_image_path():
    """Locate available test image or construct one."""
    candidate_paths = [
        os.path.join(PROJECT_ROOT, "data", "cache", "investigation", "web_media_2ccdf57e36.jpg"),
        os.path.join(PROJECT_ROOT, "data", "sample_media", "test3.png"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            return p

    # Fallback to creating a textured synthetic test image if none exists
    temp_dir = tempfile.gettempdir()
    fallback_path = os.path.join(temp_dir, "synthetic_test_base.jpg")
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Add distinctive shapes and text
    cv2.circle(img, (200, 200), 80, (0, 255, 0), -1)
    cv2.rectangle(img, (350, 100), (500, 300), (0, 0, 255), -1)
    cv2.putText(img, "POLICE FORENSICS", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(fallback_path, img)
    return fallback_path


def test_subimage_matcher_synthetic_crop(test_image_path, tmp_path):
    """
    Test that a synthetic central 60% crop is identified as 'target_is_crop_of_candidate'
    with accurate bounding box and scale factor.
    """
    img = cv2.imread(test_image_path)
    assert img is not None, f"Failed to load test image from {test_image_path}"

    h, w = img.shape[:2]
    # Central 60% crop: bounds from 20% to 80%
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    expected_w = x2 - x1
    expected_h = y2 - y1

    crop_img = img[y1:y2, x1:x2]
    crop_path = str(tmp_path / "synthetic_crop.jpg")
    cv2.imwrite(crop_path, crop_img)

    matcher = SubImageMatcher()
    result: MatchResult = matcher.compare_images(crop_path, test_image_path)

    print(f"\nCrop Test Result: {result}")

    # Core requirements assertions
    assert result.is_match is True, "Expected is_match to be True"
    assert result.is_partial_crop is True, "Expected is_partial_crop to be True"
    assert result.crop_type == "target_is_crop_of_candidate", (
        f"Expected crop_type 'target_is_crop_of_candidate', got '{result.crop_type}'"
    )
    assert result.bounding_box is not None, "Expected bounding box to be detected"
    assert len(result.bounding_box) == 4, "Expected 4 elements in bounding box [x, y, w, h]"

    bx, by, bw, bh = result.bounding_box
    # Verify bounding box matches ground truth within 4 pixel tolerance
    assert abs(bx - x1) <= 4, f"Detected x ({bx}) deviates from expected ({x1})"
    assert abs(by - y1) <= 4, f"Detected y ({by}) deviates from expected ({y1})"
    assert abs(bw - expected_w) <= 4, f"Detected width ({bw}) deviates from expected ({expected_w})"
    assert abs(bh - expected_h) <= 4, f"Detected height ({bh}) deviates from expected ({expected_h})"

    # Scale factor should be ~1.0 for unscaled crop
    assert 0.95 <= result.scale_factor <= 1.05, f"Unexpected scale factor: {result.scale_factor}"

    # Inliers and threshold checks
    assert result.inlier_count >= 15, f"Expected >= 15 inliers, got {result.inlier_count}"
    assert result.inlier_ratio >= 0.20, f"Expected >= 0.20 inlier ratio, got {result.inlier_ratio}"
    assert result.similarity_score > 0.5, f"Expected high similarity score, got {result.similarity_score}"


def test_subimage_matcher_reverse_crop(test_image_path, tmp_path):
    """
    Test when Target is original and Candidate is the crop:
    Should detect 'candidate_is_crop_of_target'.
    """
    img = cv2.imread(test_image_path)
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)

    crop_img = img[y1:y2, x1:x2]
    crop_path = str(tmp_path / "synthetic_crop_reverse.jpg")
    cv2.imwrite(crop_path, crop_img)

    matcher = SubImageMatcher()
    result = matcher.compare_images(test_image_path, crop_path)

    assert result.is_match is True
    assert result.is_partial_crop is True
    assert result.crop_type == "candidate_is_crop_of_target"
    assert result.bounding_box is not None

    bx, by, bw, bh = result.bounding_box
    assert abs(bx - x1) <= 4
    assert abs(by - y1) <= 4


def test_subimage_matcher_exact_clone(test_image_path):
    """Test that identical images are detected with crop_type == 'exact' and PDQ distance <= 30."""
    matcher = SubImageMatcher()
    result = matcher.compare_images(test_image_path, test_image_path)

    assert result.is_match is True
    assert result.is_partial_crop is False
    assert result.crop_type == "exact"
    assert result.pdq_distance <= 30
    assert result.scale_factor == pytest.approx(1.0, abs=0.05)


def test_subimage_matcher_unrelated_images(test_image_path, tmp_path):
    """Test that unrelated images produce is_match == False and crop_type == 'none'."""
    unrelated = np.zeros((300, 300, 3), dtype=np.uint8)
    # Add random noise pattern
    cv2.randu(unrelated, 0, 255)
    unrelated_path = str(tmp_path / "unrelated_noise.jpg")
    cv2.imwrite(unrelated_path, unrelated)

    matcher = SubImageMatcher()
    result = matcher.compare_images(test_image_path, unrelated_path)

    assert result.is_match is False
    assert result.is_partial_crop is False
    assert result.crop_type == "none"
    assert result.bounding_box is None


def test_subimage_matcher_visualization(test_image_path, tmp_path):
    """Test the visualizer method."""
    img = cv2.imread(test_image_path)
    h, w = img.shape[:2]
    crop = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    crop_path = str(tmp_path / "viz_crop.jpg")
    cv2.imwrite(crop_path, crop)

    output_viz_path = str(tmp_path / "annotated_match.jpg")

    matcher = SubImageMatcher()
    res = matcher.compare_images(crop_path, test_image_path)
    success = matcher.visualize_match(crop_path, test_image_path, output_viz_path, res)

    assert success is True
    assert os.path.isfile(output_viz_path)
    assert os.path.getsize(output_viz_path) > 0


if __name__ == "__main__":
    import shutil

    print("Running subimage_matcher tests standalone...")
    base_img = os.path.join(PROJECT_ROOT, "data", "cache", "investigation", "web_media_2ccdf57e36.jpg")
    temp_dir = tempfile.mkdtemp()
    try:
        from pathlib import Path
        temp_path = Path(temp_dir)
        print(f"Using base image: {base_img}")

        print("1. Testing synthetic crop...")
        test_subimage_matcher_synthetic_crop(base_img, temp_path)
        print("   -> PASSED")

        print("2. Testing reverse crop...")
        test_subimage_matcher_reverse_crop(base_img, temp_path)
        print("   -> PASSED")

        print("3. Testing exact clone...")
        test_subimage_matcher_exact_clone(base_img)
        print("   -> PASSED")

        print("4. Testing unrelated images...")
        test_subimage_matcher_unrelated_images(base_img, temp_path)
        print("   -> PASSED")

        print("5. Testing visualization...")
        test_subimage_matcher_visualization(base_img, temp_path)
        print("   -> PASSED")

        print("\nALL STANDALONE TESTS PASSED SUCCESSFULLY!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
