"""Synthetic reference models must not identify external evidence by default."""

from pathlib import Path

from PIL import Image

from prnu_attribution.api import _status
from prnu_attribution.model import predict


def test_synthetic_model_is_inconclusive_for_external_image(tmp_path):
    image = tmp_path / "external.png"
    Image.new("RGB", (64, 64), (40, 60, 80)).save(image)

    result = predict(image, Path(__file__).parent / "models", mode="device", allow_demo_model=False)

    assert result["case"]["trained_model_allowed"] is False
    assert result["device_attribution"]["prediction"] is None
    assert result["device_attribution"]["primary_method"] == "inconclusive"


def test_health_marks_checked_in_synthetic_model_unusable():
    root = Path(__file__).parent
    status = _status(root / "dataset", root / "models")

    assert status["synthetic_demo_model"] is True
    assert status["reference_model_usable"] is False
    assert status["capabilities"]["video_attribution"]["available"] is False
