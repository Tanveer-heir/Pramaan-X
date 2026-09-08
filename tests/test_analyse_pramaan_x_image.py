import importlib.util
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inference" / "analyse_pramaan_x_image.py"
SPEC = importlib.util.spec_from_file_location("analyse_pramaan_x_image", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_model_result():
    return {
        "assessment": {
            "label": "SUSPICIOUS",
            "confidence_level": "MEDIUM",
            "summary": "The image contains a small number of ambiguous visual inconsistencies.",
        },
        "visual_findings": [
            {
                "category": "lighting",
                "severity": "MEDIUM",
                "region": "face",
                "finding": "Facial illumination is difficult to reconcile with the visible background shadow.",
                "region_bbox": {"x": 0.10, "y": 0.20, "width": 0.40, "height": 0.30},
            }
        ],
        "supporting_signals": {key: "INCONCLUSIVE" for key in MODULE.SIGNAL_KEYS},
    }


class ImageAnalysisTests(unittest.TestCase):
    def make_image(self, directory: str, suffix: str = ".png") -> Path:
        path = Path(directory) / f"sample{suffix}"
        Image.new("RGB", (32, 24), (128, 64, 32)).save(path)
        return path

    def test_metadata_and_hash_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            image = self.make_image(directory)
            first = MODULE.inspect_image(image)
            second = MODULE.inspect_image(image)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["width"], 32)
            self.assertEqual(first["height"], 24)
            self.assertEqual(first["format"], "PNG")
            self.assertFalse(first["exif"]["present"])

    def test_unsupported_extension_fails_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gif"
            path.write_bytes(b"not an accepted input")
            with self.assertRaisesRegex(ValueError, "Unsupported image extension"):
                MODULE.inspect_image(path)

    def test_valid_model_result_is_strictly_normalized(self):
        result = MODULE.validate_model_result(valid_model_result())
        self.assertEqual(result["assessment"]["label"], "SUSPICIOUS")
        self.assertEqual(len(result["visual_findings"]), 1)
        self.assertEqual(result["visual_findings"][0]["region_bbox"]["width"], 0.4)
        self.assertEqual(set(result["supporting_signals"]), set(MODULE.SIGNAL_KEYS))

    def test_invalid_region_bbox_is_rejected(self):
        value = valid_model_result()
        value["visual_findings"][0]["region_bbox"] = {"x": 0.8, "y": 0.2, "width": 0.4, "height": 0.2}
        with self.assertRaisesRegex(ValueError, "invalid region_bbox"):
            MODULE.validate_model_result(value)

    def test_invalid_label_is_rejected(self):
        value = valid_model_result()
        value["assessment"]["label"] = "DEFINITELY_FAKE"
        with self.assertRaisesRegex(ValueError, "Invalid image assessment label"):
            MODULE.validate_model_result(value)

    def test_invalid_confidence_is_rejected(self):
        value = valid_model_result()
        value["assessment"]["confidence_level"] = "0.99"
        with self.assertRaisesRegex(ValueError, "Invalid image confidence"):
            MODULE.validate_model_result(value)

    def test_mocked_vlm_response_and_one_bounded_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            image = self.make_image(directory)
            calls = []

            def request_fn(payload, api_key, base_url, timeout):
                calls.append((payload, api_key, base_url, timeout))
                if len(calls) == 1:
                    raise ValueError("malformed JSON")
                return valid_model_result()

            result = MODULE.analyse_image_with_vlm(
                image,
                "test-vision-model",
                api_key="secret-value",
                request_fn=request_fn,
            )
            self.assertEqual(result["assessment"]["label"], "SUSPICIOUS")
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1], "secret-value")
            self.assertNotIn("secret-value", str(result))

    def test_missing_credential_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            image = self.make_image(directory)
            with self.assertRaisesRegex(RuntimeError, "credential is missing"):
                MODULE.analyse_image_with_vlm(image, "test-vision-model", api_key="")

    def test_result_contains_non_forensic_limitations(self):
        with tempfile.TemporaryDirectory() as directory:
            image = self.make_image(directory)
            result = MODULE.build_result(image, MODULE.validate_model_result(valid_model_result()), "test-model")
            self.assertEqual(result["media_type"], "image")
            self.assertTrue(result["limitations"])
            self.assertIn("calibrated forensic probability", result["limitations"][0])


    def test_dotenv_loads_kimi_key_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                'KIMI_API_KEY="from-file"\nKIMI_BASE_URL=https://api.moonshot.ai/v1\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                MODULE.load_env_file(env_file)
                self.assertEqual(os.environ["KIMI_API_KEY"], "from-file")
                self.assertEqual(os.environ["KIMI_BASE_URL"], "https://api.moonshot.ai/v1")
            with patch.dict(os.environ, {"KIMI_API_KEY": "existing"}, clear=True):
                MODULE.load_env_file(env_file)
                self.assertEqual(os.environ["KIMI_API_KEY"], "existing")

    def test_kimi_defaults_are_explicit(self):
        self.assertEqual(MODULE.DEFAULT_KIMI_BASE_URL, "https://api.moonshot.ai/v1")
        self.assertEqual(MODULE.DEFAULT_KIMI_MODEL, "kimi-k3")


if __name__ == "__main__":
    unittest.main()
