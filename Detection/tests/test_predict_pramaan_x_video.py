import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inference" / "predict_pramaan_x_video.py"
SPEC = importlib.util.spec_from_file_location("pramaan_x_video", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RawVideoWrapperTests(unittest.TestCase):
    def test_generated_sample_id_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.write_bytes(b"video bytes")
            self.assertEqual(MODULE.raw_sample_id(video, None), MODULE.raw_sample_id(video, None))
            self.assertTrue(MODULE.raw_sample_id(video, None).startswith("raw_"))
            self.assertEqual(MODULE.raw_sample_id(video, "demo_video"), "demo_video")

    def test_manifest_uses_explicit_absolute_video_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, manifest = root / "input.mp4", root / "manifest.csv"
            video.write_bytes(b"x")
            MODULE.write_manifest(manifest, "raw_example", video)
            content = manifest.read_text(encoding="utf-8")
            self.assertIn("raw_example", content)
            self.assertIn(str(video.resolve()), content)

    def test_cuda_ordinal_is_compatible_with_audio_extractor(self):
        self.assertEqual(MODULE.device_for_audio("cuda:0"), "cuda")
        self.assertEqual(MODULE.device_for_audio("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
