import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing" / "07_extract_blink_features.py"
SPEC = importlib.util.spec_from_file_location("blink_extractor", SCRIPT)
blink_extractor = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = blink_extractor
SPEC.loader.exec_module(blink_extractor)


class BlinkFeatureExtractionTests(unittest.TestCase):
    def test_feature_vector_has_documented_small_dimension(self):
        vector, mean = blink_extractor.feature_vector(0.20, 0.30, previous_mean=0.15, delta_seconds=0.25)
        self.assertEqual(len(vector), blink_extractor.FEATURE_DIM)
        self.assertAlmostEqual(mean, 0.25)
        self.assertAlmostEqual(vector[-2], 0.4)
        self.assertAlmostEqual(vector[-1], 0.4)

    def test_validator_accepts_explicit_not_applicable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(blink_extractor.SCHEMA_VERSION),
                sample_id=np.asarray("sample"),
                availability=np.asarray("NOT_APPLICABLE"),
                not_applicable_reason=np.asarray("INSUFFICIENT_VALID_EYE_FRAMES:0/4"),
                timestamps_sec=np.empty((0,), dtype=np.float32),
                source_frame_indices=np.empty((0,), dtype=np.int32),
                features=np.empty((0, blink_extractor.FEATURE_DIM), dtype=np.float32),
                feature_names=np.asarray(blink_extractor.FEATURE_NAMES),
            )
            self.assertEqual(blink_extractor.validate_feature_file(path, np), (True, None))

    def test_validator_rejects_truncated_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.npz"
            path.write_bytes(b"")
            valid, reason = blink_extractor.validate_feature_file(path, np)
            self.assertFalse(valid)
            self.assertIn("unreadable npz", reason)


if __name__ == "__main__":
    unittest.main()
