import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing" / "09_extract_compression_features.py"
SPEC = importlib.util.spec_from_file_location("compression_extractor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CompressionFeatureExtractionTests(unittest.TestCase):
    def test_fraction_parser_handles_ffprobe_values(self):
        self.assertAlmostEqual(MODULE.fraction("30000/1001"), 29.97002997)
        self.assertEqual(MODULE.fraction("N/A"), 0.0)

    def test_blockiness_ratio_increases_for_block_boundaries(self):
        smooth = np.tile(np.arange(32, dtype=np.float32), (32, 1))
        blocked = smooth.copy()
        blocked[:, 8:] += 40
        blocked[:, 16:] += 40
        blocked[:, 24:] += 40
        self.assertGreater(MODULE.blockiness_ratio(blocked, np), MODULE.blockiness_ratio(smooth, np))

    def test_feature_vector_has_fixed_finite_contract(self):
        probe = {
            "bitrate": 1_000_000.0, "width": 640, "height": 480, "fps": 25.0,
            "keyframe_ratio": 0.04, "gop_mean_seconds": 1.0, "gop_std_seconds": 0.0,
            "gop_max_seconds": 1.0, "packet_size_cv": 0.5, "packet_size_p90_over_mean": 1.5,
            "frame_interval_cv": 0.0, "i_frame_ratio": 0.04, "p_frame_ratio": 0.7, "b_frame_ratio": 0.26,
        }
        frame = {
            "blockiness_ratio_mean": 1.2, "blockiness_ratio_std": 0.1,
            "laplacian_variance_mean": 100.0, "laplacian_variance_cv": 0.2,
            "noise_residual_mean": 0.03, "noise_residual_cv": 0.1,
            "near_duplicate_rate": 0.0, "frame_difference_mean": 0.1, "frame_difference_cv": 0.2,
        }
        features = MODULE.build_feature_vector(probe, frame, np)
        self.assertEqual(features.shape, (MODULE.FEATURE_DIM,))
        self.assertTrue(np.isfinite(features).all())

    def test_validator_rejects_truncated_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.npz"
            path.write_bytes(b"")
            valid, reason = MODULE.validate_feature_file(path, np)
            self.assertFalse(valid)
            self.assertIn("unreadable npz", reason)


if __name__ == "__main__":
    unittest.main()
