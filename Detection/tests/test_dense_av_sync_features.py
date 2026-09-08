import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing" / "18_extract_dense_av_sync_features.py"
SPEC = importlib.util.spec_from_file_location("dense_av_sync_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DenseAVSyncFeatureTests(unittest.TestCase):
    def test_descriptor_has_fixed_finite_shape(self):
        time = np.arange(4_000, dtype=np.float32) / MODULE.SAMPLE_RATE
        segment = np.sin(2 * np.pi * 440 * time).astype(np.float32)
        descriptor = MODULE.audio_descriptor(segment, MODULE.build_mel_filterbank(np), np)
        self.assertEqual(descriptor.shape, (MODULE.AUDIO_FEATURE_DIM,))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_dense_features_follow_valid_mouth_timestamps(self):
        waveform = np.ones(MODULE.SAMPLE_RATE, dtype=np.float32)
        timestamps = np.asarray([0.0, 0.25, 0.75, 1.25])
        features, positions = MODULE.extract_dense_features(waveform, timestamps, 4_000, np)
        self.assertEqual(features.shape, (3, MODULE.AUDIO_FEATURE_DIM))
        np.testing.assert_array_equal(positions, [0, 1, 2])

    def test_context_is_fixed_and_zero_padded_at_boundaries(self):
        waveform = np.ones(100, dtype=np.float32)
        segment = MODULE.fixed_context(waveform, 0, 400, np)
        self.assertEqual(segment.shape, (400,))
        self.assertEqual(int(np.count_nonzero(segment)), 100)

    def test_validator_rejects_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            path.write_bytes(b"")
            self.assertFalse(MODULE.validate_output(path, np)[0])

    def test_empty_modalities_are_empty_not_zero_evidence(self):
        features, positions = MODULE.extract_dense_features(
            np.empty((0,), dtype=np.float32), np.asarray([0.0]), 4_000, np,
        )
        self.assertEqual(features.shape, (0, MODULE.AUDIO_FEATURE_DIM))
        self.assertEqual(positions.shape, (0,))


if __name__ == "__main__":
    unittest.main()
