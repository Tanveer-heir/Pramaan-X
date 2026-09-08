import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing" / "13_extract_xlsr_sls_features.py"
SPEC = importlib.util.spec_from_file_location("xlsr_sls_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class XLSRSLSFeatureExtractionTests(unittest.TestCase):
    def test_audio_label_mapping_matches_fakeavceleb_semantics(self):
        self.assertNotIn("REAL", MODULE.AUDIO_FAKE_CLASSES)
        self.assertNotIn("VISUAL_MANIPULATION", MODULE.AUDIO_FAKE_CLASSES)
        self.assertIn("AUDIO_MANIPULATION", MODULE.AUDIO_FAKE_CLASSES)
        self.assertIn("AUDIO_VISUAL_MANIPULATION", MODULE.AUDIO_FAKE_CLASSES)

    def test_short_audio_is_repeat_padded(self):
        waveform = np.asarray([0.25, -0.5], dtype=np.float32)
        windows, starts, valid = MODULE.make_windows(waveform, MODULE.WINDOW_SAMPLES, np)
        self.assertEqual(windows.shape, (1, MODULE.WINDOW_SAMPLES))
        np.testing.assert_array_equal(windows[0, :6], [0.25, -0.5, 0.25, -0.5, 0.25, -0.5])
        np.testing.assert_array_equal(starts, [0])
        np.testing.assert_array_equal(valid, [2])

    def test_empty_audio_is_not_zero_valued_evidence(self):
        windows, starts, valid = MODULE.make_windows(np.empty((0,), dtype=np.float32), MODULE.WINDOW_SAMPLES, np)
        self.assertEqual(windows.shape, (0, MODULE.WINDOW_SAMPLES))
        self.assertEqual(starts.size, 0)
        self.assertEqual(valid.size, 0)

    def test_feature_validator_accepts_available_and_not_applicable(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, count, modality in (("available", 2, "AVAILABLE"), ("missing", 0, "NOT_APPLICABLE")):
                path = Path(directory) / f"{name}.npz"
                np.savez_compressed(path, **self._payload(count, modality))
                self.assertEqual(MODULE.validate_feature_file(path, np), (True, None))

    def test_feature_validator_rejects_corrupt_and_zero_filled_missing_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.npz"
            corrupt.write_bytes(b"")
            self.assertFalse(MODULE.validate_feature_file(corrupt, np)[0])
            invalid = Path(directory) / "invalid.npz"
            np.savez_compressed(invalid, **self._payload(1, "NOT_APPLICABLE"))
            self.assertFalse(MODULE.validate_feature_file(invalid, np)[0])

    def test_cache_validation_binds_model_hop_and_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(path, **self._payload(1, "AVAILABLE"))
            self.assertTrue(MODULE.validate_feature_file(
                path, np, expected_model_sha256="a" * 64,
                expected_hop_samples=MODULE.WINDOW_SAMPLES, expected_sample_id="fac_test",
            )[0])
            self.assertFalse(MODULE.validate_feature_file(path, np, expected_model_sha256="b" * 64)[0])
            self.assertFalse(MODULE.validate_feature_file(path, np, expected_hop_samples=123)[0])
            self.assertFalse(MODULE.validate_feature_file(path, np, expected_sample_id="fac_other")[0])

    @staticmethod
    def _payload(count, modality):
        return {
            "schema_version": np.asarray(MODULE.SCHEMA_VERSION),
            "backbone": np.asarray(MODULE.BACKBONE_NAME),
            "model_sha256": np.asarray("a" * 64),
            "sample_id": np.asarray("fac_test"),
            "audio_modality": np.asarray(modality),
            "sample_rate": np.asarray(MODULE.SAMPLE_RATE, dtype=np.int32),
            "window_samples": np.asarray(MODULE.WINDOW_SAMPLES, dtype=np.int32),
            "hop_samples": np.asarray(MODULE.WINDOW_SAMPLES, dtype=np.int32),
            "source_audio_samples": np.asarray(0, dtype=np.int64),
            "window_start_samples": np.arange(count, dtype=np.int64),
            "window_valid_samples": np.full(count, MODULE.WINDOW_SAMPLES, dtype=np.int32),
            "timestamps_start_sec": np.arange(count, dtype=np.float32),
            "timestamps_end_sec": np.arange(count, dtype=np.float32) + 1,
            "embeddings": np.zeros((count, MODULE.EMBEDDING_DIM), dtype=np.float16),
            "pretrained_log_probabilities": np.zeros((count, 2), dtype=np.float32),
        }


if __name__ == "__main__":
    unittest.main()
