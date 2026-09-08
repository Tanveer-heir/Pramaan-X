import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "17_train_av_sync.py"
SPEC = importlib.util.spec_from_file_location("av_sync_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def payload(count=3, modality="AVAILABLE"):
    return {
        "schema_version": np.asarray(MODULE.FEATURE_SCHEMA_VERSION),
        "sample_id": np.asarray("fac_test"),
        "av_modality": np.asarray(modality),
        "audio_model_sha256": np.asarray(MODULE.AUDIO_MODEL_SHA256),
        "visual_backbone": np.asarray(MODULE.VISUAL_BACKBONE),
        "timestamps_start_sec": np.arange(count, dtype=np.float32) * 4,
        "timestamps_end_sec": (np.arange(count, dtype=np.float32) + 1) * 4,
        "audio_embeddings": np.arange(count * MODULE.AUDIO_DIM, dtype=np.float32).reshape(count, MODULE.AUDIO_DIM),
        "mouth_mean_embeddings": np.arange(count * MODULE.MOUTH_DIM, dtype=np.float32).reshape(count, MODULE.MOUTH_DIM),
        "mouth_motion_embeddings": np.ones((count, MODULE.MOUTH_DIM), dtype=np.float32),
    }


class AVSyncTrainingTests(unittest.TestCase):
    def test_available_feature_loads(self):
        arrays, reason = MODULE.load_aligned_arrays(payload(), np, "fac_test")
        self.assertIsNone(reason)
        self.assertEqual(arrays[0].shape, (3, MODULE.AUDIO_DIM))

    def test_not_applicable_is_excluded(self):
        arrays, reason = MODULE.load_aligned_arrays(payload(0, "NOT_APPLICABLE"), np, "fac_test")
        self.assertIsNone(arrays)
        self.assertEqual(reason, "AV_NOT_APPLICABLE")

    def test_one_window_cannot_form_negative(self):
        arrays, reason = MODULE.load_aligned_arrays(payload(1), np, "fac_test")
        self.assertIsNone(arrays)
        self.assertEqual(reason, "INSUFFICIENT_WINDOWS_FOR_WITHIN_VIDEO_NEGATIVE")

    def test_circular_negative_is_deterministic_derangement(self):
        first = MODULE.circular_negative_indices(5, "fac_test", 42)
        second = MODULE.circular_negative_indices(5, "fac_test", 42)
        self.assertEqual(first, second)
        self.assertTrue(all(index != shifted for index, shifted in enumerate(first)))

    def test_pair_builder_balances_labels_and_shifts_mouth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fac_test.npz"
            np.savez_compressed(path, **payload())
            source = MODULE.SourceExample("fac_test", path, 3)
            audio, mouth_mean, _, labels = MODULE.build_pair_arrays(source, np, 42)
        np.testing.assert_array_equal(labels, [1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(audio[:3], audio[3:])
        self.assertTrue(all(not np.array_equal(mouth_mean[i], mouth_mean[i + 3]) for i in range(3)))

    def test_binary_metrics_perfect_ranking(self):
        metrics = MODULE.binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]), np)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
