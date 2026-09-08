import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "14_train_audio_classifier.py"
SPEC = importlib.util.spec_from_file_location("audio_classifier_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AudioClassifierTrainingTests(unittest.TestCase):
    def test_available_sequence_loads_without_zero_filling(self):
        payload = {
            "schema_version": np.asarray(MODULE.FEATURE_SCHEMA_VERSION),
            "model_sha256": np.asarray(MODULE.MODEL_SHA256),
            "audio_modality": np.asarray("AVAILABLE"),
            "embeddings": np.ones((2, MODULE.INPUT_DIM), dtype=np.float16),
        }
        sequence, reason = MODULE.load_sequence(payload, np)
        self.assertIsNone(reason)
        self.assertEqual(sequence.shape, (2, MODULE.INPUT_DIM))

    def test_not_applicable_audio_is_excluded(self):
        payload = {
            "schema_version": np.asarray(MODULE.FEATURE_SCHEMA_VERSION),
            "model_sha256": np.asarray(MODULE.MODEL_SHA256),
            "audio_modality": np.asarray("NOT_APPLICABLE"),
            "embeddings": np.empty((0, MODULE.INPUT_DIM), dtype=np.float16),
        }
        sequence, reason = MODULE.load_sequence(payload, np)
        self.assertIsNone(sequence)
        self.assertEqual(reason, "AUDIO_NOT_APPLICABLE")

    def test_feature_provenance_is_enforced(self):
        payload = {
            "schema_version": np.asarray(MODULE.FEATURE_SCHEMA_VERSION),
            "model_sha256": np.asarray("bad"),
            "audio_modality": np.asarray("AVAILABLE"),
            "embeddings": np.ones((1, MODULE.INPUT_DIM), dtype=np.float16),
        }
        self.assertEqual(MODULE.load_sequence(payload, np)[1], "checkpoint_digest_mismatch")

    def test_binary_metrics_are_exact_for_perfect_ranking(self):
        metrics = MODULE.binary_metrics(
            np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]), np,
        )
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)

    def test_auc_tie_handling(self):
        metrics = MODULE.binary_metrics(
            np.asarray([0, 1, 0, 1]), np.asarray([0.5, 0.5, 0.5, 0.5]), np,
        )
        self.assertAlmostEqual(metrics["roc_auc"], 0.5)
        self.assertAlmostEqual(metrics["average_precision"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
