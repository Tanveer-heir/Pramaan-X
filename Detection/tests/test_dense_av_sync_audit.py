import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "20_audit_dense_av_sync.py"
SPEC = importlib.util.spec_from_file_location("dense_av_sync_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainingStub:
    DEFAULT_SHIFT_FRAMES = (2, 4)

    @staticmethod
    def binary_metrics(labels, probabilities, np_module):
        training_path = Path(__file__).resolve().parents[1] / "scripts" / "training" / "19_train_dense_av_sync.py"
        spec = importlib.util.spec_from_file_location("dense_training_metric_stub", training_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.binary_metrics(labels, probabilities, np_module)


class DenseAVSyncAuditTests(unittest.TestCase):
    def test_overlap_reports_unique_shared_keys(self):
        count, examples = MODULE.overlap([{"key": "a"}, {"key": "a"}], [{"key": "a"}], lambda item: item["key"])
        self.assertEqual(count, 1)
        self.assertEqual(examples, ["a"])

    def test_offset_breakdown_pairs_each_negative_with_preceding_positive(self):
        scored = {
            "labels": np.asarray([1, 0, 1, 0, 1, 0, 1, 0]),
            "probabilities": np.asarray([0.9, 0.1, 0.8, 0.2, 0.85, 0.15, 0.75, 0.25]),
            "offsets": np.asarray([0, 2, 0, -2, 0, 4, 0, -4]),
        }
        breakdown = MODULE.offset_breakdown(scored, TrainingStub, np)
        self.assertAlmostEqual(breakdown["500ms"]["roc_auc"], 1.0)
        self.assertAlmostEqual(breakdown["1000ms"]["roc_auc"], 1.0)
        self.assertEqual(breakdown["direction_negative"]["negative_pairs"], 2)

    def test_identity_bootstrap_is_deterministic(self):
        scored = {
            "labels": np.asarray([1, 0, 1, 0, 1, 0, 1, 0]),
            "probabilities": np.asarray([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
            "sample_ids": np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"]),
        }
        first = MODULE.identity_bootstrap_ci(scored, TrainingStub, np, 100, 42)
        second = MODULE.identity_bootstrap_ci(scored, TrainingStub, np, 100, 42)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["lower_95"], 1.0)


if __name__ == "__main__":
    unittest.main()
