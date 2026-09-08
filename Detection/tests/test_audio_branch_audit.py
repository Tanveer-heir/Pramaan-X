import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "15_audit_audio_branch.py"
SPEC = importlib.util.spec_from_file_location("audio_branch_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AudioBranchAuditTests(unittest.TestCase):
    def test_feature_fingerprint_is_deterministic_and_content_bound(self):
        payload = {
            "source_audio_samples": np.asarray(10, dtype=np.int64),
            "window_start_samples": np.asarray([0], dtype=np.int64),
            "window_valid_samples": np.asarray([10], dtype=np.int32),
            "embeddings": np.ones((1, 1024), dtype=np.float16),
        }
        first = MODULE.feature_fingerprint(payload, np)
        second = MODULE.feature_fingerprint(payload, np)
        self.assertEqual(first, second)
        payload["embeddings"][0, 0] = 2
        self.assertNotEqual(first, MODULE.feature_fingerprint(payload, np))

    def test_overlap_reports_shared_keys_not_pair_cartesian_product(self):
        left = [{"sample_id": "train_a", "fingerprint": "same"}, {"sample_id": "train_b", "fingerprint": "same"}]
        right = [{"sample_id": "validation_a", "fingerprint": "same"}]
        count, examples = MODULE.overlap_examples(left, right, "fingerprint")
        self.assertEqual(count, 1)
        self.assertEqual(examples[0]["train_sample_id"], "train_a")
        self.assertEqual(examples[0]["validation_sample_id"], "validation_a")

    def test_separation_margin_is_positive_for_disjoint_scores(self):
        summary = MODULE.separation_summary([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], np)
        self.assertAlmostEqual(summary["separation_margin"], 0.6)

    def test_group_breakdown_counts_threshold_errors(self):
        records = [
            {"semantic_class": "REAL", "label": 0, "score": 0.1},
            {"semantic_class": "REAL", "label": 0, "score": 0.8},
            {"semantic_class": "AUDIO_MANIPULATION", "label": 1, "score": 0.9},
        ]
        summary = MODULE.group_summary(records, "score", 0.5, np)
        self.assertEqual(summary["REAL"]["errors_at_threshold"], 1)
        self.assertEqual(summary["AUDIO_MANIPULATION"]["errors_at_threshold"], 0)


if __name__ == "__main__":
    unittest.main()
