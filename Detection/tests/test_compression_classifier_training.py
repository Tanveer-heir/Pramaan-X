import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "11_train_compression_classifier.py"
SPEC = importlib.util.spec_from_file_location("compression_classifier_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CompressionClassifierTrainingTests(unittest.TestCase):
    def payload(self, label):
        return {
            "schema_version": np.asarray(MODULE.FEATURE_SCHEMA_VERSION),
            "pair_schema_version": np.asarray(MODULE.PAIR_SCHEMA_VERSION),
            "source_sample_id": np.asarray("fac_test"),
            "compression_label": np.asarray(label, dtype=np.int8),
            "controlled_generation": np.asarray(label + 1, dtype=np.int8),
            "availability": np.asarray("AVAILABLE"),
            "features": np.ones(MODULE.FEATURE_DIM, dtype=np.float32),
        }

    def test_valid_pair_feature_is_loaded(self):
        features, reason = MODULE.pair_feature(self.payload(1), np, "fac_test", 1)
        self.assertIsNone(reason)
        self.assertEqual(features.shape, (MODULE.FEATURE_DIM,))

    def test_label_mismatch_is_rejected(self):
        features, reason = MODULE.pair_feature(self.payload(0), np, "fac_test", 1)
        self.assertIsNone(features)
        self.assertEqual(reason, "compression_label_mismatch")

    def test_classifier_defaults_are_lightweight(self):
        args = MODULE.build_parser().parse_args([])
        self.assertEqual(args.hidden_dim, 32)
        self.assertEqual(args.batch_size, 64)


if __name__ == "__main__":
    unittest.main()
