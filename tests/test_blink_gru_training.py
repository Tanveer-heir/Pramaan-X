import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "08_train_blink_gru.py"
SPEC = importlib.util.spec_from_file_location("blink_gru_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BlinkGruTrainingTests(unittest.TestCase):
    def test_available_sequence_is_retained(self):
        payload = {
            "schema_version": np.asarray(MODULE.SCHEMA_VERSION),
            "availability": np.asarray("AVAILABLE"),
            "features": np.ones((4, MODULE.FEATURE_DIM), dtype=np.float32),
            "timestamps_sec": np.arange(4, dtype=np.float32),
            "source_frame_indices": np.arange(4, dtype=np.int32),
        }
        sequence, reason = MODULE.blink_sequence(payload, np, 4)
        self.assertIsNone(reason)
        self.assertEqual(sequence.shape, (4, MODULE.FEATURE_DIM))

    def test_not_applicable_sequence_is_not_zero_filled(self):
        payload = {
            "schema_version": np.asarray(MODULE.SCHEMA_VERSION),
            "availability": np.asarray("NOT_APPLICABLE"),
            "not_applicable_reason": np.asarray("INSUFFICIENT_VALID_EYE_FRAMES:0/4"),
            "features": np.empty((0, MODULE.FEATURE_DIM), dtype=np.float32),
            "timestamps_sec": np.empty((0,), dtype=np.float32),
            "source_frame_indices": np.empty((0,), dtype=np.int32),
        }
        sequence, reason = MODULE.blink_sequence(payload, np, 4)
        self.assertIsNone(sequence)
        self.assertTrue(reason.startswith("NOT_APPLICABLE:"), reason)

    def test_small_model_defaults_match_architecture(self):
        args = MODULE.build_parser().parse_args([])
        self.assertEqual(args.hidden_dim, 40)
        self.assertEqual(args.batch_size, 64)


if __name__ == "__main__":
    unittest.main()
