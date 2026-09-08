import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "training" / "07_train_visual_temporal.py"
SPEC = importlib.util.spec_from_file_location("visual_temporal_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VisualTemporalTrainingTests(unittest.TestCase):
    def test_shared_frames_produce_a_1536_dim_sequence(self):
        payload = {
            "schema_version": np.asarray(MODULE.SCHEMA_VERSION),
            "face_embeddings": np.ones((2, MODULE.FACE_DIM), dtype=np.float16),
            "face_frame_indices": np.asarray([1, 4], dtype=np.int32),
            "mouth_embeddings": np.ones((2, MODULE.MOUTH_DIM), dtype=np.float16),
            "mouth_frame_indices": np.asarray([1, 3], dtype=np.int32),
        }
        sequence, reason = MODULE.shared_frame_sequence(payload, np)
        self.assertIsNone(reason)
        self.assertEqual(sequence.shape, (1, MODULE.INPUT_DIM))

    def test_missing_modality_is_not_zero_filled(self):
        payload = {
            "schema_version": np.asarray(MODULE.SCHEMA_VERSION),
            "face_embeddings": np.ones((1, MODULE.FACE_DIM), dtype=np.float16),
            "face_frame_indices": np.asarray([1], dtype=np.int32),
            "mouth_embeddings": np.empty((0, MODULE.MOUTH_DIM), dtype=np.float16),
            "mouth_frame_indices": np.empty((0,), dtype=np.int32),
        }
        sequence, reason = MODULE.shared_frame_sequence(payload, np)
        self.assertIsNone(sequence)
        self.assertEqual(reason, "NO_SHARED_FACE_MOUTH_FRAMES")

    def test_training_defaults_are_lightweight(self):
        args = MODULE.build_parser().parse_args([])
        self.assertEqual(args.hidden_dim, 192)
        self.assertEqual(args.batch_size, 16)


if __name__ == "__main__":
    unittest.main()
