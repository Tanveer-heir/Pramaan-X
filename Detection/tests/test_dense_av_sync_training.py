import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "19_train_dense_av_sync.py"
SPEC = importlib.util.spec_from_file_location("dense_av_sync_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_source(directory: Path, count: int = 20):
    dense_path, visual_path = directory / "dense.npz", directory / "visual.npz"
    indices = np.arange(count, dtype=np.int32)
    np.savez_compressed(
        dense_path,
        schema_version=np.asarray(MODULE.DENSE_SCHEMA_VERSION), sample_id=np.asarray("fac_test"),
        av_modality=np.asarray("AVAILABLE"), audio_descriptor=np.asarray(MODULE.AUDIO_DESCRIPTOR),
        visual_schema_version=np.asarray(MODULE.VISUAL_SCHEMA_VERSION), visual_backbone=np.asarray(MODULE.VISUAL_BACKBONE),
        timestamps_sec=np.arange(count, dtype=np.float32) / 4.0, mouth_frame_indices=indices,
        audio_features=np.arange(count * MODULE.AUDIO_DIM, dtype=np.float32).reshape(count, MODULE.AUDIO_DIM),
    )
    np.savez_compressed(
        visual_path,
        schema_version=np.asarray(MODULE.VISUAL_SCHEMA_VERSION), sample_id=np.asarray("fac_test"),
        backbone=np.asarray(MODULE.VISUAL_BACKBONE), mouth_frame_indices=indices,
        mouth_embeddings=np.arange(count * MODULE.MOUTH_DIM, dtype=np.float32).reshape(count, MODULE.MOUTH_DIM),
    )
    return MODULE.Source("fac_test", dense_path, visual_path, count)


class DenseAVSyncTrainingTests(unittest.TestCase):
    def test_contiguous_runs_split_tracking_gaps(self):
        runs = MODULE.contiguous_runs(np.asarray([0.0, 0.25, 0.5, 1.5, 1.75]), np)
        self.assertEqual(runs, [(0, 3), (3, 5)])

    def test_pair_generation_is_balanced_and_uses_requested_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory))
            pairs = MODULE.build_pairs(source, np, segment_frames=8, stride_frames=4, seed=42)
        self.assertGreater(len(pairs), 0)
        self.assertEqual(sum(pair.label == 1 for pair in pairs), sum(pair.label == 0 for pair in pairs))
        self.assertTrue(all(abs(pair.offset_frames) in MODULE.DEFAULT_SHIFT_FRAMES for pair in pairs if pair.label == 0))
        self.assertTrue(all(pair.offset_frames == 0 for pair in pairs if pair.label == 1))

    def test_pair_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory))
            first = MODULE.build_pairs(source, np, 8, 4, 42)
            second = MODULE.build_pairs(source, np, 8, 4, 42)
        self.assertEqual([pair.offset_frames for pair in first], [pair.offset_frames for pair in second])

    def test_dense_to_visual_index_mapping_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory), 12)
            with np.load(source.dense_path, allow_pickle=False) as dense, np.load(source.visual_path, allow_pickle=False) as visual:
                arrays, reason = MODULE.load_source_arrays(dense, visual, np, "fac_test")
        self.assertIsNone(reason)
        self.assertEqual(arrays[0].shape, (12, MODULE.AUDIO_DIM))
        self.assertEqual(arrays[1].shape, (12, MODULE.MOUTH_DIM))

    def test_binary_metrics_perfect_ranking(self):
        metrics = MODULE.binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]), np)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
