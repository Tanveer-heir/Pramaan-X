import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "preprocessing" / "06_extract_convnext_features.py"
SPEC = importlib.util.spec_from_file_location("convnext_extraction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConvNeXtFeatureExtractionTests(unittest.TestCase):
    def test_defaults_to_streaming_video_mode(self):
        self.assertEqual(MODULE.build_parser().parse_args([]).input_mode, "video")

    def test_sample_id_parser_deduplicates_and_strips(self):
        self.assertEqual(MODULE.parse_sample_ids(["fac_a, fac_b", "fac_a"]), {"fac_a", "fac_b"})

    def test_feature_contract_uses_sparse_temporal_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(MODULE.SCHEMA_VERSION),
                backbone=np.asarray(MODULE.BACKBONE_NAME),
                sample_id=np.asarray("fac_test"),
                timestamps_sec=np.asarray([0.0, 0.25], dtype=np.float32),
                source_frame_indices=np.asarray([0, 6], dtype=np.int32),
                face_embeddings=np.empty((0, MODULE.EMBEDDING_DIM), dtype=np.float16),
                face_frame_indices=np.empty((0,), dtype=np.int32),
                face_bboxes_xyxy=np.empty((0, 4), dtype=np.int32),
                mouth_embeddings=np.empty((0, MODULE.EMBEDDING_DIM), dtype=np.float16),
                mouth_frame_indices=np.empty((0,), dtype=np.int32),
                mouth_bboxes_xyxy=np.empty((0, 4), dtype=np.int32),
            )
            self.assertEqual(MODULE.validate_feature_file(path, np), (True, None))

    def test_validator_recognises_an_empty_interrupted_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.npz"
            path.write_bytes(b"")
            valid, reason = MODULE.validate_feature_file(path, np)
            self.assertFalse(valid)
            self.assertIn("No data left in file", reason)

    def test_atomic_writer_replaces_a_corrupt_final_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.npz"
            path.write_bytes(b"")
            payload = {
                "schema_version": np.asarray(MODULE.SCHEMA_VERSION),
                "backbone": np.asarray(MODULE.BACKBONE_NAME),
                "sample_id": np.asarray("fac_test"),
                "timestamps_sec": np.asarray([0.0], dtype=np.float32),
                "source_frame_indices": np.asarray([0], dtype=np.int32),
                "face_embeddings": np.empty((0, MODULE.EMBEDDING_DIM), dtype=np.float16),
                "face_frame_indices": np.empty((0,), dtype=np.int32),
                "face_bboxes_xyxy": np.empty((0, 4), dtype=np.int32),
                "mouth_embeddings": np.empty((0, MODULE.EMBEDDING_DIM), dtype=np.float16),
                "mouth_frame_indices": np.empty((0,), dtype=np.int32),
                "mouth_bboxes_xyxy": np.empty((0, 4), dtype=np.int32),
            }
            MODULE.write_feature_file(path, payload, np)
            self.assertEqual(MODULE.validate_feature_file(path, np), (True, None))


if __name__ == "__main__":
    unittest.main()
