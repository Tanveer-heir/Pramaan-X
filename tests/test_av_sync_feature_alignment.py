import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing" / "16_build_av_sync_features.py"
SPEC = importlib.util.spec_from_file_location("av_sync_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Payload(dict):
    @property
    def files(self):
        return list(self)


def source_payloads():
    audio = Payload({
        "schema_version": np.asarray(MODULE.AUDIO_SCHEMA_VERSION),
        "model_sha256": np.asarray(MODULE.AUDIO_MODEL_SHA256),
        "sample_id": np.asarray("fac_test"),
        "audio_modality": np.asarray("AVAILABLE"),
        "timestamps_start_sec": np.asarray([0.0, 4.0], dtype=np.float32),
        "timestamps_end_sec": np.asarray([4.0, 8.0], dtype=np.float32),
        "embeddings": np.ones((2, MODULE.AUDIO_DIM), dtype=np.float16),
    })
    visual = Payload({
        "schema_version": np.asarray(MODULE.VISUAL_SCHEMA_VERSION),
        "backbone": np.asarray(MODULE.VISUAL_BACKBONE),
        "sample_id": np.asarray("fac_test"),
        "timestamps_sec": np.asarray([0.0, 1.0, 4.0, 5.0], dtype=np.float32),
        "mouth_frame_indices": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "mouth_embeddings": np.asarray([
            np.zeros(MODULE.MOUTH_DIM), np.ones(MODULE.MOUTH_DIM),
            np.ones(MODULE.MOUTH_DIM) * 2, np.ones(MODULE.MOUTH_DIM) * 4,
        ], dtype=np.float16),
    })
    return audio, visual


class AVSyncAlignmentTests(unittest.TestCase):
    def test_timestamp_alignment_and_motion_summary(self):
        audio, visual = source_payloads()
        result = MODULE.align_payloads(audio, visual, 2, np)
        self.assertEqual(result["av_modality"], "AVAILABLE")
        self.assertEqual(result["audio_embeddings"].shape, (2, MODULE.AUDIO_DIM))
        np.testing.assert_allclose(result["mouth_mean_embeddings"][:, 0], [0.5, 3.0])
        np.testing.assert_allclose(result["mouth_motion_embeddings"][:, 0], [1.0, 2.0])
        np.testing.assert_array_equal(result["mouth_frame_counts"], [2, 2])

    def test_missing_audio_is_not_applicable_not_zero_evidence(self):
        audio, visual = source_payloads()
        audio["audio_modality"] = np.asarray("NOT_APPLICABLE")
        audio["embeddings"] = np.empty((0, MODULE.AUDIO_DIM), dtype=np.float16)
        audio["timestamps_start_sec"] = np.empty((0,), dtype=np.float32)
        audio["timestamps_end_sec"] = np.empty((0,), dtype=np.float32)
        result = MODULE.align_payloads(audio, visual, 2, np)
        self.assertEqual(result["av_modality"], "NOT_APPLICABLE")
        self.assertEqual(result["not_applicable_reason"], "AUDIO_NOT_APPLICABLE")
        self.assertEqual(result["audio_embeddings"].shape, (0, MODULE.AUDIO_DIM))

    def test_insufficient_mouth_frames_is_not_applicable(self):
        audio, visual = source_payloads()
        visual["mouth_embeddings"] = visual["mouth_embeddings"][:1]
        visual["mouth_frame_indices"] = visual["mouth_frame_indices"][:1]
        result = MODULE.align_payloads(audio, visual, 2, np)
        self.assertEqual(result["av_modality"], "NOT_APPLICABLE")
        self.assertEqual(result["not_applicable_reason"], "NO_WINDOWS_WITH_SUFFICIENT_MOUTH_FRAMES")

    def test_atomic_output_validator_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.npz"
            corrupt.write_bytes(b"")
            self.assertFalse(MODULE.validate_output(corrupt, np)[0])


if __name__ == "__main__":
    unittest.main()
