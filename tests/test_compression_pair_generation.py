import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing" / "10_generate_compression_pairs.py"
SPEC = importlib.util.spec_from_file_location("compression_pair_generator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CompressionPairGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = MODULE.load_extractor()

    def payload(self, sample_id, generation):
        extractor = self.extractor
        return {
            "schema_version": np.asarray(extractor.SCHEMA_VERSION),
            "sample_id": np.asarray(f"{sample_id}_generation{generation}"),
            "availability": np.asarray("AVAILABLE"),
            "features": np.ones(extractor.FEATURE_DIM, dtype=np.float32),
            "feature_names": np.asarray(extractor.FEATURE_NAMES),
            "codec_name": np.asarray("h264"),
            "pixel_format": np.asarray("yuv420p"),
            "duration_seconds": np.asarray(1.0, dtype=np.float32),
            "source_width": np.asarray(128, dtype=np.int32),
            "source_height": np.asarray(96, dtype=np.int32),
            "source_fps": np.asarray(10.0, dtype=np.float32),
            "sampled_frame_count": np.asarray(4, dtype=np.int32),
            "probed_frame_count": np.asarray(10, dtype=np.int32),
            "pair_schema_version": np.asarray(MODULE.PAIR_SCHEMA_VERSION),
            "source_sample_id": np.asarray(sample_id),
            "controlled_generation": np.asarray(generation, dtype=np.int8),
            "compression_label": np.asarray(generation - 1, dtype=np.int8),
        }

    def test_generation_labels_are_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            for generation in (1, 2):
                path = Path(directory) / f"pair{generation}.npz"
                np.savez_compressed(path, **self.payload("fac_test", generation))
                self.assertEqual(MODULE.validate_pair_file(path, generation, self.extractor, np), (True, None))

    def test_validator_rejects_wrong_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.npz"
            np.savez_compressed(path, **self.payload("fac_test", 1))
            valid, reason = MODULE.validate_pair_file(path, 2, self.extractor, np)
            self.assertFalse(valid)
            self.assertEqual(reason, "generation mismatch")

    def test_pair_paths_do_not_replace_source_video(self):
        root = Path("features")
        self.assertEqual(MODULE.pair_path(root, "fac_test", 1), root / "fac_test_generation1.npz")
        self.assertEqual(MODULE.pair_path(root, "fac_test", 2), root / "fac_test_generation2.npz")


if __name__ == "__main__":
    unittest.main()
