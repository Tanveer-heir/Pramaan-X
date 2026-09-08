import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inference" / "predict_pramaan_x.py"
SPEC = importlib.util.spec_from_file_location("predict_pramaan_x", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PredictorArgumentsTests(unittest.TestCase):
    def test_cache_prediction_requires_explicit_fusion_checkpoint(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                MODULE.parser().parse_args([])

    def test_sample_id_mode_is_supported(self):
        args = MODULE.parser().parse_args(["--fusion-checkpoint", "model.pth", "--sample-id", "fac_example", "--output", "result.json"])
        self.assertEqual(args.sample_id, "fac_example")
        self.assertEqual(str(args.output), "result.json")

    def test_repository_checkpoint_replaces_saved_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / MODULE.STANDARD_BRANCH_CHECKPOINTS["visual"]
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"checkpoint")
            saved = {"visual": {"path": "/old/laptop/checkpoint.pt"}}
            self.assertEqual(MODULE.resolve_branch_checkpoint("visual", None, saved, root), expected)

    def test_explicit_checkpoint_override_has_priority(self):
        override = Path("/explicit/visual.pt")
        saved = {"visual": {"path": "/old/laptop/checkpoint.pt"}}
        self.assertEqual(MODULE.resolve_branch_checkpoint("visual", override, saved), override)


if __name__ == "__main__":
    unittest.main()
