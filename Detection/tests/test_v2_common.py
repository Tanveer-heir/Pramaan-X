import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "v2"))
import v2_common as v


def row(sample_id, language="hindi", identity="unknown", original_id=None):
    return {"sample_id": sample_id, "dataset": "MLAAD", "language": language, "audio_label": "1", "modality": "audio_only", "identity": identity, "original_id": original_id or sample_id, "generator_tool": "g"}


class ProtocolTests(unittest.TestCase):
    def test_deterministic_group_safe_hindi_split(self):
        rows = [row(f"h{i}", original_id=f"source-{i}") for i in range(10)]
        first, second = v.deterministic_assign(rows, 7), v.deterministic_assign(rows, 7)
        self.assertEqual(first, second)
        self.assertEqual({x["role"] for x in first}, set(v.ROLES))
        self.assertTrue(all(x["language"] == "hindi" for x in first))

    def test_non_hindi_is_rejected(self):
        with self.assertRaises(RuntimeError): v.deterministic_assign([row("hi"), row("en", "english"), row("x")], 1)

    def test_fake_only_metrics_are_honest(self):
        import numpy as np
        result = v.binary_metrics(np.asarray([1, 1]), np.asarray([.2, .8]), .5, np)
        self.assertIsNone(result["roc_auc"]); self.assertIsNone(result["balanced_accuracy"])
        self.assertEqual(result["detection_rate"], .5)


if __name__ == "__main__": unittest.main()
