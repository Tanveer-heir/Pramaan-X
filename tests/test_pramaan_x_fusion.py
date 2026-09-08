import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fusion" / "pramaan_x_fusion.py"
SPEC = importlib.util.spec_from_file_location("pramaan_x_fusion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PramaanXFusionTests(unittest.TestCase):
    def test_fixed_class_contract(self):
        self.assertEqual(MODULE.CLASS_NAMES, ("REAL", "VISUAL_MANIPULATION", "AUDIO_MANIPULATION", "AUDIO_VISUAL_MANIPULATION"))
        self.assertEqual(MODULE.CLASS_TO_INDEX["AUDIO_MANIPULATION"], 2)

    def test_missing_logit_is_zero_only_with_mask(self):
        order = MODULE.feature_order(["visual", "audio"])
        vector = MODULE.vectorize({"visual_logit": None, "visual_available": False, "audio_logit": 2.0, "audio_available": True}, order)
        self.assertEqual(vector, [0.0, 0.0, 2.0, 1.0])

    def test_masks_are_not_normalized(self):
        order = MODULE.feature_order(["visual", "audio"])
        matrix = np.asarray([[1.0, 1.0, 2.0, 1.0], [3.0, 0.0, 4.0, 1.0]], dtype=np.float32)
        normalized, _, _ = MODULE.normalize_features(matrix, order)
        self.assertTrue(np.array_equal(normalized[:, [1, 3]], matrix[:, [1, 3]]))

    def test_av_requires_predeclared_clear_gain(self):
        base = {"macro_f1": .80, "per_class": {"REAL": {"f1": .4}, "AUDIO_MANIPULATION": {"f1": .0}}}
        av = {"macro_f1": .81, "per_class": {"REAL": {"f1": .57}, "AUDIO_MANIPULATION": {"f1": .31}}}
        self.assertEqual(MODULE.selection_reason(base, av)[0], "visual_audio")
        av["macro_f1"] = .82
        self.assertEqual(MODULE.selection_reason(base, av)[0], "visual_audio_av")

    def test_av_is_rejected_when_protected_f1_declines(self):
        base = {"macro_f1": .80, "per_class": {"REAL": {"f1": .4}, "AUDIO_MANIPULATION": {"f1": .2}}}
        av = {"macro_f1": .90, "per_class": {"REAL": {"f1": .39}, "AUDIO_MANIPULATION": {"f1": .4}}}
        self.assertEqual(MODULE.selection_reason(base, av)[0], "visual_audio")


    def test_counterfactuals_only_remove_available_evidence(self):
        evidence = {
            "visual_logit": 2.0, "visual_available": True,
            "audio_logit": None, "audio_available": False,
            "av_inconsistency_logit": -1.0, "av_available": True,
        }
        def score(row):
            selected = "VISUAL_MANIPULATION" if row["visual_available"] else "AUDIO_MANIPULATION"
            return {
                "selected_class": selected,
                "class_logits": {name: 0.0 for name in MODULE.CLASS_NAMES},
                "softmax_scores": {name: 0.25 for name in MODULE.CLASS_NAMES},
            }
        result = MODULE.counterfactual_modality_analysis(evidence, ["visual", "audio", "dense_av"], score)
        self.assertEqual(set(result["interventions"]), {"without_visual", "without_dense_av"})
        self.assertEqual(result["summary"]["class_changed_without"], ["visual"])
        self.assertEqual(result["summary"]["class_stable_without"], ["dense_av"])

    def test_av_aggregate_wrapper_uses_timeline_records(self):
        runner = object.__new__(MODULE.BranchRunner)
        runner.av_inconsistency_timeline = lambda dense, visual: (0.75, None, [{"segment_index": 0}, {"segment_index": 1}])
        self.assertEqual(runner.av_inconsistency_logit(Path("dense.npz"), Path("visual.npz")), (0.75, None, 2))


if __name__ == "__main__":
    unittest.main()
