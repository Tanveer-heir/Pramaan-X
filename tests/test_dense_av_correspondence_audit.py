import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "21_audit_dense_av_correspondence.py"
SPEC = importlib.util.spec_from_file_location("dense_av_correspondence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

class DenseAVCorrespondenceAuditTests(unittest.TestCase):
    def cache(self, timestamps):
        n=len(timestamps)
        source=MODULE.Source("fac_test","id_a","2","train",Path("dense"),Path("visual"))
        return {"source":source,"audio":np.ones((n,82),np.float32),"mouth":np.arange(n*768,dtype=np.float32).reshape(n,768),
                "timestamps":np.asarray(timestamps,dtype=float),"indices":np.arange(n),"sample_rate":16000,"context":4000,"samples":160000}

    def test_positive_offset_is_later_mouth_content(self):
        cache=self.cache(np.arange(24)*.25)
        result=MODULE.build_cohort(cache,"matched_boundary",8,4,np)
        shifted=[x for x in result if x["label"]==0 and x["offset_frames"]==2][0]
        self.assertTrue(all(value > 0 for value in shifted["actual_offsets_sec"]))

    def test_duplicate_decreasing_and_nonfinite_timestamps_rejected(self):
        for timestamps in ([0,.25,.25,.5], [0,.25,.2,.5], [0,np.nan,.5]):
            with self.assertRaises(RuntimeError):
                MODULE.timestamp_runs(np.asarray(timestamps),np)

    def test_all_offsets_are_from_identical_anchor_population(self):
        cache=self.cache(np.arange(32)*.25)
        examples=MODULE.build_cohort(cache,"matched_boundary",8,4,np)
        negatives=[x for x in examples if x["label"]==0]
        for offset in (-4,-2,2,4):
            self.assertEqual({x["paired_anchor_id"] for x in negatives if x["offset_frames"]==offset},
                             {x["paired_anchor_id"] for x in negatives if x["offset_frames"]==-4})

    def test_context_rounding_and_padding(self):
        cache=self.cache(np.arange(24)*.25)
        cache["samples"]=4000
        meta=MODULE.padding(.0,cache)
        self.assertEqual(meta["centre_sample"],0)
        self.assertEqual(meta["left_padding"],2000)
        self.assertEqual(meta["valid_samples"],2000)

    def test_crossover_has_exact_marginal_balance(self):
        cache=self.cache(np.arange(36)*.25)
        records=MODULE.crossover(cache,8,np)
        self.assertGreater(len(records),0)
        for block in {r["block_id"] for r in records}:
            rows=[r for r in records if r["block_id"]==block]
            self.assertEqual(sum(r["label"]==1 for r in rows),2)
            self.assertEqual(sum(r["label"]==0 for r in rows),2)
            self.assertEqual({r["role"] for r in rows},{"aa","bb","ab","ba"})
            self.assertTrue(all(r["matched_anchor_ids"] for r in rows))

    def test_missing_cached_mouth_frame_splits_run(self):
        cache=self.cache(np.arange(12)*.25)
        cache["indices"]=np.asarray([0,1,2,3,4,8,9,10,11,12,13,14])
        self.assertEqual(MODULE.source_runs(cache,np),[(0,5),(5,12)])

    def test_additive_logits_have_zero_crossover_interaction(self):
        rows=[
            {"cohort":"crossover","block_id":"one","role":"aa","logit":3.0,"score":.9,"identity":"i","source":"s"},
            {"cohort":"crossover","block_id":"one","role":"bb","logit":7.0,"score":.9,"identity":"i","source":"s"},
            {"cohort":"crossover","block_id":"one","role":"ab","logit":5.0,"score":.5,"identity":"i","source":"s"},
            {"cohort":"crossover","block_id":"one","role":"ba","logit":5.0,"score":.5,"identity":"i","source":"s"},
        ]
        self.assertEqual(MODULE.crossover_interactions(rows,np)["mean_logit_interaction"],0.0)

    def test_correspondence_sensitive_logits_have_positive_interaction(self):
        rows=[
            {"cohort":"crossover","block_id":"one","role":role,"logit":value,"score":value,
             "identity":"i","source":"s"}
            for role,value in (("aa",1.0),("bb",1.0),("ab",0.0),("ba",0.0))]
        self.assertEqual(MODULE.crossover_interactions(rows,np)["mean_logit_interaction"],1.0)

    def test_matched_metrics_are_balanced_per_offset(self):
        cache=self.cache(np.arange(32)*.25)
        rows=MODULE.build_cohort(cache,"matched_boundary",8,4,np)
        for row in rows:
            row["score"]=.9 if row["label"] else .1
            row["logit"]=2.0 if row["label"] else -2.0
        metrics=MODULE.matched_balanced_metrics(rows,.5,np)
        for result in metrics.values():
            if result["anchors"]:
                self.assertEqual(result["roc_auc"],1.0)
                self.assertEqual(result["average_precision"],1.0)

    def test_compact_records_reconstruct_paired_effects(self):
        cache=self.cache(np.arange(32)*.25)
        rows=MODULE.build_cohort(cache,"matched_boundary",8,4,np)
        for row in rows:
            row["score"]=.8 if row["label"] else .2
            row["logit"]=1.0 if row["label"] else -1.0
        compact=[MODULE.compact(row) for row in rows]
        result=MODULE.paired_summary(compact,np)
        self.assertEqual(result["matched_boundary:2"]["mean_aligned_minus_shifted_logit"],2.0)

    def test_complete_summary_reconstructs_from_compact_records(self):
        cache=self.cache(np.arange(40)*.25)
        rows=[]
        rows.extend(MODULE.build_cohort(cache,"matched_boundary",8,4,np))
        rows.extend(MODULE.build_cohort(cache,"matched_strict",8,4,np))
        rows.extend(MODULE.crossover(cache,8,np,4))
        for row in rows:
            row["score"]=.8 if row["label"] else .2
            row["logit"]=1.0 if row["label"] else -1.0
        compact=[MODULE.compact(row) for row in rows]
        summary=MODULE.summarize(compact,.5,np,20,42,None,"smoke")
        self.assertEqual(summary["scientific_interpretation"]["status"],"INCONCLUSIVE_SMOKE_ONLY")
        self.assertIn("matched_boundary:2",summary["matched"]["balanced_metrics_by_offset"])
        self.assertGreater(summary["crossover"]["interaction"]["blocks"],0)

    def test_fixed_threshold_metrics_ties_get_half_credit(self):
        metrics=MODULE.fixed_metrics(np.asarray([1,0]),np.asarray([.5,.5]),.5,np)
        self.assertEqual(metrics["ordering_fraction"],.5)
        self.assertEqual(metrics["balanced_accuracy_fixed"],.5)

    def test_empty_metrics_have_explicit_reason(self):
        self.assertEqual(MODULE.fixed_metrics(np.asarray([],dtype=int),np.asarray([]),.5,np)["reason"],"empty_or_one_class")

    def test_deterministic_historical_choice(self):
        first=MODULE.hash_choice([-4,-2,2,4],"fac:a",42)
        self.assertEqual(first,MODULE.hash_choice([-4,-2,2,4],"fac:a",42))

if __name__ == "__main__":
    unittest.main()
