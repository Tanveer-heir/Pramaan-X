import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.explain.render_av_timeline import extract_timeline, render_av_timeline
from scripts.explain.render_counterfactual import render_counterfactual_chart, summarize_counterfactual
from scripts.explain.render_image_overlay import normalize_region_bbox, render_image_overlay
from scripts.report.generate_investigator_report import generate_report


def video_payload():
    return {
        "schema_version": "pramaan_x_prediction_v2",
        "sample_id": "raw_demo",
        "probability_note": "Softmax values are uncalibrated class scores.",
        "class_names": ["REAL", "VISUAL_MANIPULATION", "AUDIO_MANIPULATION", "AUDIO_VISUAL_MANIPULATION"],
        "class_logits": {"REAL": -1.0, "VISUAL_MANIPULATION": 0.2, "AUDIO_MANIPULATION": 0.8, "AUDIO_VISUAL_MANIPULATION": 2.2},
        "softmax_probabilities": {"REAL": 0.02, "VISUAL_MANIPULATION": 0.08, "AUDIO_MANIPULATION": 0.15, "AUDIO_VISUAL_MANIPULATION": 0.75},
        "selected_class": "AUDIO_VISUAL_MANIPULATION",
        "branch_evidence": {
            "visual_logit": 1.1,
            "visual_available": True,
            "audio_logit": 0.9,
            "audio_available": True,
            "av_inconsistency_logit": 1.5,
            "av_available": True,
            "branch_reasons": {"visual": "", "audio": "", "dense_av": ""},
            "coverage": {"dense_av_segments": 2},
        },
        "counterfactual_modality_analysis": {
            "method": "leave_one_available_modality_out",
            "full_evidence": {"selected_class": "AUDIO_VISUAL_MANIPULATION", "softmax_scores": {"AUDIO_VISUAL_MANIPULATION": 0.75}},
            "interventions": {
                "without_visual": {"selected_class": "AUDIO_MANIPULATION", "class_changed": True, "softmax_scores": {"AUDIO_MANIPULATION": 0.61}},
                "without_audio": {"selected_class": "VISUAL_MANIPULATION", "class_changed": True, "softmax_scores": {"VISUAL_MANIPULATION": 0.58}},
                "without_dense_av": {"selected_class": "AUDIO_VISUAL_MANIPULATION", "class_changed": False, "softmax_scores": {"AUDIO_VISUAL_MANIPULATION": 0.71}},
            },
            "summary": {"full_selected_class": "AUDIO_VISUAL_MANIPULATION", "class_changed_without": ["visual", "audio"], "class_stable_without": ["dense_av"]},
        },
        "temporal_evidence": {
            "dense_av": {
                "available": True,
                "evidence_type": "audio_mouth_correspondence",
                "aggregate": {"segment_count": 2},
                "segments": [
                    {"segment_index": 0, "start_sec": 0.25, "end_sec": 2.0, "center_sec": 1.125, "av_inconsistency_logit": 0.2},
                    {"segment_index": 1, "start_sec": 1.25, "end_sec": 3.0, "center_sec": 2.125, "av_inconsistency_logit": 1.4},
                ],
            }
        },
        "checkpoint_provenance": {
            "fusion_checkpoint": "/tmp/pramaan_x_hackathon_final.pth",
            "enabled_branches": ["visual", "audio", "dense_av"],
            "accepted_branch_checkpoints": {"visual": {"path": "/old/visual.pt", "sha256": "abc"}},
        },
        "raw_video": {"path": "/tmp/input.mp4", "sha256": "videohash", "bytes": 1234},
    }


class ExplainabilityReportingTests(unittest.TestCase):
    def test_counterfactual_summary_is_restrained(self):
        text = summarize_counterfactual(video_payload())
        self.assertIn("changed when visual, audio evidence was removed", text)
        self.assertNotIn("caused", text.lower())

    def test_renderers_use_actual_timestamps_and_write_pngs(self):
        payload = video_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counterfactual = render_counterfactual_chart(payload, root / "counterfactual.png")
            timeline = render_av_timeline(payload, root / "timeline.png")
            self.assertTrue(counterfactual and counterfactual.is_file())
            self.assertTrue(timeline and timeline.is_file())
            segments, label = extract_timeline(payload)
            self.assertEqual([row["center_sec"] for row in segments], [1.125, 2.125])
            self.assertEqual(label, "AV inconsistency logit")

    def test_missing_temporal_evidence_does_not_create_empty_chart(self):
        payload = video_payload()
        payload["temporal_evidence"]["dense_av"] = {"available": False, "segments": []}
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(render_av_timeline(payload, Path(directory) / "timeline.png"))

    def test_report_is_self_contained_and_surfaces_limitations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.json"
            output = root / "report.html"
            prediction.write_text(json.dumps(video_payload()), encoding="utf-8")
            result = generate_report(prediction, output)
            document = output.read_text(encoding="utf-8")
            self.assertTrue(result["assets"]["counterfactual"])
            self.assertIn("Counterfactual evidence sensitivity", document)
            self.assertIn("Temporal dense-AV evidence", document)
            self.assertIn("uncalibrated", document.lower())
            self.assertIn("videohash", document)
            self.assertIn("data:image/png;base64", document)

    def test_image_overlay_requires_explicit_normalized_coordinates(self):
        self.assertIsNone(normalize_region_bbox("face"))
        self.assertIsNone(normalize_region_bbox({"x": 0.8, "y": 0.1, "width": 0.4, "height": 0.2}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "image.png"
            Image.new("RGB", (100, 80), (100, 100, 100)).save(source)
            output = render_image_overlay(source, [{"category": "lighting", "region_bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.3}}], root / "overlay.png")
            self.assertTrue(output and output.is_file())


if __name__ == "__main__":
    unittest.main()

