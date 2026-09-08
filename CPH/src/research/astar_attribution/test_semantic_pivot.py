"""
Unit tests for Palantir-Style Semantic Pivot Engine & Bellingcat Crop (§2.4b).
"""

import os
import io
import sys
import json
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.research.astar_attribution.semantic_pivot import (
    SemanticPivotEngine,
    SceneDeconstruction,
)
from src.research.astar_attribution.astar_engine import AStarVisualAttributionEngine
from src.research.astar_attribution.subimage_matcher import MatchResult
from src.research.astar_attribution.neighbor_crawler import VisualNeighborCrawler


class TestSemanticPivot(unittest.IsolatedAsyncioTestCase):
    """Test suite for SemanticPivotEngine and Bellingcat salient crop search."""

    def setUp(self):
        self.cache_dir = "data/cache/test_pivot"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.engine = SemanticPivotEngine(cache_dir=self.cache_dir)

        # Create a test target image (400x300)
        self.test_img_path = os.path.join(self.cache_dir, "test_target_farmers_protest_2021.jpg")
        img = Image.new("RGB", (400, 300), color=(120, 80, 40))
        img.save(self.test_img_path)

    def test_scene_deconstruction_dataclass(self):
        """Test SceneDeconstruction serialization and fields."""
        decon = SceneDeconstruction(
            ocr_text=["A-74", "DELHI POLICE"],
            entities=["John Deere tractor", "Indian Flag"],
            event_hypothesis="Farmers rally at Red Fort 2021",
            query_battery=["A-74 tractor protest", "Farmers rally Delhi 2021"],
        )
        d = decon.to_dict()
        self.assertEqual(len(d["ocr_text"]), 2)
        self.assertEqual(len(d["entities"]), 2)
        self.assertEqual(d["event_hypothesis"], "Farmers rally at Red Fort 2021")
        self.assertEqual(len(d["query_battery"]), 2)

    def test_heuristic_deconstruction_fallback(self):
        """Test local token & filename heuristic fallback when LLM is unavailable."""
        decon = self.engine._heuristic_deconstruction(self.test_img_path)
        self.assertIsInstance(decon, SceneDeconstruction)
        self.assertIn("2021", decon.event_hypothesis)
        self.assertGreaterEqual(len(decon.query_battery), 3)
        self.assertTrue(any("farmers" in q.lower() for q in decon.query_battery))

    def test_deconstruct_scene_gemini_mock(self):
        """Test multimodal Gemini scene deconstruction with structured JSON response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "ocr_text": ["A-74", "KA-01-E-1234"],
            "entities": ["Tractor", "Indian Flag"],
            "event_hypothesis": "Farmers tractor rally protest in New Delhi 2021",
            "query_battery": [
                "A-74 tractor Delhi protest 2021",
                "Farmers tractor rally Red Fort AP Reuters 2021",
                "India farm law protest tractors Delhi",
            ]
        })
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {"FORENSIC_BENCHMARK_MODE": "0"}):
            with patch.object(self.engine, "client", mock_client):
                decon = self.engine.deconstruct_scene(self.test_img_path)
                self.assertEqual(len(decon.ocr_text), 2)
                self.assertIn("A-74", decon.ocr_text)
                self.assertIn("Farmers", decon.event_hypothesis)
                self.assertEqual(len(decon.query_battery), 3)

    def test_deconstruct_scene_ladder_failover(self):
        """Test multi-model ladder failover when initial model in ladder raises quota error."""
        mock_client = MagicMock()
        # First call fails with 429 quota, second call succeeds
        mock_response_ok = MagicMock()
        mock_response_ok.text = json.dumps({
            "ocr_text": ["POLICE"],
            "entities": ["Protesters"],
            "event_hypothesis": "Delhi tractor parade 2021",
            "query_battery": ["Delhi tractor parade 2021", "Farmers protest AP photo"]
        })
        mock_client.models.generate_content.side_effect = [
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            mock_response_ok,
        ]

        with patch.dict(os.environ, {"FORENSIC_BENCHMARK_MODE": "0"}):
            with patch.object(self.engine, "client", mock_client):
                decon = self.engine.deconstruct_scene(self.test_img_path)
                self.assertEqual(decon.event_hypothesis, "Delhi tractor parade 2021")
                self.assertEqual(mock_client.models.generate_content.call_count, 2)

    async def test_harvest_candidate_images_filtering(self):
        """Test candidate harvesting with small-image filtering and deduplication."""
        # Create small image (64x64) and valid image (300x200) in bytes
        buf_small = io.BytesIO()
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(buf_small, format="JPEG")
        small_bytes = buf_small.getvalue()

        buf_valid = io.BytesIO()
        Image.new("RGB", (300, 200), color=(0, 255, 0)).save(buf_valid, format="JPEG")
        valid_bytes = buf_valid.getvalue()

        async def mock_get(url, *args, **kwargs):
            resp = MagicMock()
            if "small" in url:
                resp.status_code = 200
                resp.content = small_bytes
                resp.headers = {"last-modified": "Wed, 20 Jan 2021 10:00:00 GMT"}
            elif "valid" in url:
                resp.status_code = 200
                resp.content = valid_bytes
                resp.headers = {"last-modified": "Fri, 22 Jan 2021 12:00:00 GMT"}
            else:
                resp.status_code = 404
                resp.content = b""
                resp.headers = {}
            return resp

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.images.return_value = [
            {"image": "https://example.com/small.jpg"},
            {"image": "https://wire.com/valid.jpg"},
            {"image": "https://wire.com/valid.jpg"},  # Duplicate
        ]
        mock_ddgs_instance.news.return_value = []

        with patch("src.research.astar_attribution.semantic_pivot.DDGS") as mock_ddgs_cls:
            mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance
            with patch("httpx.AsyncClient.get", side_effect=mock_get):
                harvested = await self.engine.harvest_candidate_images(
                    query_battery=["Farmers protest Delhi"],
                    max_images=10,
                )

                # Small image filtered out (<150px), duplicate deduped -> exactly 1 valid candidate
                self.assertEqual(len(harvested), 1)
                self.assertEqual(harvested[0]["url"], "https://wire.com/valid.jpg")
                self.assertEqual(harvested[0]["width"], 300)
                self.assertEqual(harvested[0]["height"], 200)

    async def test_run_pivot_sifting(self):
        """Test run_pivot pipeline sifting candidates via homography & PDQ thresholds."""
        mock_decon = SceneDeconstruction(
            ocr_text=["A-74"],
            entities=["Tractor"],
            event_hypothesis="Tractor rally 2021",
            query_battery=["Farmers tractor rally 2021"],
        )

        cand_img_path = os.path.join(self.cache_dir, "cand_master.jpg")
        img_cand = Image.new("RGB", (600, 450), color=(100, 150, 200))
        img_cand.save(cand_img_path)

        mock_harvested = [
            {
                "url": "https://reuters.com/master_photo.jpg",
                "local_path": cand_img_path,
                "domain": "reuters.com",
                "timestamp": "2021-01-26T00:00:00Z",
                "width": 600,
                "height": 450,
            }
        ]

        mock_matcher = MagicMock()
        # Returns partial crop match
        mock_matcher.compare_images.return_value = MatchResult(
            is_match=True,
            is_partial_crop=True,
            crop_type="target_is_crop_of_candidate",
            pdq_distance=85,
            inlier_ratio=0.88,
            inlier_count=1200,
            bounding_box=[50, 50, 400, 300],
            scale_factor=1.5,
            similarity_score=0.95,
            details="Target is crop of Reuters master",
        )

        crawler = VisualNeighborCrawler(cache_dir=self.cache_dir)

        with patch.object(self.engine, "deconstruct_scene", return_value=mock_decon):
            with patch.object(self.engine, "harvest_candidate_images", return_value=mock_harvested):
                candidates = await self.engine.run_pivot(
                    target_image_path=self.test_img_path,
                    matcher=mock_matcher,
                    neighbor_crawler=crawler,
                )

                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0].source_hop, "semantic_pivot")
                self.assertEqual(candidates[0].domain, "reuters.com")
                self.assertTrue(candidates[0].is_authoritative_wire)

    def test_bellingcat_salient_subregion_crop(self):
        """Test Bellingcat 60% salient sub-region crop creation."""
        astar_eng = AStarVisualAttributionEngine()
        crop_path = astar_eng.crop_salient_subregion(self.test_img_path, crop_ratio=0.60)
        self.assertIsNotNone(crop_path)
        self.assertTrue(os.path.isfile(crop_path))

        crop_img = Image.open(crop_path)
        # Original: 400x300, 60%: 240x180
        self.assertEqual(crop_img.size, (240, 180))


if __name__ == "__main__":
    unittest.main()