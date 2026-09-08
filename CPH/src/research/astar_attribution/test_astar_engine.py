"""
Unit tests for A* Visual Attribution Engine (§2.4b).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.research.astar_attribution.astar_engine import (
    AStarVisualAttributionEngine,
    ClosedSet,
    PartialMatchLedger,
    AStarNode,
)
from src.research.astar_attribution.neighbor_crawler import NeighborCandidate
from src.research.astar_attribution.subimage_matcher import MatchResult


class TestAStarEngine(unittest.TestCase):
    """Test suite for A* Attribution data structures and scoring heuristics."""

    def setUp(self):
        self.closed_set = ClosedSet()
        self.ledger = PartialMatchLedger()
        self.engine = AStarVisualAttributionEngine()

        self.target_cand = NeighborCandidate(
            url="https://example.com/images/crop.jpg",
            local_path="data/cache/test_target.jpg",
            width=640,
            height=360,
            pdq_hex="0000111122223333444455556666777788889999aaaabbbbccccddddeeeeffff",
            timestamp="2022-01-01T00:00:00Z",
            domain="example.com",
            is_authoritative_wire=False,
            source_hop="target",
        )

    def test_closed_set(self):
        """Test closed set URL normalization and domain-scoped deduplication."""
        self.assertFalse(self.closed_set.contains("https://example.com/img.jpg?utm_source=test", "aabbcc"))
        self.closed_set.add("https://example.com/img.jpg", "aabbcc")
        # Same URL with different query parameters is recognized as duplicate
        self.assertTrue(self.closed_set.contains("https://example.com/img.jpg?tracking=123", "different"))
        # Same domain with same PDQ hash is recognized as duplicate
        self.assertTrue(self.closed_set.contains("https://example.com/thumb.jpg", "AABBCC"))
        # Different domain with same PDQ hash is NOT blocked (allows provenance comparison!)
        self.assertFalse(self.closed_set.contains("https://other.com/photo.png", "AABBCC"))
        # Completely new URL and hash is not in set
        self.assertFalse(self.closed_set.contains("https://other.com/different.png", "ddeeff"))

    def test_partial_match_ledger(self):
        """Test ledger entry recording and dictionary serialization."""
        cand = NeighborCandidate(
            url="https://wire.com/master.jpg",
            local_path="data/cache/master.jpg",
            width=1920,
            height=1080,
            pdq_hex="1111",
            timestamp="2021-01-01T00:00:00Z",
            domain="wire.com",
            is_authoritative_wire=True,
            source_hop="yandex",
        )
        match = MatchResult(
            is_match=True,
            is_partial_crop=True,
            crop_type="target_is_crop_of_candidate",
            pdq_distance=85,
            inlier_ratio=0.92,
            inlier_count=1200,
            bounding_box=[100, 100, 640, 360],
            scale_factor=2.5,
            similarity_score=0.95,
            details="Target crop inside master",
        )

        entry = self.ledger.record_match(cand, match)
        self.assertEqual(len(self.ledger.entries), 1)
        self.assertEqual(entry.crop_type, "target_is_crop_of_candidate")
        self.assertEqual(entry.bounding_box, [100, 100, 640, 360])
        self.assertEqual(entry.inlier_count, 1200)

        entries_dict = self.ledger.get_entries()
        self.assertEqual(len(entries_dict), 1)
        self.assertEqual(entries_dict[0]["resolution"], [1920, 1080])

    def test_scoring_uncropped_master_vs_unrelated(self):
        """Test that uncropped authoritative wire master gets much lower f(n) than unrelated images."""
        master_cand = NeighborCandidate(
            url="https://apnews.com/photo_master.jpg",
            local_path="data/cache/ap_master.jpg",
            width=2560,
            height=1440,
            pdq_hex="2222",
            timestamp="2021-01-01T00:00:00Z",  # Older than target (2022)
            domain="apnews.com",
            is_authoritative_wire=True,
            source_hop="yandex",
        )
        master_match = MatchResult(
            is_match=True,
            is_partial_crop=True,
            crop_type="target_is_crop_of_candidate",
            pdq_distance=70,
            inlier_ratio=0.95,
            inlier_count=1500,
            bounding_box=[0, 100, 1920, 1080],
            scale_factor=3.0,
            similarity_score=0.98,
            details="Target is crop of AP master",
        )

        g_master, h_master, f_master = self.engine.compute_scores(
            candidate=master_cand,
            match=master_match,
            target_candidate=self.target_cand,
            depth=1,
        )

        unrelated_cand = NeighborCandidate(
            url="https://random.com/noise.jpg",
            local_path="data/cache/noise.jpg",
            width=400,
            height=300,
            pdq_hex="ffff",
            timestamp="2023-01-01T00:00:00Z",
            domain="random.com",
            is_authoritative_wire=False,
            source_hop="yandex",
        )
        unrelated_match = MatchResult(
            is_match=False,
            is_partial_crop=False,
            crop_type="none",
            pdq_distance=120,
            inlier_ratio=0.05,
            inlier_count=2,
            bounding_box=None,
            scale_factor=1.0,
            similarity_score=0.1,
            details="Unrelated image",
        )

        g_unrelated, h_unrelated, f_unrelated = self.engine.compute_scores(
            candidate=unrelated_cand,
            match=unrelated_match,
            target_candidate=self.target_cand,
            depth=1,
        )

        # Master must have substantially lower cost f(n)
        self.assertLess(f_master, 25.0)
        self.assertGreater(f_unrelated, 150.0)
        self.assertLess(f_master, f_unrelated)

    def test_origin_confidence_calculation(self):
        """Test calculation of Patient Zero confidence score."""
        wire_master_cand = NeighborCandidate(
            url="https://reuters.com/wire.jpg",
            local_path="data/cache/wire.jpg",
            width=3000,
            height=2000,
            pdq_hex="3333",
            timestamp="2021-01-01T00:00:00Z",
            domain="reuters.com",
            is_authoritative_wire=True,
            source_hop="crawl",
        )
        match = MatchResult(
            is_match=True,
            is_partial_crop=True,
            crop_type="target_is_crop_of_candidate",
            pdq_distance=60,
            inlier_ratio=0.98,
            inlier_count=1800,
            bounding_box=[100, 100, 1200, 800],
            scale_factor=3.5,
            similarity_score=0.99,
            details="Target is crop of Reuters wire",
        )

        conf = self.engine.evaluate_origin_confidence(
            candidate=wire_master_cand,
            match=match,
            target_candidate=self.target_cand,
        )
        self.assertGreaterEqual(conf, 0.85)


if __name__ == "__main__":
    unittest.main()