"""Tests for Chronology Engine and 50-Place Publication Sequence (§2.4c)."""

from datetime import datetime, timezone, timedelta
import pytest
from src.source_attribution.social_graph.chronology import ChronologyEngine
from src.source_attribution.reranking.embedder import MultimodalEmbedder
from src.common.schemas import PublicationRecord, EarliestCandidate


def test_format_elapsed():
    assert ChronologyEngine.format_elapsed(timedelta(seconds=20)) == "+0m"
    assert ChronologyEngine.format_elapsed(timedelta(minutes=14)) == "+14m"
    assert ChronologyEngine.format_elapsed(timedelta(hours=2, minutes=15)) == "+2h 15m"


def test_chronology_engine_patient_zero_and_50_places():
    # Build 70 mock candidates with varying timestamps and platforms
    candidates = []
    base_time = datetime(2026, 8, 18, 6, 30, 0, tzinfo=timezone.utc)
    
    platforms = ["open_web", "x", "reddit", "youtube", "reverse_visual_search"]
    for i in range(70):
        t = base_time + timedelta(minutes=i * 12)
        candidates.append({
            "platform": platforms[i % len(platforms)],
            "account": f"venue_account_{i}",
            "post_url": f"https://newsportal.in/post/{i}",
            "title": f"Incident report #{i}",
            "text": f"Footage and eyewitness details regarding event #{i}",
            "published_at": t.isoformat(),
            "timestamp_type": "published",
            "media_verification": "exact",
            "evidence_status": "verified",
            "is_synthetic": False,
            "similarity": 0.85 - (i * 0.002)
        })

    # Call chronology engine requesting up to 50 publication places
    publications, origin, summary = ChronologyEngine.build_publication_chronology(
        candidates=candidates,
        min_similarity=0.40,
        max_publications=50
    )

    # 1. Verify Patient Zero isolation
    assert origin is not None
    assert isinstance(origin, EarliestCandidate)
    assert origin.account == "venue_account_0"
    assert origin.platform == "open_web"
    assert origin.timestamp == base_time.isoformat()
    assert origin.confidence == "high"

    # 2. Verify 50 publication records
    assert len(publications) == 50
    assert publications[0].rank == 1
    assert publications[0].role == "primary_origin"
    assert publications[0].elapsed_time == "+0m"
    assert publications[0].account == "venue_account_0"

    assert publications[1].rank == 2
    assert publications[1].role == "early_reporting"
    assert publications[1].elapsed_time == "+12m"

    assert publications[-1].rank == 50
    assert publications[-1].role in ("viral_spread", "current_circulation")

    # 3. Verify circulation summary
    assert summary["total_candidates_scanned"] == 70
    assert summary["verified_publications_tracked"] == 70
    assert summary["first_50_places_count"] == 50
    assert summary["primary_origin_account"] == "venue_account_0"
    assert summary["dissemination_velocity_posts_per_hour"] > 0
    assert "open_web" in summary["active_platforms_breakdown"]


def test_chronology_engine_deduplication():
    # Duplicate URLs should be collapsed
    candidates = [
        {"platform": "x", "account": "@acc1", "post_url": "https://x.com/post/1", "similarity": 0.9, "published_at": "2026-08-18T08:00:00Z", "timestamp_type": "platform_decoded", "media_verification": "exact", "evidence_status": "verified", "is_synthetic": False},
        {"platform": "x", "account": "@acc1", "post_url": "https://x.com/post/1", "similarity": 0.9, "published_at": "2026-08-18T08:00:00Z", "timestamp_type": "platform_decoded", "media_verification": "exact", "evidence_status": "verified", "is_synthetic": False},
        {"platform": "reddit", "account": "u/user2", "post_url": "https://reddit.com/post/2", "similarity": 0.8, "published_at": "2026-08-18T08:30:00Z", "timestamp_type": "published", "media_verification": "near_duplicate", "evidence_status": "verified", "is_synthetic": False},
    ]
    publications, origin, summary = ChronologyEngine.build_publication_chronology(candidates)
    assert len(publications) == 2
    assert publications[0].account == "@acc1"
    assert publications[1].account == "u/user2"


def test_score_candidates_returns_sim_map():
    embedder = MultimodalEmbedder()
    candidates = [
        {"platform": "x", "account": "@breaking", "title": "protest rally footage", "text": "crowd in street", "post_url": "https://x.com/breaking/1"},
        {"platform": "web", "account": "food.com", "title": "best burger recipes", "text": "delicious cheese", "post_url": "https://food.com/recipes/1"}
    ]
    ranked, sim_map = embedder.score_candidates("protest rally in street", candidates, top_k=2)
    assert len(ranked) == 2
    assert "https://x.com/breaking/1" in sim_map
    assert sim_map["https://x.com/breaking/1"] > sim_map["https://food.com/recipes/1"]
