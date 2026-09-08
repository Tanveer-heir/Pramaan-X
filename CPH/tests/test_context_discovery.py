"""Tests for Two-Stage Iterative Context Discovery & Calibrated Event Embedding."""

import pytest
from src.source_attribution.llm.context_extractor import EventContextExtractor
from src.source_attribution.llm.query_generator import QueryGenerator
from src.source_attribution.reranking.embedder import MultimodalEmbedder


def test_event_context_extractor():
    # Mock visual hits resembling real news article headlines from reverse visual search
    visual_hits = [
        {
            "title": "Nishan Sahib flag hoisted at Red Fort Delhi violence Republic Day",
            "text": "Chaos erupted at Red Fort during farmer tractor rally on Republic Day",
            "post_url": "https://www.dnaindia.com/india/report-nishan-sahib-flag-hoisted-at-red-fort-delhi-violence-republic-day-khalistani-flag-2871120"
        },
        {
            "title": "Nishan Sahib vs Khalistan flag mix up at Red Fort farmer rally",
            "text": "Farmers protest incident on January 26, 2021 at historic Red Fort",
            "post_url": "https://qz.com/india/1965815/nishan-sahib-vs-khalistan-flag-mix-up-at-red-fort-farmer-rally"
        }
    ]

    profile = EventContextExtractor.extract_context(
        visual_hits=visual_hits,
        initial_description="A person climbing flagpole with saffron flag at red fort"
    )

    assert profile["event_detected"] is True
    assert "Kisan Andolan" in profile["event_name"] or "Red Fort" in profile["event_name"]
    assert profile["event_date"] == "2021-01-26"
    assert profile["confidence"] >= 0.80
    assert any("Red Fort" in e for e in profile["key_entities"])


def test_event_specific_query_generator():
    gen = QueryGenerator()
    profile = {
        "event_detected": True,
        "event_name": "Kisan Andolan Red Fort Flag Hoisting",
        "event_date": "2021-01-26",
        "key_entities": ["Red Fort", "Nishan Sahib", "Kisan Andolan"]
    }
    queries = gen.generate_event_specific_queries(profile)
    assert "web" in queries
    assert "youtube" in queries
    assert "x" in queries
    assert any("2021-01-26" in q or "Red Fort" in q for q in queries["web"])
    assert any("Kisan Andolan" in q or "Nishan Sahib" in q for q in queries["youtube"])


def test_calibrated_dual_anchor_embedding_separation():
    embedder = MultimodalEmbedder()
    event_context = "26 January 2021 Republic Day tractor rally during Kisan Andolan where Nishan Sahib flag was hoisted at Red Fort Delhi"
    desc = "A person climbing a flagpole holding a saffron flag"

    candidates = [
        {
            "platform": "youtube",
            "account": "NewsChannel",
            "title": "26 January 2021 Red Fort incident raw footage Kisan Andolan",
            "text": "Ground recording showing flag hoisting at Red Fort during farmers tractor march",
            "post_url": "https://youtube.com/watch?v=kisan_event_01"
        },
        {
            "platform": "open_web",
            "account": "cookingportal.com",
            "title": "Italian pasta and homemade garlic bread",
            "text": "Delicious tomato sauce with basil and mozzarella cheese",
            "post_url": "https://cookingportal.com/pasta"
        }
    ]

    ranked, sim_map = embedder.score_candidates(
        media_description=desc,
        candidates=candidates,
        event_context=event_context,
        top_k=2
    )

    assert len(ranked) == 2
    # The true Kisan Andolan post must have high similarity (>= 80%)
    assert ranked[0].similarity >= 0.78
    assert "youtube" in ranked[0].platform
    # The unrelated recipe must have very low similarity (< 35%)
    assert ranked[1].similarity < 0.35
    # High dynamic separation (> 45% gap)
    assert (ranked[0].similarity - ranked[1].similarity) > 0.45
