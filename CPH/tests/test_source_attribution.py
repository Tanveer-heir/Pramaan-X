"""Tests for Source Attribution Components (§2.4)."""

import pytest
from src.source_attribution.llm.query_generator import QueryGenerator
from src.source_attribution.reranking.embedder import SemanticReranker
from src.source_attribution.social_graph.temporal import TemporalWeightCalculator
from src.source_attribution.social_graph.graph_builder import SocialGraphBuilder
from src.common.schemas import ShortlistCandidate

def test_query_generator():
    gen = QueryGenerator()
    queries = gen.generate_queries("A rally in a public street with banners")
    assert "reddit" in queries
    assert "x" in queries
    assert "youtube" in queries
    assert len(queries["reddit"]) > 0

def test_semantic_reranker():
    reranker = SemanticReranker()
    desc = "street protest rally speech banner"
    candidates = [
        {"platform": "reddit", "account": "u/user1", "title": "street protest rally footage", "text": "video of speech"},
        {"platform": "x", "account": "@user2", "title": "cooking recipe", "text": "how to make soup"},
    ]
    ranked = reranker.rerank(desc, candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].account == "u/user1"
    assert ranked[0].similarity > ranked[1].similarity

def test_temporal_weighting_new_content():
    from datetime import datetime, timezone, timedelta
    recent_ts = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    weights = TemporalWeightCalculator.calculate(recent_ts)
    assert weights.content_age == "new"
    assert weights.platform_search_weight > weights.reverse_search_weight

def test_social_graph_builder():
    candidates = [
        ShortlistCandidate(platform="reddit", account="u/source1", similarity=0.9, timestamp="2026-08-18T10:00:00Z"),
        ShortlistCandidate(platform="x", account="@amplifier2", similarity=0.85, timestamp="2026-08-18T12:00:00Z"),
    ]
    graph = SocialGraphBuilder.build_account_social_graph(candidates)
    assert "nodes" in graph
    assert "links" in graph
    assert len(graph["nodes"]) == 2
    assert len(graph["links"]) == 1
