"""Tests for Free Search Crawlers Subsystem (§2.4d)."""

import pytest
from src.source_attribution.search_crawlers import (
    WebSearchClient,
    RedditSearchClient,
    YouTubeSearchClient,
    XSearchClient
)

@pytest.mark.asyncio
async def test_web_search_client(monkeypatch):
    monkeypatch.setenv("FORENSIC_BENCHMARK_MODE", "1")
    client = WebSearchClient()
    results = await client.search("Chandigarh police protest rally", max_results=5)
    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert first["platform"] == "open_web"
    assert "post_url" in first
    assert "account" in first
    assert "title" in first
    assert first["evidence_status"] == "fixture"
    assert first["is_synthetic"] is True

@pytest.mark.asyncio
async def test_reddit_search_client(monkeypatch):
    monkeypatch.setenv("FORENSIC_BENCHMARK_MODE", "1")
    client = RedditSearchClient()
    results = await client.search("protest rally video", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert first["platform"] == "reddit"
    assert first["account"].startswith("u/")
    assert "score" in first
    assert first["evidence_status"] == "fixture"
    assert first["is_synthetic"] is True

@pytest.mark.asyncio
async def test_youtube_search_client(monkeypatch):
    monkeypatch.setenv("FORENSIC_BENCHMARK_MODE", "1")
    client = YouTubeSearchClient()
    results = await client.search("incident rally video", max_results=5)
    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert first["platform"] == "youtube"
    assert "post_url" in first
    assert first["evidence_status"] == "fixture"
    assert first["is_synthetic"] is True

@pytest.mark.asyncio
async def test_x_search_client(monkeypatch):
    monkeypatch.setenv("FORENSIC_BENCHMARK_MODE", "1")
    client = XSearchClient()
    results = await client.search("rally breaking footage", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert first["platform"] == "x"
    assert first["account"].startswith("@")
    assert first["evidence_status"] == "fixture"
    assert first["is_synthetic"] is True
