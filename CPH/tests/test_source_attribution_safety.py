"""Focused regression tests for conservative source-attribution evidence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.source_attribution.evidence import redact_secrets
from src.source_attribution.pipeline import SourceAttributionPipeline
from src.source_attribution.social_graph.chronology import ChronologyEngine
from src.source_attribution.evidence import SourceAttributionEvidence


def _verified(url: str, when: str, *, timestamp_type: str = "published", media: str = "exact") -> dict:
    return {
        "platform": "open_web",
        "account": "real-source",
        "post_url": url,
        "title": "Observed source",
        "text": "Observed media occurrence",
        "published_at": when,
        "timestamp_type": timestamp_type,
        "retrieved_at": "2026-09-09T00:00:00+00:00",
        "media_verification": media,
        "evidence_status": "verified" if media in {"exact", "near_duplicate"} else "unverified",
        "is_synthetic": False,
        "similarity": 0.9,
    }


def test_unverified_text_and_retrieval_time_cannot_be_origin():
    candidates = [
        {
            **_verified("https://real-source.invalid/text", "2026-09-08T00:00:00Z", media="unverified_text"),
            "evidence_status": "unverified",
        },
        _verified("https://real-source.invalid/retrieval", "2026-09-07T00:00:00Z", timestamp_type="retrieved_at"),
    ]
    publications, origin, summary = ChronologyEngine.build_publication_chronology(candidates)
    assert publications == []
    assert origin is None
    assert summary["timestamp_provenance"] == []


def test_verified_ranking_preserves_timestamp_provenance():
    candidates = [
        _verified("https://news-source.in/later", "2026-09-08T10:00:00Z", timestamp_type="archive_observed", media="near_duplicate"),
        _verified("https://news-source.in/first", "2026-09-08T09:00:00Z"),
    ]
    publications, origin, summary = ChronologyEngine.build_publication_chronology(candidates)
    assert origin is not None
    assert origin.url == "https://news-source.in/first"
    assert publications[0].post_url == "https://news-source.in/first"
    assert summary["timestamp_provenance"][0]["timestamp_type"] == "published"
    assert summary["timestamp_provenance"][1]["timestamp_type"] == "archive_observed"


def test_missing_model_and_meaningless_upload_name_do_not_search(tmp_path):
    media = tmp_path / "upload.webp"
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(media, format="WEBP")

    evidence = asyncio.run(
        SourceAttributionPipeline(output_dir=str(tmp_path / "graphs")).execute(str(media))
    )
    payload = evidence.model_dump(mode="json")
    serialized = json.dumps(payload).lower()
    assert payload["earliest_candidate"] is None
    assert payload["query_provenance"]["status"] == "insufficient_search_context"
    assert payload["query_provenance"]["search_performed"] is False
    assert "upload upload" not in serialized
    assert "news.example.com" not in serialized
    assert "no verified public occurrence found" in serialized


def test_secret_redaction_never_exposes_configured_api_key(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "unit-test-secret-key")
    value = {"message": "unit-test-secret-key", "nested": ["unit-test-secret-key"]}
    redacted = redact_secrets(value)
    assert "unit-test-secret-key" not in json.dumps(redacted)
    assert "[REDACTED]" in json.dumps(redacted)


@pytest.mark.asyncio
async def test_failed_web_connector_is_empty_and_reports_error(monkeypatch):
    import src.source_attribution.search_crawlers.web_client as module

    class FailingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def text(self, *_args, **_kwargs):
            raise RuntimeError("rate limited")

    monkeypatch.delenv("FORENSIC_BENCHMARK_MODE", raising=False)
    monkeypatch.setattr(module, "HAS_DDGS", True)
    monkeypatch.setattr(module, "DDGS", FailingDDGS, raising=False)
    client = module.WebSearchClient()
    assert await client.search("genuine context", max_results=2) == []
    assert client.last_status["status"] == "error"
    assert client.last_status["result_count"] == 0
    assert "rate limited" in client.last_status["failure_reason"]


@pytest.mark.asyncio
async def test_fixtures_are_only_benchmark_and_explicitly_labelled(monkeypatch):
    import src.source_attribution.search_crawlers.web_client as web_module
    import src.source_attribution.search_crawlers.youtube_client as youtube_module
    import src.source_attribution.search_crawlers.reddit_client as reddit_module
    import src.source_attribution.search_crawlers.x_client as x_module
    from src.source_attribution.reverse_search.google_vision import GoogleVisionClient

    monkeypatch.setenv("FORENSIC_BENCHMARK_MODE", "1")
    clients = [
        (web_module.WebSearchClient(), "search", {"query": "benchmark", "max_results": 1}),
        (youtube_module.YouTubeSearchClient(), "search", {"query": "benchmark", "max_results": 1}),
        (reddit_module.RedditSearchClient(), "search", {"query": "benchmark", "limit": 1}),
        (x_module.XSearchClient(), "search", {"query": "benchmark", "limit": 1}),
    ]
    for client, method, kwargs in clients:
        result = await getattr(client, method)(**kwargs)
        assert result and result[0]["evidence_status"] == "fixture"
        assert result[0]["is_synthetic"] is True
        assert result[0]["source_connector"]

    google_result = GoogleVisionClient().search_image("not-used.jpg")
    assert google_result and google_result[0]["evidence_status"] == "fixture"
    assert google_result[0]["is_synthetic"] is True

    monkeypatch.delenv("FORENSIC_BENCHMARK_MODE", raising=False)
    monkeypatch.setattr(web_module, "HAS_DDGS", False)
    assert await web_module.WebSearchClient().search("benchmark", max_results=1) == []


def test_source_attribution_api_returns_native_extended_payload(monkeypatch):
    import src.source_attribution.pipeline as pipeline_module
    import src.api.main as api_module

    expected = SourceAttributionEvidence(
        circulation_summary={"native": True},
        connector_coverage={"open_web": {"status": "unavailable"}},
        limitations=["No verified public occurrence found."],
        query_provenance={"status": "generated"},
    )

    class FakePipeline:
        async def execute(self, **kwargs):
            assert kwargs["media_path"] == "/tmp/upload.webp"
            assert kwargs["investigator_context"] == "robotaxi event"
            return expected

    monkeypatch.setattr(pipeline_module, "SourceAttributionPipeline", FakePipeline)
    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/v1/source-attribution/analyze",
            json={"media_path": "/tmp/upload.webp", "investigator_context": "robotaxi event"},
        )
    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")
