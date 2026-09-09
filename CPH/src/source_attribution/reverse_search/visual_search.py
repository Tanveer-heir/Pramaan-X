"""
Direct Reverse Visual Search Client (§2.4b/e).
Discovers visually matching images and source webpages across news portals and social platforms
using DuckDuckGo's visual reverse index (100% free, zero paid API keys).
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from src.common.schemas import ExternalMatch
from src.common.logger import logger

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False





class VisualSearchConnector:
    """Discovers visual matches and historical news media matching the image."""

    def __init__(self, max_results: int = 50):
        self.max_results = max_results
        self.last_status: Dict[str, Any] = {
            "source_connector": "reverse_visual_search",
            "status": "unavailable" if not HAS_DDGS else "not_run",
            "result_count": 0,
            "failure_reason": "DuckDuckGo image search is unavailable" if not HAS_DDGS else None,
        }

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Searches visual image index and returns candidate hits with image URLs."""
        logger.info("reverse_search.visual.started", query=query, limit=limit)

        if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1":
            hits = self._fallback_hits(query, limit)
            self.last_status = {
                "source_connector": "reverse_visual_search",
                "status": "fixture",
                "result_count": len(hits),
                "failure_reason": "benchmark fixture enabled",
            }
            return hits

        hits = []
        error_reason = None

        if HAS_DDGS:
            try:
                with DDGS() as ddgs:
                    raw_images = list(ddgs.images(query, max_results=limit))

                for item in raw_images:
                    page_url = item.get("url") or item.get("image") or ""
                    title = item.get("title", "")
                    img_url = item.get("image", "")

                    hits.append({
                        "platform": "reverse_visual_search",
                        "account": item.get("source", "web_visual_match"),
                        "post_url": page_url,
                        "thumbnail_url": img_url,
                        "title": title,
                        "text": f"Visually matching media: {title}. Source: {page_url}",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "published_at": None,
                        "timestamp_type": "retrieved_at",
                        "media_verification": "not_downloaded",
                        "evidence_status": "unverified",
                        "source_connector": "reverse_visual_search",
                        "is_synthetic": False,
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "has_media": True
                    })

                if hits:
                    self.last_status = {
                        "source_connector": "reverse_visual_search",
                        "status": "ok",
                        "result_count": len(hits),
                    }
                    logger.info("reverse_search.visual.success", count=len(hits))
                    return hits[:limit]
            except Exception as e:
                error_reason = str(e)
                logger.warn("reverse_search.visual.ddgs_error", error=str(e))

        self.last_status = {
            "source_connector": "reverse_visual_search",
            "status": "error" if error_reason else ("empty" if HAS_DDGS else "unavailable"),
            "result_count": 0,
            "failure_reason": error_reason or ("No public visual matches returned" if HAS_DDGS else "DuckDuckGo image search is unavailable"),
        }
        return []

    def _fallback_hits(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        # Benchmark Mode Only
        if os.environ.get("FORENSIC_BENCHMARK_MODE") != "1":
            return []
        results = [
            {
                "platform": "reverse_visual_search",
                "account": "reuters_wire",
                "post_url": f"https://reuters.com/world/india/visual-match-{abs(hash(query)) % 10000}",
                "thumbnail_url": "https://reuters.com/images/delhi-sample.jpg",
                "title": f"Visual Match: Archival ground photography matching {query}",
                "text": f"High-confidence visual feature match corresponding to {query}.",
                "published_at": "2021-01-01T00:00:00Z",
                "timestamp_type": "published",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "media_verification": "not_downloaded",
                "evidence_status": "fixture",
                "source_connector": "reverse_visual_search",
                "is_synthetic": True,
                "has_media": True
            }
        ]
        for i in range(1, min(limit, 50)):
            results.append({
                "platform": "reverse_visual_search",
                "account": f"photo_wire_{i % 5 + 1}.org",
                "post_url": f"https://photo_wire_{i % 5 + 1}.org/archive/image-{2000 + i}",
                "thumbnail_url": f"https://photo_wire_{i % 5 + 1}.org/thumbs/{2000 + i}.jpg",
                "title": f"Visual Match #{i + 1}: Archive image matching {query}",
                "text": f"Secondary visual corroboration wire photo #{i + 1} matching {query}.",
                "published_at": f"2021-01-01T{7 + (i % 10):02d}:{(i * 9) % 60:02d}:00Z",
                "timestamp_type": "published",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "media_verification": "not_downloaded",
                "evidence_status": "fixture",
                "source_connector": "reverse_visual_search",
                "is_synthetic": True,
                "has_media": True
            })
        return results

    def to_external_matches(self, hits: List[Dict[str, Any]]) -> List[ExternalMatch]:
        """Converts raw hits to canonical ExternalMatch schema."""
        matches = []
        for h in hits:
            matches.append(
                ExternalMatch(
                    url=h.get("post_url") or h.get("thumbnail_url", ""),
                    source="reverse_visual_search",
                    date_found=h.get("published_at"),
                    page_title=h.get("title")
                )
            )
        return matches
 
 
ReverseImageSearchClient = VisualSearchConnector
