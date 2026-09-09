"""
Free YouTube Video Search Crawler (§2.4d).
Queries YouTube for video uploads and incident coverage without paid API subscriptions.
Scales up to ~50 candidates per platform search using ddgs.videos and public feeds.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import httpx
from src.common.logger import logger
from src.common.config import settings
from src.source_attribution.evidence import redact_secrets

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False


class YouTubeSearchClient:
    """Free YouTube search client using ddgs.videos or optional free API key."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.YOUTUBE_API_KEY
        self.timeout = 8.0
        self.last_status: Dict[str, Any] = {
            "source_connector": "youtube",
            "status": "not_run",
            "result_count": 0,
        }

    async def search(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        logger.info("crawler.youtube.search", query=query, max_results=max_results)

        if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1":
            results = self._fallback_fixtures(query, max_results)
            self.last_status = {
                "source_connector": "youtube",
                "status": "fixture",
                "result_count": len(results),
                "failure_reason": "benchmark fixture enabled",
            }
            return results

        results = []
        error_reason = None

        # 1. If a free-tier YouTube Data API v3 key is provided:
        if self.api_key:
            try:
                url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": str(min(max_results, 50)),
                    "key": self.api_key
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        items = resp.json().get("items", [])
                        for item in items:
                            snippet = item.get("snippet", {})
                            video_id = item.get("id", {}).get("videoId", "")
                            if not video_id:
                                continue
                            video_description = snippet.get("description", "")
                            published_at = snippet.get("publishedAt")
                            results.append({
                                "platform": "youtube",
                                "account": snippet.get("channelTitle") or snippet.get("channelId") or "",
                                "post_url": f"https://youtube.com/watch?v={video_id}",
                                "title": snippet.get("title", ""),
                                "text": f"{snippet.get('title', '')}. {video_description}".strip(),
                                "description": video_description,
                                "published_at": published_at,
                                "timestamp_type": "published" if published_at else "unknown",
                                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                "media_verification": "not_downloaded",
                                "evidence_status": "unverified",
                                "source_connector": "youtube",
                                "is_synthetic": False,
                                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                                "has_media": True
                            })
                        if results:
                            self.last_status = {
                                "source_connector": "youtube",
                                "status": "ok",
                                "result_count": len(results),
                            }
                            return results[:max_results]
                    elif resp.status_code in {401, 403}:
                        error_reason = f"YouTube Data API returned HTTP {resp.status_code}"
                        self.last_status = {
                            "source_connector": "youtube",
                            "status": "error",
                            "result_count": 0,
                            "failure_reason": error_reason,
                        }
            except Exception as e:
                error_reason = str(redact_secrets(str(e)))
                logger.warn("crawler.youtube.api_error", error=error_reason)
                self.last_status = {
                    "source_connector": "youtube",
                    "status": "error",
                    "result_count": 0,
                    "failure_reason": error_reason,
                }

        # 2. Free DuckDuckGo search for YouTube videos (site:youtube.com)
        if HAS_DDGS and len(results) < max_results:
            try:
                import re
                yt_query = f"{query} site:youtube.com"
                with DDGS() as ddgs:
                    raw_items = list(ddgs.text(yt_query, max_results=max_results))
                for item in raw_items:
                    href = item.get("href") or item.get("link") or ""
                    title = item.get("title", "")
                    body = item.get("body", "")

                    # Extract video ID from youtube URL
                    m = re.search(r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})', href)
                    if m:
                        vid = m.group(1)
                        post_url = f"https://www.youtube.com/watch?v={vid}"
                        thumb_url = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
                    else:
                        post_url = href
                        thumb_url = None

                    # Extract account / channel name if present in title
                    channel = ""
                    if " - YouTube" in title:
                        title_clean = title.replace(" - YouTube", "").strip()
                    else:
                        title_clean = title

                    results.append({
                        "platform": "youtube",
                        "account": channel,
                        "post_url": post_url,
                        "title": title_clean,
                        "text": f"{title_clean}. {body}".strip(),
                        "published_at": None,
                        "timestamp_type": "retrieved_at",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "media_verification": "not_downloaded",
                        "evidence_status": "unverified",
                        "source_connector": "youtube",
                        "is_synthetic": False,
                        "thumbnail_url": thumb_url,
                        "media_url": thumb_url,
                        "has_media": True
                    })
                valid_results = [r for r in results if "watch?v=" in r.get("post_url", "") or "youtu.be/" in r.get("post_url", "")]
                if valid_results:
                    self.last_status = {
                        "source_connector": "youtube",
                        "status": "degraded",
                        "result_count": len(valid_results),
                        "failure_reason": "Results obtained from public web index; publication dates unavailable",
                    }
                    logger.info("crawler.youtube.ddgs_text_success", count=len(valid_results))
                    return valid_results[:max_results]
            except Exception as e:
                error_reason = error_reason or str(redact_secrets(str(e)))
                logger.warn("crawler.youtube.ddgs_error", error=error_reason)

        if not self.api_key and not HAS_DDGS:
            reason = "YouTube Data API key and DuckDuckGo search are unavailable"
        else:
            reason = error_reason or "No public YouTube results returned"
        self.last_status = {
            "source_connector": "youtube",
            "status": "unavailable" if not self.api_key and not HAS_DDGS else ("error" if error_reason else "empty"),
            "result_count": 0,
            "failure_reason": reason,
        }
        return []

    def _fallback_fixtures(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        # 3. Scaled Forensic Fallback Fixtures (Benchmark Mode Only)
        if os.environ.get("FORENSIC_BENCHMARK_MODE") != "1":
            return []
        results = [{
            "platform": "youtube",
            "account": "ChandigarhNewsLive",
            "post_url": f"https://youtube.com/watch?v=chd_vid_{abs(hash(query)) % 1000}",
            "title": f"Full Video: Public rally and speech on {query}",
            "text": f"Raw ground footage showing {query} incident and speech delivered to crowd.",
            "published_at": "2021-01-01T16:00:00Z",
            "timestamp_type": "published",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "media_verification": "not_downloaded",
            "evidence_status": "fixture",
            "source_connector": "youtube",
            "is_synthetic": True,
            "thumbnail_url": "https://img.youtube.com/vi/sample/hqdefault.jpg",
            "has_media": True
        }]
        for i in range(1, min(max_results, 50)):
            results.append({
                "platform": "youtube",
                "account": f"RegionalMediaChannel_{i % 5 + 1}",
                "post_url": f"https://youtube.com/watch?v=vid_cph_{3000 + i}",
                "title": f"Video #{i + 1}: Ground report regarding {query}",
                "text": f"Raw broadcast and citizen camera angle #{i + 1} capturing sequence of events around {query}.",
                "published_at": f"2021-01-01T{10 + (i % 10):02d}:{(i * 13) % 60:02d}:00Z",
                "timestamp_type": "published",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "media_verification": "not_downloaded",
                "evidence_status": "fixture",
                "source_connector": "youtube",
                "is_synthetic": True,
                "thumbnail_url": "https://img.youtube.com/vi/sample/hqdefault.jpg",
                "has_media": True
            })
        return results
