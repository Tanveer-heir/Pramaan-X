"""
Free X / Twitter Search Crawler & Forensic Simulator (§2.4d).
Allows querying viral posts without Twitter's $100/mo paywalled API.
Scales up to ~50 candidates per platform search using public syndication indices.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import re
from src.common.logger import logger
from src.common.config import settings

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False


class XSearchClient:
    """Zero-cost X/Twitter crawler using public syndication or forensic simulation."""

    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token or settings.TWITTER_BEARER_TOKEN
        self.last_status: Dict[str, Any] = {
            "source_connector": "x",
            "status": "not_run",
            "result_count": 0,
        }

    async def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        logger.info("crawler.x.search", query=query, limit=limit)

        if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1":
            results = self._fallback_fixtures(query, limit)
            self.last_status = {
                "source_connector": "x",
                "status": "fixture",
                "result_count": len(results),
                "failure_reason": "benchmark fixture enabled",
            }
            return results

        results = []
        error_reason = None

        # 1. Open Web Index search for X/Twitter posts
        if HAS_DDGS:
            try:
                # Query twitter.com first (better historical index for 2021)
                with DDGS() as ddgs:
                    raw_tweets = list(ddgs.text(f"{query} site:twitter.com", max_results=limit))
                    if len(raw_tweets) < limit:
                        raw_tweets.extend(list(ddgs.text(f"{query} site:x.com", max_results=limit - len(raw_tweets))))

                for item in raw_tweets:
                    href = item.get("href", "")
                    title = item.get("title", "")
                    body = item.get("body", "")

                    # Extract account handle and status ID if present in URL: x.com/<handle>/status/...
                    handle_match = re.search(r'(?:x|twitter)\.com/([^/]+)/status/(\d+)', href)
                    if handle_match:
                        account = f"@{handle_match.group(1)}"
                        snowflake = int(handle_match.group(2))
                        ts_ms = (snowflake >> 22) + 1288834974657
                        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                        created_str = dt.isoformat()
                    elif "/status/" in href or "/events/" in href:
                        # A status URL without a public numeric status id is not
                        # sufficient to attribute an account or publication time.
                        continue
                    else:
                        # Skip analytics, ads, and policy pages
                        continue

                    results.append({
                        "platform": "x",
                        "account": account,
                        "post_url": href,
                        "title": title,
                        "text": f"{title}. {body}".strip(),
                        "published_at": created_str,
                        "timestamp_type": "platform_decoded",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "media_verification": "not_downloaded",
                        "evidence_status": "unverified",
                        "source_connector": "x",
                        "is_synthetic": False,
                        "has_media": True if ("video" in title.lower() or "photo" in title.lower() or "pic.twitter.com" in body.lower()) else False
                    })

                if results:
                    self.last_status = {
                        "source_connector": "x",
                        "status": "degraded",
                        "result_count": len(results),
                        "failure_reason": "Results obtained from a public web index; media was not downloaded",
                    }
                    logger.info("crawler.x.ddgs_success", count=len(results))
                    return results[:limit]
            except Exception as e:
                error_reason = str(e)
                logger.warn("crawler.x.ddgs_error", error=str(e))
                self.last_status = {
                    "source_connector": "x",
                    "status": "error",
                    "result_count": 0,
                    "failure_reason": str(e),
                }

        self.last_status = {
            "source_connector": "x",
            "status": "error" if error_reason else ("unavailable" if not HAS_DDGS else "empty"),
            "result_count": 0,
            "failure_reason": error_reason or ("No public X results returned" if HAS_DDGS else "DuckDuckGo search is unavailable"),
        }
        return []

    def _fallback_fixtures(self, query: str, limit: int) -> List[Dict[str, Any]]:
        # 2. Scaled Forensic structured hits (Benchmark Mode Only)
        if os.environ.get("FORENSIC_BENCHMARK_MODE") != "1":
            return []
        results = [{
            "platform": "x",
            "account": "@punjab_alert",
            "post_url": f"https://x.com/punjab_alert/status/{1825000000000000000 + abs(hash(query)) % 100000}",
            "title": "",
            "text": f"BREAKING: Footage emerges showing {query}. Crowd gathering rapidly. #BreakingNews",
            "published_at": "2021-01-01T10:05:00Z",
            "timestamp_type": "platform_decoded",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "media_verification": "not_downloaded",
            "evidence_status": "fixture",
            "source_connector": "x",
            "is_synthetic": True,
            "reposts": 142,
            "likes": 520
        }]
        for i in range(1, min(limit, 50)):
            results.append({
                "platform": "x",
                "account": f"@alert_feed_{i % 6 + 1}",
                "post_url": f"https://x.com/alert_feed_{i % 6 + 1}/status/{1825000000000000000 + i * 137}",
                "title": "",
                "text": f"Tweet #{i + 1}: Breaking eyewitness update on {query}. #Alert #IncidentReport",
                "published_at": f"2021-01-01T{7 + (i % 14):02d}:{(i * 17) % 60:02d}:00Z",
                "timestamp_type": "platform_decoded",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "media_verification": "not_downloaded",
                "evidence_status": "fixture",
                "source_connector": "x",
                "is_synthetic": True,
                "reposts": 20 + i * 8,
                "likes": 80 + i * 25
            })
        return results
