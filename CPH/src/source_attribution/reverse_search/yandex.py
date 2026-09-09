"""Yandex reverse-image connector.

Only a real configured integration may produce candidates.  The benchmark
fixture is explicitly labelled and is never treated as public evidence.
"""

import os
from datetime import datetime, timezone
from typing import Any, List
from src.common.logger import logger

class YandexSearchClient:
    """Client for Yandex Reverse Image Search (effective for non-English and deep duplicate matching)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.last_status: dict[str, Any] = {
            "source_connector": "yandex_reverse_image",
            "status": "unavailable",
            "result_count": 0,
            "failure_reason": "Yandex reverse-image client is not configured",
        }

    def search_image(self, image_path_or_bytes) -> List[dict[str, Any]]:
        logger.info("reverse_search.yandex.started")
        if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1":
            self.last_status = {
                "source_connector": "yandex_reverse_image",
                "status": "fixture",
                "result_count": 1,
                "failure_reason": "benchmark fixture enabled",
            }
            return [{
                "platform": "open_web",
                "account": "benchmark_yandex",
                "post_url": "https://benchmark.invalid/yandex-match",
                "title": "Benchmark Yandex reverse-image fixture",
                "text": "Synthetic Yandex reverse-image fixture.",
                "published_at": "2021-01-01T00:00:00Z",
                "timestamp_type": "published",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "media_verification": "not_downloaded",
                "evidence_status": "fixture",
                "source_connector": "yandex_reverse_image",
                "is_synthetic": True,
            }]
        self.last_status = {
            "source_connector": "yandex_reverse_image",
            "status": "unavailable",
            "result_count": 0,
            "failure_reason": "Yandex reverse-image client is not configured",
        }
        logger.warning("yandex.unavailable")
        return []
