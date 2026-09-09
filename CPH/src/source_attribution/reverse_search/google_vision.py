"""Google Cloud Vision Web Detection connector.

The connector deliberately fails closed.  A missing credential or an unavailable
client is not evidence and therefore produces no candidates.
"""

import os
from datetime import datetime, timezone
from typing import Any, List

from src.common.logger import logger
from src.common.config import settings

class GoogleVisionClient:
    """Client for Google Cloud Vision Web Detection."""

    def __init__(self, credentials_path: str = None):
        self.credentials_path = credentials_path or settings.GOOGLE_APPLICATION_CREDENTIALS
        self._client = None
        self.last_status: dict[str, Any] = {
            "source_connector": "google_vision_web_detection",
            "status": "unavailable",
            "result_count": 0,
            "failure_reason": "Google Vision Web Detection is not configured",
        }

    def search_image(self, image_path_or_bytes) -> List[dict[str, Any]]:
        """Perform web detection, returning only observations from a real client."""
        logger.info("reverse_search.google_vision.started", creds=bool(self.credentials_path))
        # The repository does not contain a complete Web Detection request.  Do
        # not pretend a configured credential makes it operational.
        if not self.credentials_path:
            reason = "GOOGLE_APPLICATION_CREDENTIALS is not configured"
        else:
            reason = "Google Vision Web Detection client is unavailable"
        if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1":
            self.last_status = {
                "source_connector": "google_vision_web_detection",
                "status": "fixture",
                "result_count": 1,
                "failure_reason": "benchmark fixture enabled",
            }
            return [{
                "platform": "open_web",
                "account": "benchmark_google_vision",
                "post_url": "https://benchmark.invalid/google-vision-match",
                "title": "Benchmark reverse-image fixture",
                "text": "Synthetic Google Vision Web Detection fixture.",
                "published_at": "2021-01-01T00:00:00Z",
                "timestamp_type": "published",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "media_verification": "not_downloaded",
                "evidence_status": "fixture",
                "source_connector": "google_vision_web_detection",
                "is_synthetic": True,
            }]
        self.last_status = {
            "source_connector": "google_vision_web_detection",
            "status": "unavailable",
            "result_count": 0,
            "failure_reason": reason,
        }
        logger.warning("google_vision.unavailable", reason=reason)
        return []
