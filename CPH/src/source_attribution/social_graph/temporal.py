"""
Temporal Weighting of Search Sources (§2.4e).
Dynamically blends fast platform search (Reddit/X/Telegram) with deep reverse/web search (Google Vision/Yandex/Open Web)
based on content age from ingestion.
"""

from datetime import datetime, timezone
from typing import Optional
from src.common.schemas import TemporalWeighting
from src.common.logger import logger

class TemporalWeightCalculator:
    """
    Computes age-dependent weighting between fast platform search
    (Reddit/X/Telegram) and deep reverse search (Google Vision/Yandex/Open Web).
    """

    @staticmethod
    def calculate(earliest_seen_iso: Optional[str]) -> TemporalWeighting:
        logger.info("temporal_weighting.calculate", earliest=earliest_seen_iso)
        if not earliest_seen_iso:
            return TemporalWeighting(content_age="new", platform_search_weight=0.7, reverse_search_weight=0.3)

        try:
            cleaned_iso = earliest_seen_iso.replace("Z", "+00:00")
            earliest_dt = datetime.fromisoformat(cleaned_iso)
            now = datetime.now(timezone.utc)
            delta_hours = max(0.0, (now - earliest_dt).total_seconds() / 3600.0)

            # If content is older than 48 hours: crawlers & reverse search have indexed it thoroughly
            if delta_hours >= 48.0:
                return TemporalWeighting(
                    content_age="older",
                    platform_search_weight=0.3,
                    reverse_search_weight=0.7
                )
            else:
                # Fresh breaking content: platform search is dominant
                return TemporalWeighting(
                    content_age="new",
                    platform_search_weight=0.7,
                    reverse_search_weight=0.3
                )
        except Exception as e:
            logger.warn("temporal_weighting.parse_error", error=str(e))
            return TemporalWeighting(content_age="new", platform_search_weight=0.7, reverse_search_weight=0.3)

    @staticmethod
    def blend_confidence(
        platform_score: float,
        reverse_score: float,
        weights: TemporalWeighting
    ) -> float:
        """Computes a blended confidence score using the age-dependent weights."""
        blended = (
            (platform_score * weights.platform_search_weight) +
            (reverse_score * weights.reverse_search_weight)
        )
        return round(min(1.0, max(0.0, blended)), 3)
