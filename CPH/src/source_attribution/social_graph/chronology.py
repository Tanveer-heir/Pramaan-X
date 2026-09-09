"""Chronological dissemination and conservative origin tracing.

Only verified media occurrences with explicitly typed publication timestamps are
eligible for the earliest-observed lead and publication sequence.
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
import re
from src.common.schemas import PublicationRecord, EarliestCandidate
from src.common.logger import logger
from src.source_attribution.evidence import (
    candidate_has_usable_timestamp,
    candidate_is_verified,
    normalize_candidate,
)


class ChronologyEngine:
    """Extracts a chronology without making unsupported Patient Zero claims."""

    @classmethod
    def extract_forensic_timestamp(
        cls,
        ts_str: Optional[str] = None,
        url: str = "",
        text: str = "",
        fallback_idx: int = 0
    ) -> Optional[datetime]:
        """
        Parses an explicitly supplied connector timestamp.  URL slugs, snippets,
        retrieval time, and fallback counters are intentionally ignored here;
        connectors must establish those provenance types before chronology.
        """
        # 1. If an explicit timestamp is provided by source metadata, respect it.
        if ts_str:
            clean = ts_str.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(clean)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass

        return None

    @staticmethod
    def parse_iso_or_fallback(ts_str: Optional[str], fallback_idx: int = 0) -> Optional[datetime]:
        """Parse an ISO timestamp; return ``None`` when it is not observed."""
        if ts_str:
            clean = ts_str.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(clean)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
        
        return None

    @classmethod
    def format_elapsed(cls, delta: timedelta) -> str:
        """Formats elapsed duration into a human-readable forensic offset string."""
        total_seconds = int(delta.total_seconds())
        if total_seconds <= 60:
            return "+0m"
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        if hours == 0:
            return f"+{minutes}m"
        return f"+{hours}h {minutes:02d}m"

    @classmethod
    def build_publication_chronology(
        cls,
        candidates: List[Dict[str, Any]],
        similarity_map: Optional[Dict[str, float]] = None,
        min_similarity: float = 0.40,
        max_publications: int = 50,
        event_date: Optional[str] = None
    ) -> Tuple[List[PublicationRecord], Optional[EarliestCandidate], Dict[str, Any]]:
        """
        Takes raw candidates from all search channels, dedupes, sorts chronologically,
        and extracts the first 50 publication places and Primary Origin.
        Filters out anachronistic results dated before the verified historical incident.
        """
        logger.info("chronology.build.start", total_candidates=len(candidates), event_date=event_date)
        similarity_map = similarity_map or {}

        # Parse historical event threshold (e.g. 2021-01-26 -> reject items before Jan 2021)
        min_allowed_dt = None
        if event_date:
            try:
                ev_m = re.search(r'(20[0-2][0-9])[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12][0-9]|3[01])', event_date)
                if ev_m:
                    ev_dt = datetime(int(ev_m.group(1)), int(ev_m.group(2)), int(ev_m.group(3)), tzinfo=timezone.utc)
                    min_allowed_dt = ev_dt - timedelta(days=30)
                else:
                    # Year only check
                    y_m = re.search(r'\b(20[0-2][0-9])\b', event_date)
                    if y_m:
                        min_allowed_dt = datetime(int(y_m.group(1)), 1, 1, tzinfo=timezone.utc) - timedelta(days=30)
            except Exception:
                min_allowed_dt = None

        # 1. Normalize and deduplicate only candidates which have passed the
        # media verification gate.  A text hit, fixture, placeholder, connector
        # error, or retrieval-only observation is an investigative lead, never a
        # publication in an origin chronology.
        seen_keys = set()
        deduped = []

        for idx, c in enumerate(candidates):
            normalized = normalize_candidate(c, str(c.get("source_connector") or c.get("platform") or "unknown"))
            if not candidate_is_verified(normalized):
                continue
            if not candidate_has_usable_timestamp(normalized):
                continue

            url = normalized.get("post_url") or normalized.get("url") or ""
            platform = normalized.get("platform", "unknown")
            account = normalized.get("account", "unknown")
            title = normalized.get("title", "")
            snippet = normalized.get("text") or normalized.get("snippet", "")
            norm_key = url if url else f"{platform}:{account}:{title[:30]}"

            if norm_key in seen_keys:
                continue
            seen_keys.add(norm_key)

            # Extract or assign similarity score
            sim = similarity_map.get(norm_key) or similarity_map.get(url)
            if sim is None:
                sim = normalized.get("similarity", 0.50)

            # Filter out non-matching noise
            if sim < min_similarity and normalized.get("media_verification") not in {"exact", "near_duplicate"}:
                continue

            # Use only a timestamp explicitly supplied by a connector and keep
            # its provenance.  Never parse a retrieval time as publication time.
            ts_str = normalized.get("published_at")
            try:
                clean = str(ts_str).replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            # Filter out anachronistic results dated long before the incident
            if min_allowed_dt and dt < min_allowed_dt:
                logger.debug("chronology.filtered_anachronism", url=url[:50], dt=dt.isoformat(), min_allowed=min_allowed_dt.isoformat())
                continue

            direct_url = normalized.get("origin_post_url") or url
            cand_plat = normalized.get("platform", platform)
            if normalized.get("origin_post_url") and ("x.com" in normalized["origin_post_url"] or "twitter.com" in normalized["origin_post_url"]):
                cand_plat = "x"

            deduped.append({
                "candidate": normalized,
                "dt": dt,
                "similarity": sim,
                "url": direct_url,
                "platform": cand_plat,
                "account": normalized.get("account", account),
                "title": title,
                "snippet": normalized.get("text") or normalized.get("snippet", ""),
                "has_media": True,
                "timestamp_type": normalized.get("timestamp_type"),
            })

        # 2. Sort strictly chronologically: earliest to latest
        deduped.sort(key=lambda x: x["dt"])

        if not deduped:
            logger.warn("chronology.no_valid_candidates")
            return [], None, {
                "total_candidates_scanned": len(candidates),
                "verified_publications_tracked": 0,
                "timestamp_provenance": [],
            }

        # 3. Identify the earliest observed verified media occurrence.  The
        # confidence is deliberately conservative: exact media plus a connector
        # publication/archive timestamp is needed for a high-confidence label.
        first_entry = deduped[0]

        origin_dt = first_entry["dt"]
        origin_candidate = EarliestCandidate(
            instance_id=first_entry["candidate"].get("instance_id"),
            url=first_entry["url"] if first_entry["url"] else None,
            account=first_entry["account"],
            platform=first_entry["platform"],
            timestamp=origin_dt.isoformat(),
            confidence=(
                "high"
                if first_entry["candidate"].get("media_verification") == "exact"
                and first_entry.get("timestamp_type") in {"published", "archive_observed"}
                else "moderate"
            ),
            title=first_entry["title"]
        )

        # 4. Filter out any remaining pre-origin noise and build first 50 records
        timeline_candidates = [d for d in deduped if d["dt"] >= origin_dt]
        if not timeline_candidates:
            timeline_candidates = deduped

        publications: List[PublicationRecord] = []
        platform_counts: Dict[str, int] = {}

        for rank, item in enumerate(timeline_candidates[:max_publications], start=1):
            elapsed_delta = item["dt"] - origin_dt
            elapsed_str = cls.format_elapsed(elapsed_delta)
            elapsed_hours = elapsed_delta.total_seconds() / 3600.0

            # Assign propagation role
            if rank == 1:
                role = "primary_origin"
            elif elapsed_hours <= 2.0:
                role = "early_reporting"
            elif elapsed_hours <= 24.0:
                role = "viral_spread"
            else:
                role = "current_circulation"

            plat = item["platform"]
            platform_counts[plat] = platform_counts.get(plat, 0) + 1

            publications.append(
                PublicationRecord(
                    rank=rank,
                    role=role,
                    platform=plat,
                    account=item["account"],
                    timestamp=item["dt"].isoformat(),
                    elapsed_time=elapsed_str,
                    post_url=item["url"],
                    similarity=round(item["similarity"], 3),
                    title=item["title"],
                    snippet=item["snippet"][:120] if item["snippet"] else ""
                )
            )

        # 5. Circulation Summary & Metrics
        latest_dt = deduped[-1]["dt"]
        time_span_hours = max(0.1, (latest_dt - origin_dt).total_seconds() / 3600.0)
        velocity = round(len(deduped) / time_span_hours, 2)

        circulation_summary = {
            "total_candidates_scanned": len(candidates),
            "verified_publications_tracked": len(deduped),
            "primary_origin_account": origin_candidate.account,
            "primary_origin_platform": origin_candidate.platform,
            "first_published_utc": origin_dt.isoformat(),
            "latest_published_utc": latest_dt.isoformat(),
            "total_span_hours": round(time_span_hours, 1),
            "dissemination_velocity_posts_per_hour": velocity,
            "active_platforms_breakdown": platform_counts,
            "first_50_places_count": len(publications),
            "timestamp_provenance": [
                {
                    "url": item["url"],
                    "published_at": item["dt"].isoformat(),
                    "timestamp_type": item["timestamp_type"],
                    "media_verification": item["candidate"].get("media_verification"),
                    "source_connector": item["candidate"].get("source_connector"),
                }
                for item in timeline_candidates[:max_publications]
            ]
        }

        logger.info(
            "chronology.build.complete",
            origin=origin_candidate.account,
            places_count=len(publications),
            total_scanned=len(candidates)
        )

        return publications, origin_candidate, circulation_summary
