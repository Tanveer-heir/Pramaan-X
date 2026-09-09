"""Conservative Source Attribution / Origin Tracing pipeline.

This boundary intentionally distinguishes observations from investigative leads.
Unavailable connectors and text-only search hits are reported, but they cannot be
promoted to a public origin without a real URL, downloaded media verification, and
an explicitly typed publication timestamp.
"""

from __future__ import annotations

import os
from typing import Any, List, Mapping, Optional

from src.common.logger import logger
from src.common.schemas import AccountAttribution, InternalMatch
from src.source_attribution.evidence import (
    SourceAttributionEvidence,
    candidate_is_verified,
    candidate_to_external_match,
    connector_status,
    local_context,
    normalize_candidate,
    public_url,
    redact_secrets,
    utc_now,
)
from src.source_attribution.llm import EventContextExtractor, MediaDescriptor, QueryGenerator
from src.source_attribution.reranking import MultimodalEmbedder
from src.source_attribution.reverse_search import (
    GoogleVisionClient,
    VideoKeyframeExtractor,
    VisualSearchConnector,
    YandexSearchClient,
)
from src.source_attribution.search_crawlers import (
    MediaScraper,
    RedditSearchClient,
    WebSearchClient,
    XSearchClient,
    YouTubeSearchClient,
)
from src.source_attribution.social_graph import ChronologyEngine, SocialGraphBuilder, TemporalWeightCalculator


class SourceAttributionPipeline:
    """End-to-end source tracing with explicit provenance and verification gates."""

    def __init__(self, output_dir: str = "data/graphs"):
        self.output_dir = output_dir
        self.keyframe_extractor = VideoKeyframeExtractor(output_dir=os.path.join(output_dir, "keyframes"))
        self.google_vision = GoogleVisionClient()
        self.yandex = YandexSearchClient()
        self.visual_search = VisualSearchConnector(max_results=50)
        self.web_search = WebSearchClient(max_results=50)
        self.reddit = RedditSearchClient()
        self.youtube = YouTubeSearchClient()
        self.x = XSearchClient()
        self.descriptor = MediaDescriptor()
        self.query_gen = QueryGenerator()
        self.reranker = MultimodalEmbedder()

    @staticmethod
    def _status_from_connector(
        connector: str,
        client: Any,
        *,
        result_count: int = 0,
        default_status: str = "empty",
        reason: str | None = None,
    ) -> dict[str, Any]:
        status = getattr(client, "last_status", None)
        if isinstance(status, Mapping):
            result = dict(status)
            result["source_connector"] = connector
            if result.get("status") == "not_run":
                result["status"] = default_status
            result.setdefault("result_count", result_count)
            result.setdefault("retrieved_at", utc_now())
            return result
        return connector_status(
            connector,
            default_status,
            result_count=result_count,
            reason=reason,
        )

    @staticmethod
    def _lead_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Return safe, typed fields for an unverified investigative lead."""
        url = candidate.get("post_url") or candidate.get("url")
        return {
            "url": url,
            "post_url": url,
            "platform": candidate.get("platform"),
            "account": candidate.get("account"),
            "title": candidate.get("title") or None,
            "text": candidate.get("text") or candidate.get("snippet") or None,
            "has_media": bool(candidate.get("has_media")),
            "source_connector": candidate.get("source_connector"),
            "retrieved_at": candidate.get("retrieved_at"),
            "published_at": candidate.get("published_at"),
            "timestamp_type": candidate.get("timestamp_type", "unknown"),
            "media_verification": candidate.get("media_verification", "unverified_text"),
            "evidence_status": candidate.get("evidence_status", "unverified"),
            "is_synthetic": bool(candidate.get("is_synthetic", False)),
            "failure_reason": candidate.get("failure_reason"),
        }

    async def execute(
        self,
        media_path: str,
        internal_matches: Optional[List[InternalMatch]] = None,
        investigator_context: Optional[str] = None,
        search_context: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> SourceAttributionEvidence:
        logger.info("source_attribution.pipeline.started", media_path=media_path)
        internal_matches = internal_matches or []
        connector_coverage: dict[str, dict[str, Any]] = {}
        raw_candidates: list[dict[str, Any]] = []

        # Video attribution still uses the existing first-keyframe behaviour.
        active_image_path = media_path
        if self.keyframe_extractor.is_video(media_path):
            keyframes = self.keyframe_extractor.extract_keyframes(media_path, sample_rate_sec=1.0)
            if not keyframes:
                raise ValueError("Video source attribution requires at least one decoded keyframe")
            active_image_path = keyframes[0]["frame_path"]
            logger.info("source_attribution.video_sampled", frames_count=len(keyframes))

        description = ""
        llm_succeeded = False
        try:
            model_description = await self.descriptor.describe_media(active_image_path)
            llm_succeeded = bool(getattr(self.descriptor, "llm_succeeded", False))
            if llm_succeeded and model_description:
                description = str(model_description).strip()
            connector_coverage["gemini"] = {
                "source_connector": "gemini",
                "status": "ok" if llm_succeeded else "unavailable",
                "result_count": 1 if llm_succeeded else 0,
                "retrieved_at": utc_now(),
                **({} if llm_succeeded else {"failure_reason": "visual description unavailable"}),
            }
        except Exception as exc:
            safe_error = str(redact_secrets(str(exc)))
            connector_coverage["gemini"] = connector_status("gemini", "error", reason=safe_error)
            logger.warning("source_attribution.description_failed", error=safe_error)

        # Context priority is explicit investigator context, successful Gemini
        # output, then locally observed metadata / a meaningful original name.
        investigator_value = (investigator_context or "").strip()
        search_value = (search_context or "").strip()
        supplied_context = investigator_value or search_value
        local_evidence = local_context(media_path, original_filename=original_filename)
        if investigator_value:
            context_text = supplied_context
            context_source = "investigator_context"
            context_reason = "investigator supplied context"
        elif search_value:
            context_text = supplied_context
            context_source = "search_context"
            context_reason = "search context supplied by caller"
        elif llm_succeeded and description:
            context_text = description
            context_source = "gemini"
            context_reason = "successful visual description"
        elif local_evidence:
            context_text = " ".join(local_evidence)
            context_source = "local_metadata"
            context_reason = "locally observed filename or EXIF fields"
        else:
            context_text = ""
            context_source = None
            context_reason = "no trustworthy visual, OCR, metadata, filename, URL, or investigator context"

        queries: dict[str, list[str]] = {}
        if context_text:
            try:
                queries = self.query_gen.generate_queries(context_text) or {}
                queries = {
                    str(key): [str(item).strip() for item in values if isinstance(item, str) and str(item).strip()]
                    for key, values in queries.items()
                    if isinstance(values, list)
                }
            except Exception as exc:
                safe_error = str(redact_secrets(str(exc)))
                logger.warning("source_attribution.query_generation_failed", error=safe_error)
                connector_coverage["query_generator"] = connector_status("query_generator", "error", reason=safe_error)
        if not queries:
            query_provenance = {
                "status": "insufficient_search_context" if not context_text else "no_queries_generated",
                "source": context_source,
                "reason": context_reason,
                "search_performed": False,
                "queries": {},
            }
        else:
            query_provenance = {
                "status": "generated",
                "source": context_source,
                "reason": context_reason,
                "search_performed": True,
                "queries": queries,
            }

        connector_names = {
            "google_vision": self.google_vision,
            "yandex": self.yandex,
            "reverse_visual_search": self.visual_search,
            "open_web": self.web_search,
            "reddit": self.reddit,
            "youtube": self.youtube,
            "x": self.x,
        }

        # Never fan out when there is no trustworthy context.  This is the gate
        # that prevents temporary names such as ``upload.webp`` becoming queries.
        if queries:
            try:
                values = self.google_vision.search_image(active_image_path)
                raw_candidates.extend(normalize_candidate(v, "google_vision_web_detection") for v in (values or []))
                connector_coverage["google_vision"] = self._status_from_connector(
                    "google_vision_web_detection", self.google_vision, result_count=len(values or [])
                )
            except Exception as exc:
                connector_coverage["google_vision"] = connector_status(
                    "google_vision_web_detection", "error", reason=str(redact_secrets(str(exc)))
                )

            try:
                values = self.yandex.search_image(active_image_path)
                raw_candidates.extend(normalize_candidate(v, "yandex_reverse_image") for v in (values or []))
                connector_coverage["yandex"] = self._status_from_connector(
                    "yandex_reverse_image", self.yandex, result_count=len(values or [])
                )
            except Exception as exc:
                connector_coverage["yandex"] = connector_status(
                    "yandex_reverse_image", "error", reason=str(redact_secrets(str(exc)))
                )

            reverse_queries = queries.get("reverse_image", [])[:4]
            reverse_count = 0
            for reverse_query in reverse_queries:
                try:
                    values = self.visual_search.search(reverse_query, limit=50) or []
                    reverse_count += len(values)
                    raw_candidates.extend(normalize_candidate(v, "reverse_visual_search") for v in values)
                except Exception as exc:
                    connector_coverage["reverse_visual_search"] = connector_status(
                        "reverse_visual_search", "error", reason=str(redact_secrets(str(exc)))
                    )
            if "reverse_visual_search" not in connector_coverage and not reverse_queries:
                connector_coverage["reverse_visual_search"] = connector_status(
                    "reverse_visual_search", "not_run", reason="no reverse-image query generated"
                )
            elif "reverse_visual_search" not in connector_coverage:
                connector_coverage["reverse_visual_search"] = self._status_from_connector(
                    "reverse_visual_search", self.visual_search, result_count=reverse_count
                )

            async_searches = (
                ("open_web", self.web_search, "web", "max_results"),
                ("reddit", self.reddit, "reddit", "limit"),
                ("youtube", self.youtube, "youtube", "max_results"),
                ("x", self.x, "x", "limit"),
            )
            for connector, client, query_key, limit_kw in async_searches:
                if not queries.get(query_key):
                    connector_coverage[connector] = connector_status(
                        connector, "not_run", reason="no query generated for connector"
                    )
                    continue
                connector_count = 0
                connector_error: str | None = None
                for query in queries.get(query_key, [])[:4]:
                    try:
                        if limit_kw == "max_results":
                            values = await client.search(query, max_results=50)
                        else:
                            values = await client.search(query, limit=50)
                        values = values or []
                        connector_count += len(values)
                        raw_candidates.extend(normalize_candidate(v, connector) for v in values)
                    except Exception as exc:
                        connector_error = str(redact_secrets(str(exc)))
                        logger.warning("source_attribution.connector_failed", connector=connector, error=connector_error)
                if connector_error:
                    connector_coverage[connector] = connector_status(
                        connector, "error", result_count=connector_count, reason=connector_error
                    )
                else:
                    connector_coverage[connector] = self._status_from_connector(
                        connector, client, result_count=connector_count
                    )
        else:
            for connector, client in connector_names.items():
                connector_coverage[connector] = connector_status(
                    connector,
                    "not_run",
                    reason="insufficient search context",
                )

        # Normalize and exclude reserved / synthetic records before any media
        # download.  Benchmark fixtures remain visible only as labelled leads.
        candidates = [
            normalize_candidate(candidate, str(candidate.get("source_connector") or "unknown"))
            for candidate in raw_candidates
        ]
        public_candidates = [
            candidate
            for candidate in candidates
            if public_url(candidate.get("post_url") or candidate.get("url"))
            and not candidate.get("is_synthetic")
            and candidate.get("evidence_status") not in {"fixture", "connector_error", "unavailable"}
            and candidate.get("status") not in {"fixture", "connector_error", "unavailable"}
        ]
        fixture_leads = (
            [
                self._lead_view(candidate)
                for candidate in candidates
                if candidate.get("is_synthetic") is True
            ]
            if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1"
            else []
        )

        if public_candidates:
            try:
                await MediaScraper.enrich_candidates_with_media(public_candidates)
                MediaScraper.verify_scraped_visuals(public_candidates, active_image_path, self.reranker)
            except Exception as exc:
                safe_error = str(redact_secrets(str(exc)))
                logger.warning("source_attribution.media_verification_failed", error=safe_error)
                for candidate in public_candidates:
                    if candidate.get("media_url") and candidate.get("media_verification") == "not_downloaded":
                        candidate["media_verification"] = "failed"
                        candidate["failure_reason"] = safe_error

        verified_candidates = [candidate for candidate in public_candidates if candidate_is_verified(candidate)]
        unverified_leads = fixture_leads + [
            self._lead_view(candidate)
            for candidate in public_candidates
            if not candidate_is_verified(candidate)
        ]

        # Only verified candidates are allowed to establish an event context or
        # a shortlist.  Text-only search results remain separate investigative leads.
        event_profile = EventContextExtractor.extract_context(
            visual_hits=verified_candidates,
            initial_description=description,
            require_verified=True,
        )
        event_context = event_profile.get("event_summary") if event_profile.get("event_detected") else ""

        shortlist = []
        similarity_map: dict[str, float] = {}
        if verified_candidates:
            try:
                shortlist, similarity_map = self.reranker.score_candidates(
                    media_description=description or context_text,
                    candidates=verified_candidates,
                    query_image_path=active_image_path,
                    event_context=event_context,
                    top_k=10,
                )
            except Exception as exc:
                logger.warning("source_attribution.rerank_failed", error=str(redact_secrets(str(exc))))
                shortlist = []

        # Chronology receives only exact/near-duplicate media records with an
        # explicit publication/platform/archive timestamp.
        first_50_publications, earliest_candidate, chronology_summary = ChronologyEngine.build_publication_chronology(
            candidates=verified_candidates,
            similarity_map=similarity_map,
            min_similarity=0.45,
            max_publications=50,
            event_date=event_profile.get("event_date") if event_profile.get("event_detected") else None,
        )
        external_matches = [candidate_to_external_match(candidate) for candidate in verified_candidates]
        temporal_weighting = TemporalWeightCalculator.calculate(
            earliest_candidate.timestamp if earliest_candidate else None
        )

        social_graph = SocialGraphBuilder.build_account_social_graph(shortlist)
        dissemination_graph = SocialGraphBuilder.build_dissemination_graph(
            internal_matches=internal_matches,
            external_matches=external_matches,
            publications=first_50_publications,
        )
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            SocialGraphBuilder.export_pyvis_html(
                social_graph,
                os.path.join(self.output_dir, "social_graph.html"),
                title="Account-Level Social Amplification Graph",
            )
            SocialGraphBuilder.export_pyvis_html(
                dissemination_graph,
                os.path.join(self.output_dir, "dissemination_graph.html"),
                title="Content Dissemination Timeline (Verified Media Only)",
            )
        except Exception as exc:
            logger.warning("source_attribution.graph_export_failed", error=str(redact_secrets(str(exc))))

        limitations: list[str] = []
        if not context_text:
            limitations.append("Insufficient trustworthy search context; public connectors were not searched.")
        elif not queries:
            limitations.append("No safe search queries could be generated from the available context.")
        if not verified_candidates:
            limitations.append("No verified public occurrence found.")
        elif not earliest_candidate:
            limitations.append("Verified media was observed, but no usable publication timestamp was available; earliest origin was not assigned.")
        if unverified_leads:
            limitations.append("Unverified search leads are shown separately and were excluded from origin ranking.")
        if not limitations:
            limitations.append("Origin ranking is limited to downloaded exact or near-duplicate media with typed timestamps.")

        timestamp_provenance = chronology_summary.get("timestamp_provenance", []) if chronology_summary else []
        verified_occurrences = [
            {
                "url": candidate.get("post_url") or candidate.get("url"),
                "platform": candidate.get("platform"),
                "account": candidate.get("account"),
                "title": candidate.get("title") or None,
                "text": candidate.get("text") or candidate.get("snippet") or None,
                "source_connector": candidate.get("source_connector"),
                "retrieved_at": candidate.get("retrieved_at"),
                "published_at": candidate.get("published_at"),
                "timestamp_type": candidate.get("timestamp_type", "unknown"),
                "media_verification": candidate.get("media_verification"),
                "evidence_status": candidate.get("evidence_status"),
                "is_synthetic": bool(candidate.get("is_synthetic", False)),
                "failure_reason": candidate.get("failure_reason"),
            }
            for candidate in verified_candidates
        ]

        circulation_summary = dict(chronology_summary or {})
        circulation_summary.update(
            {
                "connector_coverage": connector_coverage,
                "limitations": limitations,
                "unverified_leads": unverified_leads,
                "verified_occurrences": verified_occurrences,
                "timestamp_provenance": timestamp_provenance,
                "query_provenance": query_provenance,
                "identified_event": event_profile,
            }
        )

        payload = SourceAttributionEvidence(
            internal_matches=internal_matches,
            external_matches=external_matches,
            earliest_candidate=earliest_candidate,
            first_50_publications=first_50_publications,
            circulation_summary=circulation_summary,
            dissemination_graph=dissemination_graph,
            account_attribution=AccountAttribution(
                media_description=description,
                search_queries=queries,
                shortlist=shortlist,
                sources_not_searched=["instagram", "facebook"],
                google_trends_corroboration={
                    "checked": False,
                    "spike_aligned_with_earliest_candidate": None,
                },
                temporal_weighting=temporal_weighting,
                social_graph=social_graph,
                identified_event=event_profile,
            ),
            connector_coverage=connector_coverage,
            limitations=limitations,
            unverified_leads=unverified_leads,
            verified_occurrences=verified_occurrences,
            timestamp_provenance=timestamp_provenance,
            query_provenance=query_provenance,
        )
        return SourceAttributionEvidence.model_validate(redact_secrets(payload.model_dump(mode="json")))
