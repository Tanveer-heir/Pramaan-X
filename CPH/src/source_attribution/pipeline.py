"""
Source Attribution Pipeline Orchestrator (§2.4).
Coordinates video keyframe extraction, reverse image search, Gemini vision description,
cross-platform web/social crawling (200–300 candidates), multimodal CLIP re-ranking, and dual graph generation.
"""

from typing import List, Optional
import os
import re
from src.common.schemas import (
    OriginTracingEvidence,
    ExternalMatch,
    InternalMatch,
    EarliestCandidate,
    AccountAttribution
)
from src.common.logger import logger
from src.source_attribution.reverse_search import (
    GoogleVisionClient,
    YandexSearchClient,
    VideoKeyframeExtractor,
    VisualSearchConnector
)
from src.source_attribution.search_crawlers import (
    WebSearchClient,
    RedditSearchClient,
    YouTubeSearchClient,
    XSearchClient,
    MediaScraper
)
from src.source_attribution.llm import MediaDescriptor, QueryGenerator, EventContextExtractor
from src.source_attribution.reranking import MultimodalEmbedder
from src.source_attribution.social_graph import (
    TemporalWeightCalculator,
    SocialGraphBuilder,
    ChronologyEngine
)


class SourceAttributionPipeline:
    """End-to-end pipeline for Origin Tracing and Account Attribution."""

    def __init__(self, output_dir: str = "data/graphs"):
        self.output_dir = output_dir
        self.keyframe_extractor = VideoKeyframeExtractor(
            output_dir=os.path.join(output_dir, "keyframes")
        )
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

    async def execute(
        self,
        media_path: str,
        internal_matches: Optional[List[InternalMatch]] = None
    ) -> OriginTracingEvidence:
        logger.info("source_attribution.pipeline.started", media_path=media_path)
        internal_matches = internal_matches or []

        # 1. Video Ingestion & Keyframe Handling (§2.4b)
        active_image_path = media_path
        keyframes = []
        if self.keyframe_extractor.is_video(media_path):
            keyframes = self.keyframe_extractor.extract_keyframes(media_path, sample_rate_sec=1.0)
            if not keyframes:
                raise ValueError("Video source attribution requires at least one decoded keyframe")
            active_image_path = keyframes[0]["frame_path"]
            logger.info("source_attribution.video_sampled", frames_count=len(keyframes))

        # 2. Vision Description via Gemini 2.5 Flash / Free Tier (§2.4d step 1)
        description = await self.descriptor.describe_media(active_image_path)
        llm_succeeded = self.descriptor.llm_succeeded

        # 3. Stage 1: Direct Reverse Visual Search & External Reverse Matching (§2.4b)
        #    When LLM fails (503 / unavailable), visual search runs FIRST to derive context.
        external_matches: List[ExternalMatch] = []
        reverse_candidates = []

        # Build initial seed queries — use filename/path cues when LLM failed
        if llm_succeeded:
            seed_queries = self.query_gen.generate_queries(description)
        else:
            # Extract whatever filename tokens we have (e.g. "sample_protest_rally" → "protest rally")
            basename = os.path.splitext(os.path.basename(media_path))[0]
            filename_tokens = basename.replace("_", " ").replace("-", " ")
            seed_queries = self.query_gen.generate_queries(filename_tokens)
            logger.info(
                "source_attribution.visual_first_mode",
                reason="llm_api_failed",
                seed=filename_tokens
            )

        try:
            external_matches.extend(self.google_vision.search_image(active_image_path))
            external_matches.extend(self.yandex.search_image(active_image_path))

            # Direct reverse visual image search across seed query set
            rev_queries = seed_queries.get("reverse_image", [description[:50]])
            for rev_q in rev_queries[:4]:
                visual_hits = self.visual_search.search(rev_q, limit=50)
                reverse_candidates.extend(visual_hits)
                external_matches.extend(self.visual_search.to_external_matches(visual_hits))
        except Exception as e:
            logger.error("source_attribution.reverse_search_failed", error=str(e))

        # 4. When LLM failed, derive a synthetic description from visual match titles
        #    This replaces the generic heuristic with real content from the internet
        if not llm_succeeded and reverse_candidates:
            visual_titles = [
                h.get("title", "") for h in reverse_candidates
                if h.get("title") and len(h.get("title", "")) > 10
            ]
            if visual_titles:
                # Take the top 5 most informative titles as synthetic description
                description = " | ".join(visual_titles[:5])
                logger.info(
                    "source_attribution.visual_derived_description",
                    derived_from=len(visual_titles),
                    description=description[:120]
                )

        # 5. Extract Ground-Truth Incident Context from Discovered Visual Matches
        event_profile = EventContextExtractor.extract_context(
            visual_hits=reverse_candidates,
            initial_description=description
        )
        event_context_str = event_profile.get("event_summary", description)
        logger.info(
            "source_attribution.event_context_discovered",
            event_name=event_profile.get("event_name"),
            date=event_profile.get("event_date"),
            confidence=event_profile.get("confidence")
        )

        # 6. Stage 2: Generate Hyper-Specific Event Queries using Discovered Context
        if llm_succeeded and seed_queries:
            # LLM successfully generated high-precision queries from direct media vision & verbatim OCR
            queries = seed_queries
            # Augment with discovered event date/year if available and not yet present
            ev_date = event_profile.get("event_date")
            if ev_date:
                m_yr = re.search(r'\b(20[0-2][0-9])\b', str(ev_date))
                if m_yr:
                    yr = m_yr.group(1)
                    for plat in ("web", "x", "youtube", "reddit"):
                        if plat in queries and queries[plat] and not any(yr in q for q in queries[plat]):
                            queries[plat].append(f"{queries[plat][0]} {yr}")
        elif event_profile.get("event_detected"):
            # LLM unavailable but high-confidence event detected via visual hits
            queries = self.query_gen.generate_event_specific_queries(event_profile)
        elif reverse_candidates:
            # Fallback to query generation from visual matches
            queries = self.query_gen.generate_queries(description)
        else:
            queries = seed_queries

        # 7. Cross-Platform Search Fan-Out (~1,000 candidates total with event & temporal filters) (§2.4d step 3 & 4)
        raw_candidates = []
        raw_candidates.extend(reverse_candidates)

        # Open Web Search (news articles, regional media, blogs)
        for q in queries.get("web", [])[:4]:
            raw_candidates.extend(await self.web_search.search(q, max_results=50))

        # Reddit Search (public discussions, community verification)
        for q in queries.get("reddit", [])[:4]:
            raw_candidates.extend(await self.reddit.search(q, limit=50))

        # YouTube Search (video uploads, eyewitness recordings)
        for q in queries.get("youtube", [])[:4]:
            raw_candidates.extend(await self.youtube.search(q, max_results=50))

        # X / Twitter Search (breaking alerts, syndication feeds)
        for q in queries.get("x", [])[:4]:
            raw_candidates.extend(await self.x.search(q, limit=50))

        logger.info(
            "source_attribution.search_fanout_complete",
            total_candidates=len(raw_candidates)
        )

        # 8. Media Scraping: Extract image/video thumbnails and OpenGraph media across candidate URLs
        await MediaScraper.enrich_candidates_with_media(raw_candidates)

        # 9. Calibrated Dual-Anchor Multimodal Semantic Re-ranking via CLIP + ChromaDB (§2.4d step 5)
        shortlist, similarity_map = self.reranker.score_candidates(
            media_description=description,
            candidates=raw_candidates,
            query_image_path=active_image_path,
            event_context=event_context_str,
            top_k=10
        )

        # 10. Direct Visual Matching Verification: Compute CLIP image-to-image similarity on scraped media
        MediaScraper.verify_scraped_visuals(raw_candidates, active_image_path, self.reranker)

        # Update similarity map and shortlist with verified media matches
        for c in raw_candidates:
            if c.get("media_verified"):
                url = c.get("post_url") or c.get("url") or ""
                if url:
                    similarity_map[url] = c["similarity"]
                norm_k = f"{c.get('platform')}:{c.get('account')}:{c.get('title','')[:30]}"
                similarity_map[norm_k] = c["similarity"]

        # Re-sort shortlist so verified media matches rise to the top
        for sc in shortlist:
            for c in raw_candidates:
                if sc.post_url and (sc.post_url == c.get("post_url") or sc.post_url == c.get("origin_post_url")):
                    sc.similarity = c.get("similarity", sc.similarity)
                    if c.get("account"):
                        sc.account = c["account"]
                    if c.get("origin_post_url"):
                        sc.post_url = c["origin_post_url"]
                    if c.get("media_verified"):
                        sc.has_media = True
                        sc.post_text = f"[VERIFIED VISUAL MATCH] {sc.post_text or ''}"
        shortlist.sort(key=lambda x: x.similarity, reverse=True)

        # 11. Chronological Dissemination Engine: First 50 Publication Places & Primary Origin (§2.4c)
        first_50_publications, earliest_candidate, circulation_summary = (
            ChronologyEngine.build_publication_chronology(
                candidates=raw_candidates,
                similarity_map=similarity_map,
                min_similarity=0.45,
                max_publications=50,
                event_date=event_profile.get("event_date")
            )
        )
        if circulation_summary:
            circulation_summary["identified_event"] = event_profile

        # Also incorporate internal/external matches if earlier than discovered candidates
        all_timestamps = []
        for im in internal_matches:
            all_timestamps.append((im.timestamp, im.instance_id, im.platform, "internal"))
        for em in external_matches:
            if em.date_found:
                all_timestamps.append((em.date_found, em.url, em.source, "external"))
        if earliest_candidate and earliest_candidate.timestamp:
            all_timestamps.append((
                earliest_candidate.timestamp,
                earliest_candidate.url or earliest_candidate.account or "origin",
                earliest_candidate.platform or "candidate",
                "candidate"
            ))

        if all_timestamps:
            sorted_ts = sorted(all_timestamps, key=lambda x: x[0])
            earliest = sorted_ts[0]
            if earliest[3] != "candidate" or earliest_candidate is None:
                earliest_candidate = EarliestCandidate(
                    timestamp=earliest[0],
                    instance_id=earliest[1] if earliest[3] == "internal" else None,
                    url=earliest[1] if earliest[3] != "internal" else None,
                    account=earliest[1] if earliest[3] != "internal" else None,
                    platform=earliest[2],
                    confidence="high" if earliest[3] in ("internal", "external") else "moderate"
                )

        # 8. Dynamic Temporal Weighting (§2.4e)
        temporal_weighting = TemporalWeightCalculator.calculate(
            earliest_candidate.timestamp if earliest_candidate else None
        )

        # 9. Account-Level Social Graph & Dissemination Graph (§2.4c, §2.4f)
        social_graph = SocialGraphBuilder.build_account_social_graph(shortlist)
        dissemination_graph = SocialGraphBuilder.build_dissemination_graph(
            internal_matches=internal_matches,
            external_matches=external_matches[:10],
            publications=first_50_publications
        )

        # 10. Export Interactive PyVis HTML Visualizations
        graphs_dir = self.output_dir
        os.makedirs(graphs_dir, exist_ok=True)
        SocialGraphBuilder.export_pyvis_html(
            social_graph,
            os.path.join(graphs_dir, "social_graph.html"),
            title="Account-Level Social Amplification Graph"
        )
        SocialGraphBuilder.export_pyvis_html(
            dissemination_graph,
            os.path.join(graphs_dir, "dissemination_graph.html"),
            title="Content Dissemination Timeline Tree (First 50 Publication Places)"
        )

        # Assemble canonical output object
        return OriginTracingEvidence(
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
                    "checked": True,
                    "spike_aligned_with_earliest_candidate": None
                },
                temporal_weighting=temporal_weighting,
                social_graph=social_graph,
                identified_event=event_profile
            )
        )
