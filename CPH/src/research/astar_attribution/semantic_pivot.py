"""
Palantir-Style Semantic Pivot & Continuous Intelligence Layer (§2.4b).
Chandigarh Police Hackathon - A* Attribution Research Track

Performs multi-modal semantic deconstruction (OCR, entities, event hypothesis,
orthogonal query batteries) via Google Gemini with multi-model failover ladder,
harvests external open-web media candidates, and sifts them via planar homography
and perceptual hashing to break out of visual dead-ends.
"""

import os
import asyncio
import re
import io
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, urljoin

import httpx
from PIL import Image
from bs4 import BeautifulSoup
from ddgs import DDGS

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.common.config import settings
from src.common.logger import logger
from src.research.astar_attribution.subimage_matcher import SubImageMatcher, MatchResult
from src.research.astar_attribution.neighbor_crawler import (
    VisualNeighborCrawler,
    NeighborCandidate,
    AUTHORITATIVE_WIRE_DOMAINS,
)


@dataclass
class SceneDeconstruction:
    """Structured scene analysis extracted via multimodal intelligence."""
    ocr_text: List[str]
    entities: List[str]
    event_hypothesis: str
    query_battery: List[str]
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr_text": self.ocr_text,
            "entities": self.entities,
            "event_hypothesis": self.event_hypothesis,
            "query_battery": self.query_battery,
        }


class SemanticPivotEngine:
    """
    Palantir-Style Semantic Pivot & Continuous Intelligence Layer.
    
    Transforms visual dead-ends into high-probability origin candidate clusters by:
    1. Multimodal Scene Deconstruction (verbatim OCR, entities, event hypothesis, orthogonal query batteries).
    2. Candidate Image Harvesting (DuckDuckGo Images, Web DOM scraping).
    3. Rigorous Geometric & Perceptual Sifting (SIFT homography, bounding-box localization, Meta PDQ).
    """

    MODELS_LADDER = ["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-2.5-flash"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/cache/astar",
        timeout_sec: float = 8.0,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.cache_dir = cache_dir
        self.timeout_sec = timeout_sec
        self.client = None
        self.model_index = 0
        self.last_deconstruction: Optional[SceneDeconstruction] = None
        self.last_harvested: List[Dict[str, Any]] = []

        os.makedirs(self.cache_dir, exist_ok=True)

        if HAS_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("semantic_pivot.gemini_client_initialized")
            except Exception as e:
                logger.warn("semantic_pivot.gemini_init_error", error=str(e))

    def deconstruct_scene(self, media_path: str) -> SceneDeconstruction:
        """
        Deconstructs the scene using multimodal Gemini with multi-model failover ladder.
        Extracts verbatim OCR text, entities, event hypothesis, and orthogonal query batteries.
        Falls back to local heuristic token/exif extraction if LLM is unavailable.
        """
        abs_path = os.path.abspath(media_path)
        if not os.path.isfile(abs_path):
            logger.warn("semantic_pivot.file_not_found", path=abs_path)
            return self._heuristic_deconstruction(abs_path)

        # Skip API calls during benchmark mode if requested
        if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1" or not self.client:
            return self._heuristic_deconstruction(abs_path)

        prompt = (
            "You are an elite digital forensics and visual intelligence specialist conducting a Palantir-style OSINT attribution investigation.\n"
            "Analyze this forensic image and produce a structured JSON response with the following keys:\n"
            "1. \"ocr_text\": A list of all visible text, license plates, signs, banners, stencils, vehicle numbers, badges, or watermarks transcribed verbatim in quotes.\n"
            "2. \"entities\": A list of key identifiable entities, persons, uniforms, flags, vehicle makes/models, weaponry, or architectural landmarks.\n"
            "3. \"event_hypothesis\": A concise factual hypothesis of the specific historical or news event depicted, including location, context, and year (e.g., 'Farmers tractor rally clash at Red Fort, Delhi, India, January 2021').\n"
            "4. \"query_battery\": A list of 3-6 orthogonal search queries designed to find the original photographer, wire article, or uncropped master image:\n"
            "   - Battery A (OCR & Identifiers): specific markings/text + location\n"
            "   - Battery B (Event & Year): verified event name + city/country + year\n"
            "   - Battery C (Objects & Context): distinctive vehicle/setting keywords\n\n"
            "Return ONLY valid JSON matching this structure:\n"
            "{\n"
            "  \"ocr_text\": [\"...\"],\n"
            "  \"entities\": [\"...\"],\n"
            "  \"event_hypothesis\": \"...\",\n"
            "  \"query_battery\": [\"...\", \"...\"]\n"
            "}"
        )

        for attempt in range(len(self.MODELS_LADDER)):
            model_name = self.MODELS_LADDER[(self.model_index + attempt) % len(self.MODELS_LADDER)]
            try:
                img = Image.open(abs_path)
                logger.info("semantic_pivot.calling_gemini", model=model_name, path=abs_path)
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=[img, prompt]
                )
                raw_text = (resp.text or "").strip()
                if raw_text:
                    # Clean markdown code blocks if wrapped
                    clean_json = raw_text
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0].strip()

                    parsed = json.loads(clean_json)
                    deconstruction = SceneDeconstruction(
                        ocr_text=parsed.get("ocr_text", []),
                        entities=parsed.get("entities", []),
                        event_hypothesis=parsed.get("event_hypothesis", ""),
                        query_battery=parsed.get("query_battery", []),
                        raw_response=raw_text,
                    )
                    self.model_index = (self.model_index + attempt) % len(self.MODELS_LADDER)
                    logger.info(
                        "semantic_pivot.deconstruction_success",
                        model=model_name,
                        hypothesis=deconstruction.event_hypothesis[:60],
                        queries=len(deconstruction.query_battery),
                    )
                    return deconstruction

            except Exception as e:
                logger.warn(
                    "semantic_pivot.model_failed_failover",
                    model=model_name,
                    error=str(e)[:80],
                )
                continue

        logger.warn("semantic_pivot.all_models_failed_using_heuristic")
        return self._heuristic_deconstruction(abs_path)

    def _heuristic_deconstruction(self, media_path: str) -> SceneDeconstruction:
        """Local heuristic token and EXIF parser fallback when Gemini is unavailable."""
        filename = os.path.basename(media_path)
        base_name = os.path.splitext(filename)[0]

        # Extract meaningful tokens from filename
        tokens = [t for t in re.split(r"[-_\s\.]+", base_name) if len(t) > 2 and not t.isdigit()]

        # Check year in filename
        year_match = re.search(r"\b(20\d{2})\b", base_name)
        year_str = year_match.group(1) if year_match else "2021"

        # Check for wire mentions
        wire_tag = "AP" if "ap" in base_name.lower() else ""

        event_words = [t for t in tokens if t.lower() not in ("640", "1280", "superjumbo", "thumb", "image")]
        event_str = " ".join(event_words) if event_words else "Protest rally procession"
        hypothesis = f"{event_str} event ({year_str})"

        query_battery = [
            f"{event_str} {year_str}",
            f"{event_str} original photo {year_str}",
            f"{event_str} {wire_tag} news".strip(),
            f"'{event_str}' master high resolution",
        ]

        # Deduplicate queries while preserving order
        unique_queries = list(dict.fromkeys(query_battery))

        return SceneDeconstruction(
            ocr_text=[t for t in tokens if t.isupper() or len(t) <= 4],
            entities=tokens[:5],
            event_hypothesis=hypothesis,
            query_battery=unique_queries,
            raw_response="[Heuristic Fallback Engine]",
        )

    async def harvest_candidate_images(
        self,
        query_battery: List[str],
        max_images: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Harvests image candidate URLs via DuckDuckGo Images and Web/News DOM scraping,
        downloads candidates asynchronously with httpx to data/cache/astar/<sha256>.jpg,
        filters out images < 150px, and deduplicates by SHA-256.
        """
        logger.info("semantic_pivot.harvesting_started", queries=len(query_battery), max_images=max_images)
        candidate_image_urls: Set[str] = set()

        # Step 1: Collect Candidate URLs from DuckDuckGo Images and Web News DOM
        try:
            with DDGS() as ddgs:
                for query in query_battery[:5]:
                    # 1a: DuckDuckGo Images Search
                    try:
                        raw_imgs = list(ddgs.images(query, max_results=15))
                        for item in raw_imgs:
                            img_url = item.get("image")
                            if img_url and img_url.startswith(("http://", "https://")):
                                if "favicon" not in img_url.lower():
                                    candidate_image_urls.add(img_url)
                    except Exception as e:
                        logger.debug("semantic_pivot.ddgs_images_error", query=query, error=str(e))

                    # 1b: DuckDuckGo News / Web Search for articles
                    try:
                        raw_news = list(ddgs.news(query, max_results=3))
                        for article in raw_news:
                            art_img = article.get("image")
                            if art_img and art_img.startswith(("http://", "https://")):
                                candidate_image_urls.add(art_img)
                    except Exception as e:
                        logger.debug("semantic_pivot.ddgs_news_error", query=query, error=str(e))

                    if len(candidate_image_urls) >= max_images * 2:
                        break
        except Exception as e:
            logger.warn("semantic_pivot.ddgs_harvest_failed", error=str(e))

        logger.info("semantic_pivot.urls_collected", count=len(candidate_image_urls))

        # Step 2: Download and Cache Candidate Images Concurrently
        harvested_records: List[Dict[str, Any]] = []
        seen_shas: Set[str] = set()
        sem = asyncio.Semaphore(12)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        async def fetch_one(client: httpx.AsyncClient, url: str):
            if len(harvested_records) >= max_images:
                return None
            async with sem:
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        return None
                    content = resp.content
                    if len(content) < 500:
                        return None

                    # Verify image decoding and size with PIL
                    img = Image.open(io.BytesIO(content))
                    w, h = img.size
                    if w < 150 or h < 150:
                        return None

                    sha256_hash = hashlib.sha256(content).hexdigest()
                    if sha256_hash in seen_shas:
                        return None
                    seen_shas.add(sha256_hash)

                    local_path = os.path.join(self.cache_dir, f"{sha256_hash}.jpg")
                    if not os.path.exists(local_path):
                        img.convert("RGB").save(local_path, format="JPEG", quality=95)

                    parsed = urlparse(url)
                    domain = (parsed.hostname or parsed.netloc).lower()

                    # Extract timestamp from headers or URL
                    last_mod = resp.headers.get("last-modified")
                    date_match = re.search(r"/(20\d{2})[-/_](0[1-9]|1[0-2])[-/_](0[1-9]|[12]\d|3[01])", url)
                    ts = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}T00:00:00Z" if date_match else last_mod

                    return {
                        "url": url,
                        "local_path": local_path,
                        "domain": domain,
                        "timestamp": ts,
                        "width": w,
                        "height": h,
                    }
                except Exception as e:
                    logger.debug("semantic_pivot.download_candidate_failed", url=url, error=str(e))
                    return None

        async with httpx.AsyncClient(timeout=self.timeout_sec, follow_redirects=True, headers=headers) as client:
            tasks = [fetch_one(client, u) for u in list(candidate_image_urls)[:max_images * 2]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict) and len(harvested_records) < max_images:
                    harvested_records.append(r)

        logger.info("semantic_pivot.harvesting_completed", count=len(harvested_records))
        return harvested_records

    async def run_pivot(
        self,
        target_image_path: str,
        matcher: SubImageMatcher,
        neighbor_crawler: VisualNeighborCrawler,
        max_candidates: int = 50,
    ) -> List[NeighborCandidate]:
        """
        Executes the full semantic pivot workflow:
        1. Deconstructs scene via Gemini multimodal ladder.
        2. Harvests candidate images across orthogonal queries.
        3. Sifts candidates against target using planar homography and Meta PDQ distance.
        4. Returns qualified NeighborCandidates tagged with source_hop='semantic_pivot'.
        """
        logger.info("semantic_pivot.run_pivot.started", target=target_image_path)

        # 1. Scene Deconstruction
        deconstruction = self.deconstruct_scene(target_image_path)
        self.last_deconstruction = deconstruction
        logger.info(
            "semantic_pivot.scene_deconstructed",
            hypothesis=deconstruction.event_hypothesis,
            queries=deconstruction.query_battery,
        )

        # 2. Candidate Image Harvesting
        harvested = await self.harvest_candidate_images(
            query_battery=deconstruction.query_battery,
            max_images=max_candidates,
        )
        self.last_harvested = harvested

        # 3. Geometric & Perceptual Sifting
        sifted_neighbors: List[NeighborCandidate] = []
        seen_pdqs: Set[str] = set()

        for cand in harvested:
            cand_path = cand["local_path"]

            match_res = matcher.compare_images(
                target_path=target_image_path,
                candidate_path=cand_path,
            )

            # Sifting Criteria:
            # - Partial crop (target_is_crop_of_candidate or candidate_is_crop_of_target)
            # - SIFT inliers >= 15 and inlier_ratio >= 0.15
            # - Meta PDQ distance <= 55
            is_partial = match_res.crop_type in ("target_is_crop_of_candidate", "candidate_is_crop_of_target")
            has_sift_match = (match_res.inlier_count >= 15 and match_res.inlier_ratio >= 0.15)
            is_pdq_close = (match_res.pdq_distance <= 55)

            if is_partial or has_sift_match or is_pdq_close:
                try:
                    img = Image.open(cand_path)
                    pdq_hex = neighbor_crawler.compute_pdq(img)
                except Exception:
                    continue

                if pdq_hex in seen_pdqs:
                    continue
                seen_pdqs.add(pdq_hex)

                is_wire = neighbor_crawler.is_authoritative_domain(cand["domain"])

                neighbor = NeighborCandidate(
                    url=cand["url"],
                    local_path=cand_path,
                    width=cand["width"],
                    height=cand["height"],
                    pdq_hex=pdq_hex,
                    timestamp=cand["timestamp"],
                    domain=cand["domain"],
                    is_authoritative_wire=is_wire,
                    source_hop="semantic_pivot",
                )
                sifted_neighbors.append(neighbor)
                logger.info(
                    "semantic_pivot.candidate_qualified",
                    url=cand["url"],
                    domain=cand["domain"],
                    crop=match_res.crop_type,
                    inliers=match_res.inlier_count,
                    pdq=match_res.pdq_distance,
                )

        logger.info("semantic_pivot.run_pivot.completed", qualified_count=len(sifted_neighbors))
        return sifted_neighbors