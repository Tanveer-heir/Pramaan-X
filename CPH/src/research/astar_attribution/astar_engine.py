"""
A* Heuristic Traversal Engine for Recursive Visual Media Attribution.
Chandigarh Police Hackathon - A* Attribution Research Track (§2.4b)

Traverses reverse visual search graphs using an A* priority queue to isolate
the original 'Patient Zero' master image, resolving crops, resizes, and syndication.
"""

import os
import sys
import math
import time
import heapq
import hashlib
import itertools
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse

from PIL import Image
import numpy as np

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

from src.common.logger import logger
from src.research.astar_attribution.subimage_matcher import SubImageMatcher, MatchResult
from src.research.astar_attribution.neighbor_crawler import (
    VisualNeighborCrawler,
    NeighborCandidate,
    AUTHORITATIVE_WIRE_DOMAINS,
)
from src.research.astar_attribution.semantic_pivot import SemanticPivotEngine, SceneDeconstruction


class ClosedSet:
    """
    Tracks visited URLs and (domain, PDQ) pairs to prevent redundant crawl cycles
    while allowing different domains to be compared for provenance seniority.
    """

    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.visited_domain_hashes: Set[Tuple[str, str]] = set()

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalizes URL by stripping query parameters and trailing slashes."""
        if not url:
            return ""
        parsed = urlparse(url)
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/").lower()
        return clean

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extracts base domain from URL for domain-scoped deduplication."""
        if not url:
            return ""
        try:
            return urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def add(self, url: str, pdq_hex: Optional[str] = None):
        """Adds URL and (domain, PDQ) pair to closed set."""
        if url:
            clean_url = self.normalize_url(url)
            self.visited_urls.add(clean_url)
            if pdq_hex:
                dom = self.extract_domain(url)
                self.visited_domain_hashes.add((dom, pdq_hex.lower().strip()))

    def contains(self, url: str, pdq_hex: Optional[str] = None) -> bool:
        """
        Checks if URL has been visited, or if identical image was already processed on the SAME domain.
        Allows identical images on DIFFERENT domains to proceed for provenance evaluation.
        """
        if not url:
            return False
        clean_url = self.normalize_url(url)
        if clean_url in self.visited_urls:
            return True
        if pdq_hex:
            dom = self.extract_domain(url)
            if (dom, pdq_hex.lower().strip()) in self.visited_domain_hashes:
                return True
        return False


@dataclass
class LedgerEntry:
    """Records match and crop localization details for forensic audit."""
    url: str
    local_path: str
    timestamp: Optional[str]
    resolution: Tuple[int, int]  # (width, height)
    crop_type: str  # 'exact', 'target_is_crop_of_candidate', 'candidate_is_crop_of_target'
    bounding_box: Optional[List[int]]  # [x, y, w, h]
    scale_factor: float
    inlier_count: int
    inlier_ratio: float
    pdq_distance: int
    similarity_score: float = 0.0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "local_path": self.local_path,
            "timestamp": self.timestamp,
            "resolution": list(self.resolution),
            "crop_type": self.crop_type,
            "bounding_box": self.bounding_box,
            "scale_factor": round(self.scale_factor, 4),
            "inlier_count": self.inlier_count,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "pdq_distance": self.pdq_distance,
            "similarity_score": round(self.similarity_score, 4),
            "details": self.details,
        }


class PartialMatchLedger:
    """Maintains an audit ledger of all matching media and localized bounding boxes."""

    def __init__(self):
        self.entries: List[LedgerEntry] = []

    def record_match(
        self,
        candidate: NeighborCandidate,
        match: MatchResult,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            url=candidate.url,
            local_path=candidate.local_path,
            timestamp=candidate.timestamp,
            resolution=(candidate.width, candidate.height),
            crop_type=match.crop_type,
            bounding_box=match.bounding_box,
            scale_factor=match.scale_factor,
            inlier_count=match.inlier_count,
            inlier_ratio=match.inlier_ratio,
            pdq_distance=match.pdq_distance,
            similarity_score=match.similarity_score,
            details=match.details,
        )
        self.entries.append(entry)
        return entry

    def get_entries(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]


@dataclass
class AStarNode:
    """Node in the A* traversal search frontier."""
    node_id: str
    candidate: NeighborCandidate
    parent_id: Optional[str]
    depth: int
    g_score: float
    h_score: float
    f_score: float
    match_result: Optional[MatchResult] = None
    role: str = "frontier"  # 'target', 'patient_zero', 'uncropped_parent', 'derivative_crop', 'exact_clone', 'frontier'


class AStarVisualAttributionEngine:
    """
    A* Heuristic Traversal Engine for recursive visual media attribution.
    
    Explores visual graphs guided by an objective cost function:
        f(n) = g(n) + h(n)
    where:
        g(n): Visual transformation penalty (crop alignment, homography, PDQ distance).
        h(n): Origin distance heuristic (temporal seniority, master resolution, wire authority).
    """

    def __init__(
        self,
        subimage_matcher: Optional[SubImageMatcher] = None,
        neighbor_crawler: Optional[VisualNeighborCrawler] = None,
        semantic_pivot: Optional[SemanticPivotEngine] = None,
        max_iterations: int = 12,
        max_depth: int = 3,
        time_budget_sec: float = 180.0,
        output_graph_path: str = "data/graphs/astar_lineage_tree.html",
    ):
        self.subimage_matcher = subimage_matcher or SubImageMatcher()
        self.neighbor_crawler = neighbor_crawler or VisualNeighborCrawler()
        self.semantic_pivot = semantic_pivot or SemanticPivotEngine()
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.time_budget_sec = time_budget_sec
        self.output_graph_path = output_graph_path

    @staticmethod
    def parse_iso_datetime(ts: Optional[str]) -> Optional[datetime]:
        """Safely parses ISO timestamp strings to aware datetime objects."""
        if not ts:
            return None
        try:
            # Replace Z with UTC offset
            clean_ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def compute_scores(
        self,
        candidate: NeighborCandidate,
        match: MatchResult,
        target_candidate: NeighborCandidate,
        depth: int,
    ) -> Tuple[float, float, float]:
        """
        Computes f(n) = g(n) + h(n):
        
        g(n): Visual distance & crop alignment penalty.
          - Exact match (PDQ <= 30 or full frame homography): near 0.
          - target_is_crop_of_candidate: low penalty (proves candidate is parent master!).
          - candidate_is_crop_of_target: medium penalty (candidate is child derivative).
          - Unrelated images: high penalty.
          
        h(n): Origin distance estimate.
          - Temporal: older timestamps get lower cost.
          - Resolution: candidate area > target area gives negative cost bonus.
          - Domain wire authority: AP, Reuters, Getty, NYT gives cost bonus.
        """
        # -------------------------------------------------------------
        # 1. Compute g(n): Visual Distance & Crop Alignment Penalty
        # -------------------------------------------------------------
        if match.crop_type == "exact" or match.pdq_distance <= 30:
            # Exact or near-identical copy: minimal penalty
            g = (match.pdq_distance / 10.0) + (depth * 0.5)
        elif match.crop_type == "target_is_crop_of_candidate":
            # Target is a crop of candidate: Candidate is the larger parent master!
            # Low penalty reward, scaling slightly with inlier ratio
            g = 5.0 + (1.0 - min(1.0, match.inlier_ratio)) * 5.0 + (depth * 0.5)
        elif match.crop_type == "candidate_is_crop_of_target":
            # Candidate is a crop of target: Candidate is a child derivative
            g = 45.0 + (depth * 2.0)
        elif match.is_match:
            # Planar alignment detected
            g = 12.0 + (match.pdq_distance / 20.0) + (depth * 1.0)
        else:
            # Unrelated image / no match
            g = 150.0 + min(50.0, float(match.pdq_distance))

        # -------------------------------------------------------------
        # 2. Compute h(n): Origin Distance Heuristic
        # -------------------------------------------------------------
        base_h = 45.0

        # a) Temporal Seniority
        dt_cand = self.parse_iso_datetime(candidate.timestamp)
        dt_target = self.parse_iso_datetime(target_candidate.timestamp)

        temporal_bonus = 0.0
        if dt_cand and dt_target:
            diff_days = (dt_target - dt_cand).total_seconds() / 86400.0
            if diff_days > 0:
                temporal_bonus = -min(12.0, 3.0 + math.log2(max(1.0, diff_days)) * 1.0)
            elif diff_days < 0:
                temporal_bonus = min(15.0, 3.0 + (abs(diff_days) / 60.0) * 1.5)
        elif dt_cand:
            year_diff = max(0.0, 2026.0 - dt_cand.year)
            temporal_bonus = -min(10.0, year_diff * 1.5)

        # Monotonic seniority: earlier dates (e.g. Jan 2021 vs June 2021) get strictly lower cost
        seniority_bonus = 0.0
        if dt_cand:
            ref_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
            cand_age_days = (ref_dt - dt_cand).total_seconds() / 86400.0
            seniority_bonus = -min(8.0, (cand_age_days / 365.0) * 1.5)

        # b) Resolution Advantage
        cand_area = float(candidate.width * candidate.height)
        target_area = float(target_candidate.width * target_candidate.height)

        resolution_bonus = 0.0
        if cand_area > target_area and target_area > 0:
            ratio = cand_area / target_area
            resolution_bonus = -min(12.0, 4.0 * math.log2(max(1.0, ratio)))
        elif cand_area < target_area and cand_area > 0:
            ratio = target_area / cand_area
            resolution_bonus = min(10.0, 3.0 * math.log2(max(1.0, ratio)))

        # c) Domain Wire Authority
        wire_bonus = 0.0
        d_lower = candidate.domain.lower()
        if candidate.is_authoritative_wire:
            wire_bonus = -18.0
        elif any(w in d_lower for w in ["apnews", "reuters", "getty", "afp", "pti", "nyt", "nytimes", "ap.org", "bloomberg", "bwbx", "abcnews", "politico", "wsj", "washingtonpost"]):
            wire_bonus = -18.0
        elif any(w in d_lower for w in ["bbc", "thehindu", "indianexpress", "tribuneindia", "firstpost", "cnn"]):
            wire_bonus = -8.0

        h = max(0.0, base_h + temporal_bonus + seniority_bonus + resolution_bonus + wire_bonus)
        f = g + h
        return round(g, 3), round(h, 3), round(f, 3)

    def evaluate_origin_confidence(
        self,
        candidate: NeighborCandidate,
        match: MatchResult,
        target_candidate: NeighborCandidate,
        earliest_dt: Optional[datetime] = None,
    ) -> float:
        """
        Evaluates the probability (0.0 to 1.0) that a candidate is the primary origin (Patient Zero).
        Requires wire authority or primary social verification for high confidence (>0.85).
        """
        if not match.is_match and match.pdq_distance > 30:
            return 0.0

        score = 0.0

        # 1. Master crop alignment: candidate has more content than target
        if match.crop_type == "target_is_crop_of_candidate":
            score += 0.35
        elif match.crop_type == "exact" or match.pdq_distance <= 30:
            score += 0.20
        elif match.is_match and match.inlier_count >= 18:
            score += 0.15
        elif match.crop_type == "candidate_is_crop_of_target":
            score -= 0.20

        # Inlier quality
        if match.inlier_count >= 1000:
            score += 0.10
        elif match.inlier_count >= 100:
            score += 0.05
        elif match.inlier_count < 20 and match.pdq_distance > 30:
            score -= 0.10

        # 2. Resolution seniority (rewards uncompressed high-res master)
        cand_area = candidate.width * candidate.height
        target_area = target_candidate.width * target_candidate.height
        if cand_area >= 3.5 * target_area:
            score += 0.20
        elif cand_area >= 1.8 * target_area:
            score += 0.12
        elif cand_area > target_area:
            score += 0.06
        elif cand_area < 0.5 * target_area and match.pdq_distance > 10:
            score -= 0.10

        # 3. Wire authority & institutional publisher
        cand_url_lower = candidate.url.lower()
        d_lower = candidate.domain.lower()
        is_wire = (
            candidate.is_authoritative_wire
            or any(w in d_lower for w in ["apnews", "reuters", "getty", "afp", "pti", "nyt", "nytimes", "ap.org", "bloomberg", "bwbx", "abcnews", "politico", "wsj", "washingtonpost", "bbc"])
            or any(w in cand_url_lower for w in ["nyt", "nytimes", "apnews", "reuters", "gettyimages", "bloomberg", "bwbx", "abcnews", "politico"])
        )
        if is_wire:
            score += 0.25

        # 4. Temporal seniority
        dt_cand = self.parse_iso_datetime(candidate.timestamp)
        dt_target = self.parse_iso_datetime(target_candidate.timestamp)
        if dt_cand and dt_target and dt_cand < dt_target:
            score += 0.10
            if dt_cand.year <= 2021:
                score += 0.05
        elif dt_cand and dt_cand.year <= 2021:
            score += 0.05

        # 5. Comparative seniority against earliest verified match in ledger
        if dt_cand and earliest_dt:
            diff_from_earliest_sec = (dt_cand - earliest_dt).total_seconds()
            if diff_from_earliest_sec > 43200:  # > 12 hours after earliest publication
                days_late = diff_from_earliest_sec / 86400.0
                penalty = min(0.30, 0.15 + 0.05 * math.log2(max(1.0, days_late)))
                score -= penalty

        return round(min(0.99, max(0.05, score)), 4)

    def crop_salient_subregion(self, image_path: str, crop_ratio: float = 0.60) -> Optional[str]:
        """
        Bellingcat Sub-Region Crop technique:
        Crops the central salient 60% sub-region of the image to bypass peripheral
        watermarks, letterboxing, broadcast banners, or edge modifications.
        """
        try:
            img = Image.open(image_path)
            w, h = img.size
            cw = int(w * crop_ratio)
            ch = int(h * crop_ratio)
            x1 = int(w * (1.0 - crop_ratio) / 2.0)
            y1 = int(h * (1.0 - crop_ratio) / 2.0)
            x2 = x1 + cw
            y2 = y1 + ch

            cropped = img.crop((x1, y1, x2, y2))
            sha = hashlib.sha256(cropped.tobytes()).hexdigest()[:16]
            crop_path = os.path.join(self.neighbor_crawler.cache_dir, f"bellingcat_crop_{sha}.jpg")
            cropped.convert("RGB").save(crop_path, format="JPEG", quality=95)
            logger.info("astar_engine.bellingcat_subregion_created", path=crop_path, box=[x1, y1, x2, y2])
            return crop_path
        except Exception as e:
            logger.warn("astar_engine.bellingcat_crop_failed", error=str(e))
            return None

    async def _ingest_target(self, target_input: str) -> NeighborCandidate:
        """Ingests target whether provided as a local file or external HTTP URL."""
        if target_input.startswith(("http://", "https://")):
            cand = await self.neighbor_crawler.download_and_ingest(target_input, source_hop="target_query")
            if not cand:
                raise ValueError(f"Failed to download target image from URL: {target_input}")
            return cand
        
        abs_path = os.path.abspath(target_input)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Target local image file not found: {abs_path}")

        img = Image.open(abs_path)
        w, h = img.size
        pdq_hex = self.neighbor_crawler.compute_pdq(img)

        # File modification time
        mtime = os.path.getmtime(abs_path)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)

        return NeighborCandidate(
            url=f"file://{abs_path}",
            local_path=abs_path,
            width=w,
            height=h,
            pdq_hex=pdq_hex,
            timestamp=dt.isoformat(),
            domain="local_target",
            is_authoritative_wire=False,
            source_hop="target",
        )

    async def trace_origin_async(self, target_input: str) -> Dict[str, Any]:
        """
        Asynchronous execution of the A* visual attribution graph traversal.
        """
        start_time = time.time()
        logger.info("astar_engine.trace_origin.start", target=target_input)

        # 1. Ingest target media
        target_cand = await self._ingest_target(target_input)
        logger.info(
            "astar_engine.target_ingested",
            url=target_cand.url,
            res=f"{target_cand.width}x{target_cand.height}",
            pdq=target_cand.pdq_hex,
            timestamp=target_cand.timestamp,
        )

        closed_set = ClosedSet()
        closed_set.add(target_cand.url, target_cand.pdq_hex)

        match_ledger = PartialMatchLedger()
        open_heap: List[Tuple[float, int, AStarNode]] = []
        tie_counter = itertools.count()

        # Target root node
        root_node = AStarNode(
            node_id="target_root",
            candidate=target_cand,
            parent_id=None,
            depth=0,
            g_score=0.0,
            h_score=0.0,
            f_score=0.0,
            match_result=MatchResult(
                is_match=True,
                is_partial_crop=False,
                crop_type="exact",
                pdq_distance=0,
                inlier_ratio=1.0,
                inlier_count=5000,
                bounding_box=[0, 0, target_cand.width, target_cand.height],
                scale_factor=1.0,
                similarity_score=1.0,
                details="Query Target Image",
            ),
            role="target",
        )

        heapq.heappush(open_heap, (0.0, next(tie_counter), root_node))

        nodes_by_id: Dict[str, AStarNode] = {root_node.node_id: root_node}
        explored_nodes: List[AStarNode] = []
        patient_zero_node: Optional[AStarNode] = None
        best_origin_confidence: float = 0.0

        iteration = 0
        semantic_pivot_triggered = False
        semantic_pivot_info = {"activated": False, "candidates_count": 0}

        # Traversal Loop
        while open_heap and iteration < self.max_iterations:
            elapsed = time.time() - start_time
            if elapsed > self.time_budget_sec:
                logger.info("astar_engine.time_budget_reached", elapsed=elapsed)
                break

            iteration += 1
            f_curr, _, current_node = heapq.heappop(open_heap)
            explored_nodes.append(current_node)

            logger.info(
                "astar_engine.step",
                iteration=iteration,
                node_id=current_node.node_id,
                domain=current_node.candidate.domain,
                f_score=current_node.f_score,
                g_score=current_node.g_score,
                h_score=current_node.h_score,
                crop_type=current_node.match_result.crop_type if current_node.match_result else "none",
            )

            # Evaluate Patient Zero if not the root node
            if current_node.node_id != root_node.node_id and current_node.match_result:
                conf = self.evaluate_origin_confidence(
                    candidate=current_node.candidate,
                    match=current_node.match_result,
                    target_candidate=target_cand,
                )
                if conf > best_origin_confidence:
                    best_origin_confidence = conf
                    patient_zero_node = current_node

                # If an authoritative uncropped wire master is found with high confidence, evaluate as goal
                is_wire = (
                    current_node.candidate.is_authoritative_wire
                    or any(w in current_node.candidate.domain.lower() for w in ["nyt", "nytimes", "apnews", "reuters", "getty", "afp", "pti", "ap.org"])
                )
                if (
                    current_node.match_result.crop_type == "target_is_crop_of_candidate"
                    and is_wire
                    and conf >= 0.90
                    and iteration >= 4
                ):
                    logger.info("astar_engine.goal_reached.uncropped_wire_master", url=current_node.candidate.url)
                    patient_zero_node = current_node
                    best_origin_confidence = conf
                    break

            # Check depth limit
            if current_node.depth >= self.max_depth:
                continue

            # Expand neighbors: Root target runs full web reverse search (Google Lens)
            # Child nodes only expand by scraping their host web page for uncompressed 4K master wire assets
            if current_node.node_id == root_node.node_id:
                parent_url_arg = current_node.candidate.url if current_node.candidate.url.startswith("http") else None
                neighbors = await self.neighbor_crawler.get_neighbors(
                    candidate_image_path=current_node.candidate.local_path,
                    parent_url=parent_url_arg,
                )

                # Bellingcat Sub-Region Crop search:
                # If full image reverse search returns 0 hits, crop the central salient 60% sub-region and run Google Lens on the crop.
                if len(neighbors) == 0:
                    logger.info("astar_engine.bellingcat_subregion_crop.triggered", image=current_node.candidate.local_path)
                    print("  [*] Reverse search returned 0 hits. Executing Bellingcat Salient 60% Sub-Region Crop with Google Lens...")
                    crop_path = self.crop_salient_subregion(current_node.candidate.local_path, crop_ratio=0.60)
                    if crop_path:
                        crop_neighbors = await self.neighbor_crawler.get_neighbors(crop_path)
                        neighbors.extend(crop_neighbors)
            else:
                neighbors = []
                parent_url_arg = current_node.candidate.url if current_node.candidate.url.startswith("http") else None
                if parent_url_arg and not any(parent_url_arg.lower().split("?")[0].endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    dom_urls = await self.neighbor_crawler.scrape_page_images(parent_url_arg)
                    for p_img in dom_urls[:8]:
                        c = await self.neighbor_crawler.download_and_ingest(
                            image_url=p_img,
                            referer=parent_url_arg,
                            source_hop="parent_dom_expansion",
                        )
                        if c:
                            neighbors.append(c)

            for cand in neighbors:
                if closed_set.contains(cand.url, cand.pdq_hex):
                    continue
                closed_set.add(cand.url, cand.pdq_hex)

                # Visual comparison against target
                match_res = self.subimage_matcher.compare_images(
                    target_path=target_cand.local_path,
                    candidate_path=cand.local_path,
                )

                # Compute A* scores
                g, h, f = self.compute_scores(
                    candidate=cand,
                    match=match_res,
                    target_candidate=target_cand,
                    depth=current_node.depth + 1,
                )

                # Assign role
                if match_res.crop_type == "target_is_crop_of_candidate":
                    role = "uncropped_parent"
                elif match_res.crop_type == "candidate_is_crop_of_target":
                    role = "derivative_crop"
                elif match_res.crop_type == "exact" or match_res.pdq_distance <= 30:
                    role = "exact_clone"
                else:
                    role = "frontier"

                # Record in match ledger if matching
                if match_res.is_match:
                    match_ledger.record_match(candidate=cand, match=match_res)

                child_node_id = f"node_{iteration}_{len(nodes_by_id)}"
                child_node = AStarNode(
                    node_id=child_node_id,
                    candidate=cand,
                    parent_id=current_node.node_id,
                    depth=current_node.depth + 1,
                    g_score=g,
                    h_score=h,
                    f_score=f,
                    match_result=match_res,
                    role=role,
                )

                nodes_by_id[child_node_id] = child_node
                heapq.heappush(open_heap, (f, next(tie_counter), child_node))

                # Immediately evaluate origin confidence for all matching candidates
                if match_res.is_match:
                    conf = self.evaluate_origin_confidence(
                        candidate=cand,
                        match=match_res,
                        target_candidate=target_cand,
                    )
                    if conf > best_origin_confidence or (
                        conf == best_origin_confidence and (patient_zero_node is None or f < patient_zero_node.f_score)
                    ):
                        best_origin_confidence = conf
                        patient_zero_node = child_node

            # Palantir-Style Semantic Pivot Layer:
            # When initial direct visual neighbors return 0 valid matches or all matches have high visual distance (>75 bits)
            if current_node.node_id == root_node.node_id and not semantic_pivot_triggered:
                matching_nodes = [
                    n for n in nodes_by_id.values()
                    if n.node_id != root_node.node_id and n.match_result and n.match_result.is_match
                ]
                has_close_match = any(
                    (n.match_result.pdq_distance <= 75 or n.match_result.crop_type in ("target_is_crop_of_candidate", "candidate_is_crop_of_target", "exact"))
                    for n in matching_nodes
                )
                if not has_close_match:
                    semantic_pivot_triggered = True
                    print("  [*] Direct visual search returned 0 close matches. Activating Palantir Semantic Pivot Layer...")
                    logger.info("astar_engine.semantic_pivot.activated", target=target_cand.url)
                    pivot_candidates = await self.semantic_pivot.run_pivot(
                        target_image_path=target_cand.local_path,
                        matcher=self.subimage_matcher,
                        neighbor_crawler=self.neighbor_crawler,
                    )
                    # Reset clock so harvesting duration does not exhaust A* traversal budget
                    start_time = time.time()
                    semantic_pivot_info["activated"] = True
                    semantic_pivot_info["candidates_count"] = len(pivot_candidates)
                    semantic_pivot_info["harvested_count"] = len(self.semantic_pivot.last_harvested)
                    semantic_pivot_info["deconstruction"] = (
                        self.semantic_pivot.last_deconstruction.to_dict()
                        if self.semantic_pivot.last_deconstruction else None
                    )
                    for cand in pivot_candidates:
                        if closed_set.contains(cand.url, cand.pdq_hex):
                            continue
                        closed_set.add(cand.url, cand.pdq_hex)

                        match_res = self.subimage_matcher.compare_images(
                            target_path=target_cand.local_path,
                            candidate_path=cand.local_path,
                        )
                        g, h, f = self.compute_scores(
                            candidate=cand,
                            match=match_res,
                            target_candidate=target_cand,
                            depth=1,
                        )
                        if match_res.crop_type == "target_is_crop_of_candidate":
                            role = "uncropped_parent"
                        elif match_res.crop_type == "candidate_is_crop_of_target":
                            role = "derivative_crop"
                        elif match_res.crop_type == "exact" or match_res.pdq_distance <= 30:
                            role = "exact_clone"
                        else:
                            role = "frontier"

                        if match_res.is_match:
                            match_ledger.record_match(candidate=cand, match=match_res)

                        child_node_id = f"pivot_{len(nodes_by_id)}"
                        child_node = AStarNode(
                            node_id=child_node_id,
                            candidate=cand,
                            parent_id=root_node.node_id,
                            depth=1,
                            g_score=g,
                            h_score=h,
                            f_score=f,
                            match_result=match_res,
                            role=role,
                        )
                        nodes_by_id[child_node_id] = child_node
                        heapq.heappush(open_heap, (f, next(tie_counter), child_node))

                        if match_res.is_match:
                            conf = self.evaluate_origin_confidence(
                                candidate=cand,
                                match=match_res,
                                target_candidate=target_cand,
                            )
                            if conf > best_origin_confidence or (
                                conf == best_origin_confidence and (patient_zero_node is None or f < patient_zero_node.f_score)
                            ):
                                best_origin_confidence = conf
                                patient_zero_node = child_node

        # Determine earliest verified publication timestamp across all matching nodes
        earliest_verified_dt = None
        for nid, node in nodes_by_id.items():
            if nid == root_node.node_id or not node.match_result:
                continue
            if node.match_result.is_match or node.match_result.pdq_distance <= 30:
                dt = self.parse_iso_datetime(node.candidate.timestamp)
                if dt and 2000 <= dt.year <= 2027:
                    if earliest_verified_dt is None or dt < earliest_verified_dt:
                        earliest_verified_dt = dt

        # Provenance Selection: scan all explored and frontier nodes for the optimal Patient Zero
        for nid, node in nodes_by_id.items():
            if nid == root_node.node_id or not node.match_result:
                continue
            if not node.match_result.is_match and node.match_result.pdq_distance > 30:
                continue
            conf = self.evaluate_origin_confidence(
                candidate=node.candidate,
                match=node.match_result,
                target_candidate=target_cand,
                earliest_dt=earliest_verified_dt,
            )
            # Rank candidates: higher confidence wins; tie-break on earlier date, larger area, then f_score
            is_better = False
            if patient_zero_node is None:
                is_better = True
            elif conf > best_origin_confidence:
                is_better = True
            elif conf == best_origin_confidence:
                dt_curr = self.parse_iso_datetime(node.candidate.timestamp)
                dt_best = self.parse_iso_datetime(patient_zero_node.candidate.timestamp)
                curr_area = node.candidate.width * node.candidate.height
                best_area = patient_zero_node.candidate.width * patient_zero_node.candidate.height

                # If publication times differ by more than 1 hour, earlier date takes priority
                if dt_curr and dt_best and abs((dt_curr - dt_best).total_seconds()) >= 3600:
                    if dt_curr < dt_best:
                        is_better = True
                else:
                    # Within the same publication event/hour, prefer the higher resolution master
                    if curr_area > best_area:
                        is_better = True
                    elif curr_area == best_area and node.f_score < patient_zero_node.f_score:
                        is_better = True

            if is_better:
                best_origin_confidence = conf
                patient_zero_node = node

        # Designate Patient Zero role
        if patient_zero_node:
            patient_zero_node.role = "patient_zero"

        # Construct lineage path back to root
        lineage_chain = []
        curr = patient_zero_node
        while curr:
            lineage_chain.append({
                "node_id": curr.node_id,
                "url": curr.candidate.url,
                "domain": curr.candidate.domain,
                "resolution": [curr.candidate.width, curr.candidate.height],
                "role": curr.role,
                "crop_type": curr.match_result.crop_type if curr.match_result else "none",
                "f_score": curr.f_score,
            })
            curr = nodes_by_id.get(curr.parent_id) if curr.parent_id else None
        lineage_chain.reverse()

        # Build PyVis Graph
        all_graph_nodes = list(nodes_by_id.values())
        self.generate_pyvis_tree(
            nodes=all_graph_nodes,
            patient_zero=patient_zero_node,
            target_node=root_node,
            output_path=self.output_graph_path,
        )

        total_time = round(time.time() - start_time, 2)

        # Assemble ranked candidate sources with probabilities
        candidates_map: Dict[str, Dict[str, Any]] = {}
        for nid, node in nodes_by_id.items():
            if nid == root_node.node_id or not node.match_result:
                continue
            if not node.match_result.is_match and node.match_result.pdq_distance > 30:
                continue

            conf = self.evaluate_origin_confidence(
                candidate=node.candidate,
                match=node.match_result,
                target_candidate=target_cand,
                earliest_dt=earliest_verified_dt,
            )

            crop_t = node.match_result.crop_type
            if crop_t == "target_is_crop_of_candidate":
                friendly_match = "parent_master_original"
            elif crop_t == "exact" or node.match_result.pdq_distance <= 30:
                friendly_match = "exact_clone"
            elif crop_t == "candidate_is_crop_of_target":
                friendly_match = "downscaled_crop"
            else:
                friendly_match = "visual_match"

            c_url = node.candidate.url
            clean_media_url = f"file://{os.path.abspath(node.candidate.local_path)}" if c_url.startswith("data:image/") else c_url
            if c_url not in candidates_map or conf > candidates_map[c_url]["probability"]:
                candidates_map[c_url] = {
                    "node_id": node.node_id,
                    "media_url": clean_media_url,
                    "source_page_url": getattr(node.candidate, "source_page_url", None) or node.candidate.url,
                    "domain": node.candidate.domain,
                    "probability": round(conf, 4),
                    "match_type": friendly_match,
                    "crop_type": crop_t,
                    "resolution": [node.candidate.width, node.candidate.height],
                    "timestamp": node.candidate.timestamp,
                    "is_authoritative_wire": node.candidate.is_authoritative_wire,
                    "local_cached_path": node.candidate.local_path,
                    "inlier_count": node.match_result.inlier_count,
                    "pdq_distance": node.match_result.pdq_distance,
                    "evidence": node.match_result.details,
                    "f_score": node.f_score,
                }

        candidate_sources = sorted(
            candidates_map.values(),
            key=lambda c: (
                c["probability"],
                c["resolution"][0] * c["resolution"][1],
                -c["f_score"]
            ),
            reverse=True,
        )

        # Assemble summary results
        pz_dict = None
        if patient_zero_node:
            m = patient_zero_node.match_result
            crop_t = m.crop_type if m else "none"
            if crop_t == "target_is_crop_of_candidate":
                friendly_match = "parent_master_original"
            elif crop_t == "exact" or (m and m.pdq_distance <= 30):
                friendly_match = "exact_clone"
            elif crop_t == "candidate_is_crop_of_target":
                friendly_match = "downscaled_crop"
            else:
                friendly_match = "visual_match"

            pz_media = patient_zero_node.candidate.url
            if pz_media.startswith("data:image/"):
                pz_media = f"file://{os.path.abspath(patient_zero_node.candidate.local_path)}"

            pz_dict = {
                "node_id": patient_zero_node.node_id,
                "media_url": pz_media,
                "source_page_url": getattr(patient_zero_node.candidate, "source_page_url", None) or patient_zero_node.candidate.url,
                "local_cached_path": patient_zero_node.candidate.local_path,
                "domain": patient_zero_node.candidate.domain,
                "publisher": patient_zero_node.candidate.domain,
                "probability": best_origin_confidence,
                "confidence": best_origin_confidence,
                "timestamp": patient_zero_node.candidate.timestamp,
                "resolution": [patient_zero_node.candidate.width, patient_zero_node.candidate.height],
                "match_type": friendly_match,
                "crop_type": crop_t,
                "bounding_box": m.bounding_box if m else None,
                "scale_factor": m.scale_factor if m else 1.0,
                "pdq_distance": m.pdq_distance if m else 999,
                "inlier_count": m.inlier_count if m else 0,
                "inlier_ratio": m.inlier_ratio if m else 0.0,
                "is_authoritative_wire": patient_zero_node.candidate.is_authoritative_wire,
                "f_score": patient_zero_node.f_score,
                "evidence": m.details if m else "Identified as Primary Origin",
                "details": m.details if m else "Identified as Primary Origin",
            }

        result = {
            "target": {
                "url": target_cand.url,
                "local_path": target_cand.local_path,
                "resolution": [target_cand.width, target_cand.height],
                "pdq_hex": target_cand.pdq_hex,
                "timestamp": target_cand.timestamp,
                "domain": target_cand.domain,
            },
            "target_media": {
                "url": target_cand.url,
                "local_path": target_cand.local_path,
                "resolution": [target_cand.width, target_cand.height],
                "pdq_hex": target_cand.pdq_hex,
                "timestamp": target_cand.timestamp,
                "domain": target_cand.domain,
            },
            "patient_zero": pz_dict,
            "candidate_sources": candidate_sources,
            "iterations": iteration,
            "nodes_explored": len(explored_nodes),
            "nodes_frontier": len(open_heap),
            "partial_match_ledger": match_ledger.get_entries(),
            "graph_path": self.output_graph_path,
            "graph_html_path": self.output_graph_path,
            "lineage_path": lineage_chain,
            "semantic_pivot": semantic_pivot_info,
            "execution_time_sec": total_time,
        }

        logger.info(
            "astar_engine.trace_origin.finished",
            patient_zero=pz_dict["domain"] if pz_dict else "None",
            confidence=best_origin_confidence,
            iterations=iteration,
            time_sec=total_time,
        )

        return result

    def trace_origin(self, target_input: str) -> Dict[str, Any]:
        """Synchronous wrapper for trace_origin_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.trace_origin_async(target_input))
            else:
                return loop.run_until_complete(self.trace_origin_async(target_input))
        except RuntimeError:
            return asyncio.run(self.trace_origin_async(target_input))

    def generate_pyvis_tree(
        self,
        nodes: List[AStarNode],
        patient_zero: Optional[AStarNode],
        target_node: AStarNode,
        output_path: str,
    ) -> str:
        """
        Generates interactive PyVis visual attribution tree at output_path.
        Color codes:
          - Target: Cyan (#00E5FF)
          - Patient Zero: Vibrant Red (#FF1744)
          - Uncropped Parent: Deep Orange (#FF9100)
          - Derivative Crop: Green (#00E676)
          - Exact Clone: Purple (#9C27B0)
          - Frontier: Grey (#78909C)
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if HAS_PYVIS:
            try:
                net = Network(
                    height="750px",
                    width="100%",
                    bgcolor="#0d1117",
                    font_color="#f0f6fc",
                    directed=True,
                )
                net.heading = "Recursive Visual Attribution & Crop Lineage Tree (A* Engine)"

                # Color definitions
                ROLE_CONFIG = {
                    "target": {"color": "#00E5FF", "size": 38, "shape": "box"},
                    "patient_zero": {"color": "#FF1744", "size": 42, "shape": "star"},
                    "uncropped_parent": {"color": "#FF9100", "size": 28, "shape": "box"},
                    "derivative_crop": {"color": "#00E676", "size": 24, "shape": "ellipse"},
                    "exact_clone": {"color": "#9C27B0", "size": 26, "shape": "ellipse"},
                    "frontier": {"color": "#78909C", "size": 16, "shape": "dot"},
                }

                pz_id = patient_zero.node_id if patient_zero else None

                for node in nodes:
                    role = "patient_zero" if node.node_id == pz_id else node.role
                    cfg = ROLE_CONFIG.get(role, ROLE_CONFIG["frontier"])

                    cand = node.candidate
                    match = node.match_result

                    crop_str = match.crop_type if match else "none"
                    bbox_str = f"BBox: {match.bounding_box}" if match and match.bounding_box else ""
                    inliers_str = f"Inliers: {match.inlier_count} ({match.inlier_ratio:.1%})" if match else ""

                    label = f"{cand.domain}\n{cand.width}x{cand.height}\nf={node.f_score:.1f}"
                    if role == "target":
                        label = f"[TARGET QUERY]\n{cand.domain}\n{cand.width}x{cand.height}"
                    elif role == "patient_zero":
                        label = f"[PATIENT ZERO]\n{cand.domain}\n{cand.width}x{cand.height}"

                    title_hover = (
                        f"<b>Role:</b> {role.upper()}<br>"
                        f"<b>Domain:</b> {cand.domain}<br>"
                        f"<b>Resolution:</b> {cand.width}x{cand.height}<br>"
                        f"<b>Timestamp:</b> {cand.timestamp or 'Unknown'}<br>"
                        f"<b>Wire Authority:</b> {cand.is_authoritative_wire}<br>"
                        f"<b>A* Scores:</b> f={node.f_score:.1f} (g={node.g_score:.1f}, h={node.h_score:.1f})<br>"
                        f"<b>Match Type:</b> {crop_str}<br>"
                        f"<b>{bbox_str}</b><br>"
                        f"<b>{inliers_str}</b><br>"
                        f"<b>PDQ Dist:</b> {match.pdq_distance if match else 'N/A'}<br>"
                        f"<b>URL:</b> {cand.url}"
                    )

                    net.add_node(
                        node.node_id,
                        label=label,
                        title=title_hover,
                        color=cfg["color"],
                        size=cfg["size"],
                        shape=cfg["shape"],
                    )

                # Add directed edges
                for node in nodes:
                    if node.parent_id:
                        is_to_origin = (node.node_id == pz_id)
                        is_match = node.match_result and node.match_result.is_match

                        if is_to_origin:
                            edge_color = "#FF1744"
                            edge_width = 4
                            edge_label = f"ORIGIN (f={node.f_score:.1f})"
                            dashes = False
                        elif is_match:
                            edge_color = "#00E5FF"
                            edge_width = 2
                            edge_label = f"{node.match_result.crop_type}"
                            dashes = False
                        else:
                            edge_color = "#484f58"
                            edge_width = 1
                            edge_label = f"f={node.f_score:.1f}"
                            dashes = True

                        net.add_edge(
                            node.parent_id,
                            node.node_id,
                            color=edge_color,
                            width=edge_width,
                            label=edge_label,
                            dashes=dashes,
                        )

                net.save_graph(output_path)
                logger.info("astar_engine.pyvis_exported", path=output_path, nodes=len(nodes))
                return output_path
            except Exception as e:
                logger.warn("astar_engine.pyvis_error", error=str(e))

        # Fallback standalone HTML visualization if PyVis fails
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>A* Visual Attribution Tree</title>
    <style>
        body {{ background-color: #0d1117; color: #c9d1d9; font-family: monospace; padding: 24px; }}
        h1 {{ color: #58a6ff; }}
        .node {{ border: 1px solid #30363d; padding: 12px; margin: 10px 0; border-radius: 6px; }}
        .target {{ border-left: 6px solid #00E5FF; background: #161b22; }}
        .origin {{ border-left: 6px solid #FF1744; background: #211218; }}
        .crop {{ border-left: 6px solid #FF9100; background: #1f1b18; }}
    </style>
</head>
<body>
    <h1>A* Visual Attribution Lineage</h1>
    <p>Visual tree generated with {len(nodes)} explored nodes.</p>
    <div>
""")
            for n in nodes:
                cls_name = "origin" if (patient_zero and n.node_id == patient_zero.node_id) else ("target" if n.role == "target" else "crop")
                f.write(f"""
        <div class="node {cls_name}">
            <h3>[{n.role.upper()}] {n.candidate.domain} ({n.candidate.width}x{n.candidate.height})</h3>
            <p>URL: {n.candidate.url}</p>
            <p>Scores: f={n.f_score:.1f} (g={n.g_score:.1f}, h={n.h_score:.1f}) | Match: {n.match_result.crop_type if n.match_result else 'none'}</p>
        </div>
""")
            f.write("</div></body></html>")
        return output_path