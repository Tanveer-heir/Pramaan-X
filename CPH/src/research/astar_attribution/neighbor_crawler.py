"""
Visual Graph Neighbor Expansion Module.
Expands candidate neighbors using Yandex Reverse Image Search and DOM scraping
for A* search source attribution.
"""

import os
import re
import io
import json
import hashlib
import base64
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set, Tuple
from urllib.parse import urlparse, urljoin
from email.utils import parsedate_to_datetime

import asyncio
import httpx
import numpy as np
from PIL import Image
import pdqhash
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from src.common.logger import logger
from src.tools.yandex_search import YandexPlaywrightSearchTool
from src.tools.apify_lens import ApifyGoogleLensTool

AUTHORITATIVE_WIRE_DOMAINS = [
    "apimages.com",
    "reuters.com",
    "gettyimages.com",
    "afp.com",
    "apnews.com",
    "ap.org",
    "pti.in",
    "nytimes.com",
    "nyt.com",
    "bloomberg.com",
    "bwbx.io",
    "abcnews.go.com",
    "abcnewsfe.com",
    "wsj.com",
    "washingtonpost.com",
    "bbc.co.uk",
    "bbc.com",
    "cnn.com",
    "politico.com",
    "businessinsider.com",
    "insider.com",
    "france24.com",
    "epa.eu",
    "upi.com",
    "afpforum.com",
]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

IMAGE_EXTENSIONS_REGEX = re.compile(r"\.(jpe?g|png|webp)($|\?|#)", re.IGNORECASE)
DATE_REGEX_ISO = re.compile(r"/(20\d{2})[-/_](0[1-9]|1[0-2])[-/_](0[1-9]|[12]\d|3[01])")
DATE_REGEX_NAME = re.compile(r"(20\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])")
DATE_REGEX_COMPACT = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")
DATE_REGEX_YEAR_MONTH = re.compile(r"/(20\d{2})/(0[1-9]|1[0-2])/")


@dataclass
class NeighborCandidate:
    url: str
    local_path: str
    width: int
    height: int
    pdq_hex: str
    timestamp: Optional[str]
    domain: str
    is_authoritative_wire: bool
    source_hop: str  # 'yandex_similar', 'page_dom_scrape', etc.
    source_page_url: Optional[str] = None


class VisualNeighborCrawler:
    """
    Crawls and ingests visual neighbors from reverse image search engines (Yandex)
    and DOM scraping of parent web pages.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/astar",
        user_agent: str = DEFAULT_USER_AGENT,
        lens_tool: Any = None,
        yandex_tool: Any = None,
        timeout_sec: float = 12.0,
    ):
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.lens_tool = lens_tool or yandex_tool or ApifyGoogleLensTool
        self.timeout_sec = timeout_sec
        self.page_metadata_cache: Dict[str, Dict[str, Any]] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _extract_page_metadata(html: str, page_url: str) -> Dict[str, Any]:
        """Extracts publication timestamp and metadata from HTML meta tags and JSON-LD."""
        meta_dict: Dict[str, Any] = {"timestamp": None}
        if not html:
            return meta_dict
        try:
            soup = BeautifulSoup(html, "html.parser")
            # 1. Meta tags (OpenGraph, itemprop, standard pubdate)
            for m in soup.find_all(["meta"]):
                prop = (m.get("property") or m.get("name") or m.get("itemprop") or "").lower()
                if any(k in prop for k in ["published_time", "pubdate", "publishdate", "datepublished", "datecreated", "release_date"]):
                    content = m.get("content")
                    if content and len(content.strip()) >= 10:
                        meta_dict["timestamp"] = content.strip()
                        break

            # 2. JSON-LD structured data
            if not meta_dict["timestamp"]:
                for s in soup.find_all("script", type="application/ld+json"):
                    try:
                        if s.string:
                            data = json.loads(s.string)
                            items = data if isinstance(data, list) else [data]
                            if isinstance(data, dict) and "@graph" in data:
                                items = data["@graph"]
                            for it in items:
                                if isinstance(it, dict) and "datePublished" in it:
                                    meta_dict["timestamp"] = str(it["datePublished"]).strip()
                                    break
                            if meta_dict["timestamp"]:
                                break
                    except Exception:
                        pass

            # 3. Fallback to URL regex
            if not meta_dict["timestamp"]:
                m = DATE_REGEX_ISO.search(page_url) or DATE_REGEX_NAME.search(page_url) or DATE_REGEX_COMPACT.search(page_url)
                if m:
                    y, mo, d = m.groups()
                    meta_dict["timestamp"] = f"{y}-{mo}-{d}T00:00:00Z"
        except Exception as e:
            logger.debug("neighbor_crawler.meta_extract_error", error=str(e))

        return meta_dict

    @staticmethod
    def is_authoritative_domain(domain: str) -> bool:
        """Checks if domain belongs to a known authoritative wire/news agency."""
        clean_domain = domain.lower().strip()
        if ":" in clean_domain:
            clean_domain = clean_domain.split(":")[0]
        return any(
            clean_domain == wire or clean_domain.endswith("." + wire)
            for wire in AUTHORITATIVE_WIRE_DOMAINS
        )

    @staticmethod
    def extract_timestamp(headers: Dict[str, str], url: str) -> Optional[str]:
        """
        Extracts publication/modification timestamp from HTTP headers (Last-Modified, Date)
        or regex patterns in URL/filename.
        """
        # 1. Regex date in URL / filename: YYYY/MM/DD or YYYY-MM-DD
        m = DATE_REGEX_ISO.search(url)
        if m:
            y, mo, d = m.groups()
            return f"{y}-{mo}-{d}T00:00:00Z"

        m = DATE_REGEX_NAME.search(url)
        if m:
            y, mo, d = m.groups()
            return f"{y}-{mo}-{d}T00:00:00Z"

        m = DATE_REGEX_COMPACT.search(url)
        if m:
            y, mo, d = m.groups()
            return f"{y}-{mo}-{d}T00:00:00Z"

        # 2. Last-Modified HTTP Header
        last_modified = headers.get("last-modified")
        if last_modified:
            try:
                dt = parsedate_to_datetime(last_modified)
                return dt.isoformat()
            except Exception:
                pass

        # 3. Fallback to Year/Month in URL if full day not present
        m = DATE_REGEX_YEAR_MONTH.search(url)
        if m:
            y, mo = m.groups()
            return f"{y}-{mo}-01T00:00:00Z"

        # 4. HTTP Date header as fallback
        date_hdr = headers.get("date")
        if date_hdr:
            try:
                dt = parsedate_to_datetime(date_hdr)
                return dt.isoformat()
            except Exception:
                pass

        return None

    @staticmethod
    def compute_pdq(img: Image.Image) -> str:
        """Computes Meta's 256-bit PDQ perceptual hash and formats as a 64-char hex string."""
        rgb_img = img.convert("RGB")
        img_np = np.array(rgb_img)
        hash_vector, _ = pdqhash.compute(img_np)
        return "".join(
            f"{b:02x}"
            for b in [
                int("".join(str(int(x)) for x in hash_vector[i : i + 8]), 2)
                for i in range(0, 256, 8)
            ]
        )

    async def download_and_ingest(
        self,
        image_url: str,
        referer: Optional[str] = None,
        source_hop: str = "crawl",
        explicit_domain: Optional[str] = None,
    ) -> Optional[NeighborCandidate]:
        """
        Downloads image with httpx (10s timeout, custom User-Agent) or decodes base64 data URIs.
        Saves to data/cache/astar/<sha256_hash>.jpg.
        Computes dimensions (width, height) and Meta PDQ hash.
        Extracts timestamp and domain authority.
        Filters out small icons/thumbnails (< 70px for thumbnails, < 120px for scraped).
        """
        if not image_url:
            return None

        content = None
        resp_headers = {}

        # 1. Base64 data URI handling
        if image_url.startswith("data:image/"):
            try:
                b64_part = image_url.split(",", 1)[1] if "," in image_url else image_url
                content = base64.b64decode(b64_part)
            except Exception as e:
                logger.debug("neighbor_crawler.base64_decode_error", error=str(e))
                return None
        elif not image_url.startswith(("http://", "https://")):
            return None
        elif "favicon" in image_url.lower():
            logger.debug("neighbor_crawler.filter_favicon", url=image_url)
            return None
        else:
            headers = {"User-Agent": self.user_agent}
            if referer:
                headers["Referer"] = referer

            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_sec,
                    follow_redirects=True,
                    headers=headers,
                ) as client:
                    resp = await client.get(image_url)
                    if resp.status_code != 200:
                        logger.debug(
                            "neighbor_crawler.download_http_err",
                            url=image_url,
                            status_code=resp.status_code,
                        )
                        return None
                    content = resp.content
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            except Exception as e:
                logger.debug("neighbor_crawler.download_network_err", url=image_url, error=str(e))
                return None

        if not content:
            return None

        # Open image with PIL to verify integrity and dimensions
        try:
            img = Image.open(io.BytesIO(content))
            width, height = img.size
        except Exception as e:
            logger.debug("neighbor_crawler.invalid_image", url=image_url[:60], error=str(e))
            return None

        # Filter small icons/thumbnails (< 70px for thumbnails, < 120px for page assets)
        min_dim = 70 if (image_url.startswith("data:image/") or "thumb" in source_hop) else 120
        if width < min_dim or height < min_dim:
            logger.debug(
                "neighbor_crawler.filtered_small_image",
                url=image_url[:60],
                width=width,
                height=height,
            )
            return None

        # Compute sha256 of raw content
        sha256_hash = hashlib.sha256(content).hexdigest()
        local_path = os.path.join(self.cache_dir, f"{sha256_hash}.jpg")

        # Save to cache as JPEG
        if not os.path.exists(local_path):
            img.convert("RGB").save(local_path, format="JPEG", quality=95)

        # Compute PDQ perceptual hash
        pdq_hex = self.compute_pdq(img)

        # Extract timestamp from headers or URL or referer or page metadata cache
        timestamp = self.extract_timestamp(resp_headers, image_url)
        if not timestamp and referer:
            meta_ts = self.page_metadata_cache.get(referer, {}).get("timestamp")
            if meta_ts:
                timestamp = meta_ts
            else:
                timestamp = self.extract_timestamp({}, referer)

        # Domain authority check
        if explicit_domain and explicit_domain not in ("unknown", "google.com", "google", "local_target"):
            domain = explicit_domain
        else:
            raw_parsed = urlparse(image_url)
            raw_domain = (raw_parsed.hostname or raw_parsed.netloc or "").lower().replace("www.", "")
            is_thumb_cdn = any(th in raw_domain for th in ["gstatic.com", "googleusercontent.com", "yandex.net", "bing.net", "apify.com"]) or image_url.startswith("data:image/")

            if is_thumb_cdn and referer:
                ref_parsed = urlparse(referer)
                ref_domain = (ref_parsed.hostname or ref_parsed.netloc or "").lower().replace("www.", "")
                if ref_domain and "google." not in ref_domain:
                    domain = ref_domain
                else:
                    domain = raw_domain
            else:
                domain = raw_domain if raw_domain else "unknown"

        is_wire = self.is_authoritative_domain(domain)

        candidate = NeighborCandidate(
            url=image_url,
            local_path=local_path,
            width=width,
            height=height,
            pdq_hex=pdq_hex,
            timestamp=timestamp,
            domain=domain,
            is_authoritative_wire=is_wire,
            source_hop=source_hop,
            source_page_url=referer,
        )
        logger.info(
            "neighbor_crawler.ingested_candidate",
            url=image_url,
            width=width,
            height=height,
            domain=domain,
            wire=is_wire,
        )
        return candidate

    def _extract_image_urls_from_html(self, html: str, base_url: str) -> List[str]:
        """
        Extracts candidate image URLs from HTML:
        - <img> src, data-src, data-original
        - <img> srcset
        - <a href="*.jpg|png|webp">
        Filters out favicons and obvious icons/logos.
        """
        soup = BeautifulSoup(html, "html.parser")
        candidate_urls: Set[str] = set()

        def is_icon_or_thumb(tag: Any) -> bool:
            # Check attribute dimensions if explicitly defined
            w = tag.get("width")
            h = tag.get("height")
            try:
                if w and int(w) < 150:
                    return True
                if h and int(h) < 150:
                    return True
            except (ValueError, TypeError):
                pass

            # Check class or id names
            class_str = " ".join(tag.get("class", [])) if tag.get("class") else ""
            id_str = tag.get("id", "") or ""
            target_str = (class_str + " " + id_str).lower()
            if any(k in target_str for k in ["favicon", "icon", "logo", "avatar", "badge", "emoji"]):
                return True
            return False

        # 1. Scrape <img> tags
        for img in soup.find_all("img"):
            if is_icon_or_thumb(img):
                continue

            # Standard src attributes
            for attr in ["src", "data-src", "data-original", "data-lazy-src", "data-highres"]:
                src = img.get(attr)
                if src and not src.startswith("data:"):
                    resolved = urljoin(base_url, src.strip())
                    if "favicon" not in resolved.lower():
                        candidate_urls.add(resolved)

            # srcset attribute
            srcset = img.get("srcset")
            if srcset:
                # Format: "url1 300w, url2 600w" or "url1 1x, url2 2x"
                entries = [e.strip() for e in srcset.split(",") if e.strip()]
                for entry in entries:
                    parts = entry.split()
                    if parts and not parts[0].startswith("data:"):
                        resolved = urljoin(base_url, parts[0].strip())
                        if "favicon" not in resolved.lower():
                            candidate_urls.add(resolved)

        # 2. Scrape <picture> <source srcset="..."> tags
        for src_tag in soup.find_all("source"):
            srcset = src_tag.get("srcset")
            if srcset:
                entries = [e.strip() for e in srcset.split(",") if e.strip()]
                for entry in entries:
                    parts = entry.split()
                    if parts and not parts[0].startswith("data:"):
                        resolved = urljoin(base_url, parts[0].strip())
                        if "favicon" not in resolved.lower():
                            candidate_urls.add(resolved)

        # 3. Scrape <meta property="og:image">, <meta name="twitter:image">, <meta itemprop="contentUrl">
        for meta in soup.find_all("meta"):
            prop = (meta.get("property") or meta.get("name") or meta.get("itemprop") or "").lower()
            if any(k in prop for k in ["og:image", "twitter:image", "contenturl", "thumbnailurl", "image"]):
                content = meta.get("content")
                if content and not content.startswith("data:"):
                    resolved = urljoin(base_url, content.strip())
                    if "favicon" not in resolved.lower():
                        candidate_urls.add(resolved)

        # 4. Scrape <a href="..."> links pointing directly to images
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if IMAGE_EXTENSIONS_REGEX.search(href) and not href.startswith("data:"):
                resolved = urljoin(base_url, href)
                if "favicon" not in resolved.lower():
                    candidate_urls.add(resolved)

        # 5. Expand resized/cropped CMS URLs to unscaled originals
        unscaled_additions = set()
        for u in candidate_urls:
            clean_u = re.sub(r"\?(crop|resize|width|w|quality|q)=[^#]*", "", u, flags=re.IGNORECASE).rstrip("?")
            if clean_u != u and IMAGE_EXTENSIONS_REGEX.search(clean_u):
                unscaled_additions.add(clean_u)
        candidate_urls.update(unscaled_additions)

        return list(candidate_urls)

    async def scrape_page_images(self, page_url: str, use_playwright: bool = False) -> List[str]:
        """
        Scrapes a web page using httpx first, falling back to Playwright if needed,
        and extracts candidate image URLs while caching publication metadata.
        """
        logger.info("neighbor_crawler.scrape_page.started", url=page_url)
        html = None
        effective_url = page_url

        # 1. Attempt fast scrape using httpx with UA rotation
        candidate_uas = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            self.user_agent,
        ]
        for candidate_ua in candidate_uas:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_sec,
                    follow_redirects=True,
                    headers={
                        "User-Agent": candidate_ua,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                ) as client:
                    resp = await client.get(page_url)
                    if resp.status_code == 200 and len(resp.text.strip()) > 50:
                        html = resp.text
                        effective_url = str(resp.url)
                        break
            except Exception as e:
                logger.debug("neighbor_crawler.httpx_scrape_failed", url=page_url, ua=candidate_ua, error=str(e))

        # 2. Fallback to Playwright if httpx failed and Playwright is enabled
        if not html and use_playwright:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context(
                        user_agent=self.user_agent,
                        viewport={"width": 1280, "height": 800},
                    )
                    page = await context.new_page()
                    await page.route("**/*.{woff,woff2,ttf,svg}", lambda r: r.abort())
                    resp = await page.goto(page_url, timeout=int(self.timeout_sec * 1000), wait_until="domcontentloaded")
                    await page.wait_for_timeout(1500)
                    html = await page.content()
                    effective_url = page.url
                    await browser.close()
            except Exception as e:
                logger.warn("neighbor_crawler.playwright_scrape_failed", url=page_url, error=str(e))

        if not html:
            return []

        # Extract and cache publication metadata
        meta_dict = self._extract_page_metadata(html, effective_url)
        self.page_metadata_cache[effective_url] = meta_dict
        if page_url != effective_url:
            self.page_metadata_cache[page_url] = meta_dict

        urls = self._extract_image_urls_from_html(html, effective_url)
        logger.info("neighbor_crawler.scrape_page.completed", url=effective_url, count=len(urls))
        return urls

    async def get_neighbors(
        self,
        candidate_image_path: str,
        parent_url: Optional[str] = None,
    ) -> List[NeighborCandidate]:
        """
        Fetches visual neighbors via:
        a) DOM scraping of matching web pages.
        b) Apify Google Lens Reverse Search (exclusive high-fidelity cloud engine).
        """
        neighbors: List[NeighborCandidate] = []
        seen_urls: Set[str] = set()
        seen_domain_hashes: Set[Tuple[str, str]] = set()

        def is_unique_candidate(c: Optional[NeighborCandidate]) -> bool:
            if not c:
                return False
            key = (c.domain.lower(), c.pdq_hex)
            if key in seen_domain_hashes:
                return False
            seen_domain_hashes.add(key)
            return True

        # Step 1: DOM Scrape if parent_url provided
        if parent_url:
            logger.info("neighbor_crawler.get_neighbors.dom_scrape", parent_url=parent_url)
            dom_image_urls = await self.scrape_page_images(parent_url)
            for img_url in dom_image_urls:
                if img_url in seen_urls:
                    continue
                seen_urls.add(img_url)

                cand = await self.download_and_ingest(
                    image_url=img_url,
                    referer=parent_url,
                    source_hop="page_dom_scrape",
                )
                if is_unique_candidate(cand):
                    neighbors.append(cand)

        # Step 2: Reverse Visual Search Engine (Google Lens with Yandex Fallback)
        has_token = self.lens_tool.get_token() if hasattr(self.lens_tool, "get_token") else True
        lens_hits = []
        if has_token:
            logger.info("neighbor_crawler.get_neighbors.lens_search", target=candidate_image_path)
            try:
                import inspect
                sig = inspect.signature(self.lens_tool.search)
                call_kwargs = {"timeout_sec": 120}
                if "max_results" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    call_kwargs["max_results"] = 35
                lens_hits = await self.lens_tool.search(candidate_image_path, **call_kwargs)
            except Exception as e:
                logger.warn("neighbor_crawler.lens_tool_failed", error=str(e))

        # Automatic fallback to local stealth Yandex Playwright visual search
        if not lens_hits:
            logger.info("neighbor_crawler.get_neighbors.yandex_fallback", target=candidate_image_path)
            try:
                lens_hits = await YandexPlaywrightSearchTool.search(candidate_image_path, timeout_sec=30)
                logger.info("neighbor_crawler.yandex_fallback_completed", count=len(lens_hits))
            except Exception as ye:
                logger.warn("neighbor_crawler.yandex_fallback_failed", error=str(ye))

        if lens_hits:
            try:
                pages_to_scrape: List[Tuple[str, str]] = []

                # 1. Concurrently ingest all thumbnails provided by search engines (concurrency=8, 4s timeout)
                thumb_sem = asyncio.Semaphore(8)
                thumb_tasks = []

                async def _download_thumb(t_url: str, ref_u: Optional[str], h_type: str, exp_dom: Optional[str] = None):
                    async with thumb_sem:
                        try:
                            return await asyncio.wait_for(
                                self.download_and_ingest(
                                    image_url=t_url,
                                    referer=ref_u,
                                    source_hop=f"{h_type}_thumb",
                                    explicit_domain=exp_dom,
                                ),
                                timeout=4.0,
                            )
                        except Exception:
                            return None

                for hit in lens_hits:
                    thumb_url = hit.get("thumbnail_url")
                    post_url = hit.get("post_url")
                    src = hit.get("source", "")
                    hit_domain = hit.get("domain")
                    hop_type = "yandex_similar" if "yandex" in src else f"google_lens_{hit.get('match_type', 'match')}"

                    if thumb_url and thumb_url not in seen_urls:
                        seen_urls.add(thumb_url)
                        thumb_tasks.append(_download_thumb(thumb_url, post_url, hop_type, hit_domain))

                    if not post_url or post_url in seen_urls:
                        continue
                    seen_urls.add(post_url)

                    # Skip un-scrapeable internal Google redirect tracking URLs
                    if "google.com/goto" in post_url or "google.com/url" in post_url:
                        continue

                    # Case A: Post URL is direct image URL
                    if IMAGE_EXTENSIONS_REGEX.search(post_url):
                        cand = await self.download_and_ingest(
                            image_url=post_url,
                            source_hop=hop_type,
                            explicit_domain=hit_domain,
                        )
                        if is_unique_candidate(cand):
                            neighbors.append(cand)
                    else:
                        pages_to_scrape.append((post_url, hop_type))

                if thumb_tasks:
                    ingested_thumbs = await asyncio.gather(*thumb_tasks, return_exceptions=True)
                    for cand in ingested_thumbs:
                        if isinstance(cand, NeighborCandidate) and is_unique_candidate(cand):
                            neighbors.append(cand)

                # Case B: Concurrent async batch scraping of web pages (concurrency limit = 5)
                sem = asyncio.Semaphore(5)

                async def _scrape_single_page(p_url: str, p_hop: str) -> List[NeighborCandidate]:
                    async with sem:
                        try:
                            # Strict 5.0-second timeout per page to guarantee snappy execution
                            p_images = await asyncio.wait_for(self.scrape_page_images(p_url, use_playwright=False), timeout=5.0)
                            def _img_rank(u: str) -> int:
                                u_low = u.lower()
                                if any(k in u_low for k in ["-1x-1", "master", "original", "highres", "hires"]):
                                    return 0
                                if any(k in u_low for k in ["2200", "2048", "1920", "1800", "1600", "1400", "1200"]):
                                    return 1
                                return 2
                            p_images.sort(key=_img_rank)
                            cands_for_page = []
                            for p_img in p_images[:6]:
                                if p_img not in seen_urls:
                                    seen_urls.add(p_img)
                                    c = await self.download_and_ingest(
                                        image_url=p_img,
                                        referer=p_url,
                                        source_hop=p_hop,
                                    )
                                    if is_unique_candidate(c):
                                        cands_for_page.append(c)
                            return cands_for_page
                        except Exception as scrape_err:
                            logger.debug("neighbor_crawler.page_scrape_skipped", url=p_url, error=str(scrape_err))
                            return []

                # Prioritize pages: authoritative wire domains first, then exact/visual matches
                def _page_priority(p_tuple: Tuple[str, str]) -> int:
                    u, h = p_tuple
                    dom = urlparse(u).netloc.lower()
                    if self.is_authoritative_domain(dom): return 0
                    if "exact" in h or "visual" in h: return 1
                    return 2

                pages_to_scrape.sort(key=_page_priority)
                # Take top 15 pages to scrape concurrently
                tasks = [_scrape_single_page(u, h) for u, h in pages_to_scrape[:15]]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in batch_results:
                    if isinstance(res, list):
                        neighbors.extend(res)
            except Exception as e:
                logger.warn("neighbor_crawler.lens_tool_failed", error=str(e))

        logger.info("neighbor_crawler.get_neighbors.finished", total_candidates=len(neighbors))
        return neighbors
