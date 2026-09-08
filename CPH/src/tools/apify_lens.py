"""
Apify Google Lens Visual Reverse Search Tool (§2.4b).
Provides programmatic reverse image search using Google Lens via Apify's cloud crawlers.
Bypasses Google anti-bot walls, CAPTCHAs, and IP rate limits with zero local browser overhead.
Extracts:
1. Exact matches (full images, high resolution originals, platform posts)
2. Visually similar matches
3. Web search pages, article headlines, and publication timestamps
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import re
import base64
import asyncio
import hashlib
from datetime import timedelta
from urllib.parse import urlparse
import httpx
from PIL import Image
from apify_client import ApifyClient
from src.common.config import settings
from src.common.logger import logger

class ApifyGoogleLensTool:
    """Enterprise-grade Google Lens reverse search via Apify Actor."""

    ACTOR_ID = "borderline/google-lens"

    TEMP_DIR = "data/cache/lens_temp"

    @classmethod
    def get_token(cls) -> Optional[str]:
        return os.getenv("APIFY_API_TOKEN") or getattr(settings, "APIFY_API_TOKEN", None)

    @classmethod
    async def search(
        cls,
        image_path_or_url: str,
        timeout_sec: int = 120,
        max_results: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Executes Google Lens reverse search via Apify.
        Runs synchronous ApifyClient in a worker thread to keep FastAPI/LangGraph non-blocking.
        """
        token = cls.get_token()
        if not token:
            logger.info("apify_lens.skipped_no_token", hint="Set APIFY_API_TOKEN in .env")
            return []

        logger.info("apify_lens.search.started", target=image_path_or_url)
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                cls._execute_search_sync,
                token,
                image_path_or_url,
                timeout_sec,
                max_results
            )
        except Exception as e:
            logger.warn("apify_lens.search_failed", error=str(e))
            return []

    @classmethod
    def clean_cdn_url(cls, url: str) -> str:
        """Strips CDN image resizing and quality downscaling parameters."""
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return url
        cleaned = re.sub(r'(\?|&)(resize|w|h|width|height|fit|quality|crop|zoom)=[^&]*', '', url)
        cleaned = re.sub(r'\?&', '?', cleaned)
        return cleaned.rstrip('?&')

    @classmethod
    def prepare_visual_search_payload(cls, image_path: str, max_dimension: int = 1600) -> str:
        """
        Downscales large master images (>2MB or >1600px) into a lightweight, crisp JPEG.
        Prevents cloud actor container OOM / Playwright 50MB buffer limit errors
        while preserving exact visual features for Google Lens.
        """
        try:
            if not os.path.isfile(image_path):
                return image_path
            file_size = os.path.getsize(image_path)
            with Image.open(image_path) as img:
                w, h = img.size
                if max(w, h) > max_dimension or file_size > 1_500_000:
                    scale = max_dimension / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    resized = img.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
                    opt_dir = os.path.join(cls.TEMP_DIR, "optimized")
                    os.makedirs(opt_dir, exist_ok=True)
                    h_md5 = hashlib.md5(image_path.encode()).hexdigest()
                    opt_path = os.path.join(opt_dir, f"opt_{h_md5}.jpg")
                    resized.save(opt_path, "JPEG", quality=86, optimize=True)
                    logger.info("apify_lens.optimized_payload", original=f"{w}x{h}, {file_size/1024/1024:.1f}MB", optimized=f"{new_size[0]}x{new_size[1]}")
                    return opt_path
        except Exception as e:
            logger.warn("apify_lens.payload_optimization_failed", error=str(e))
        return image_path

    @classmethod
    def _execute_search_sync(
        cls,
        token: str,
        image_path_or_url: str,
        timeout_sec: int,
        max_results: int
    ) -> List[Dict[str, Any]]:
        client = ApifyClient(token)

        run_input: Dict[str, Any] = {
            "searchTypes": ["all", "exact-match", "visual-match"],
            "language": "en"
        }

        local_file_to_process = None

        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            clean_url = cls.clean_cdn_url(image_path_or_url)
            dl_dir = os.path.join(cls.TEMP_DIR, "downloads")
            os.makedirs(dl_dir, exist_ok=True)
            dl_path = os.path.join(dl_dir, f"dl_{hashlib.md5(clean_url.encode()).hexdigest()}.jpg")

            download_success = False
            if not os.path.isfile(dl_path) or os.path.getsize(dl_path) == 0:
                for target_url in [clean_url, image_path_or_url]:
                    try:
                        with httpx.Client(timeout=12.0, follow_redirects=True) as h_client:
                            resp = h_client.get(target_url, headers={"User-Agent": "Mozilla/5.0"})
                            if resp.status_code == 200 and len(resp.content) > 1000:
                                with open(dl_path, "wb") as f:
                                    f.write(resp.content)
                                download_success = True
                                break
                    except Exception:
                        pass
            else:
                download_success = True

            if download_success and os.path.isfile(dl_path):
                local_file_to_process = dl_path
            else:
                # Fallback to direct URL
                run_input["imageUrls"] = [{"url": clean_url}]

        elif os.path.exists(image_path_or_url):
            local_file_to_process = image_path_or_url
        else:
            logger.warn("apify_lens.invalid_input", target=image_path_or_url)
            return []

        # If we have a local file, optimize it and pass as base64
        if local_file_to_process:
            optimized_path = cls.prepare_visual_search_payload(local_file_to_process)
            ext = os.path.splitext(optimized_path)[1].lower().lstrip(".")
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
            with open(optimized_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            run_input["imagesBase64"] = [f"data:{mime};base64,{b64}"]

        # Standard cloud actor call without artificial RAM or charge throttling
        run = client.actor(cls.ACTOR_ID).call(
            run_input=run_input,
            wait_duration=timedelta(seconds=timeout_sec),
        )

        if not run:
            logger.warn("apify_lens.run_returned_none")
            return []

        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "default_dataset_id", None)
        if not dataset_id:
            logger.warn("apify_lens.no_dataset_id")
            return []

        raw_items = list(client.dataset(dataset_id).iterate_items())
        parsed_results = cls._parse_dataset_items(raw_items, max_results=max_results)
        logger.info("apify_lens.search.completed", count=len(parsed_results))
        return parsed_results

    @classmethod
    def _parse_dataset_items(cls, items: List[Dict[str, Any]], max_results: int = 35) -> List[Dict[str, Any]]:
        # Bucket results by search_type
        bucketed_results: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
            "exact-match": [],
            "visual-match": [],
            "all": [],
        }

        for item in items:
            for stype, data in item.items():
                if not isinstance(data, dict) or "results" not in data:
                    continue
                target_bucket = bucketed_results.get(stype, bucketed_results["all"])
                for r in data.get("results", []):
                    if not r.get("error"):
                        target_bucket.append((stype, r))

        wire_keywords = [
            "bloomberg", "bwbx", "reuters", "apnews", "ap.org", "getty",
            "afp", "pti", "nyt", "nytimes", "washingtonpost", "wsj", "bbc",
            "politico", "abcnews", "cnn", "businessinsider", "epa.eu", "upi"
        ]

        def infer_domain(link: str, label: str) -> str:
            lbl = label.lower().strip()
            if "politico" in lbl: return "politico.com"
            if "business insider" in lbl or "insider" in lbl: return "businessinsider.com"
            if "bloomberg" in lbl: return "bloomberg.com"
            if "reuters" in lbl: return "reuters.com"
            if "associated press" in lbl or "ap news" in lbl: return "apnews.com"
            if "getty" in lbl: return "gettyimages.com"
            if "new york times" in lbl or "nyt" in lbl: return "nytimes.com"
            if "washington post" in lbl: return "washingtonpost.com"
            if "wall street journal" in lbl or "wsj" in lbl: return "wsj.com"
            if "abc news" in lbl or "abcnews" in lbl: return "abcnews.go.com"
            if "cnn" in lbl: return "cnn.com"
            if "bbc" in lbl: return "bbc.com"
            if "france 24" in lbl: return "france24.com"
            if "yahoo" in lbl: return "yahoo.com"
            if "sedaily" in lbl: return "sedaily.com"
            if "reddit" in lbl: return "reddit.com"
            if "facebook" in lbl: return "facebook.com"
            if "instagram" in lbl: return "instagram.com"
            if lbl == "x" or "twitter" in lbl or "x.com" in lbl: return "x.com"
            if "youtube" in lbl: return "youtube.com"

            if link and not ("google.com/goto" in link or "google.com/url" in link):
                try:
                    parsed = urlparse(link)
                    netloc = (parsed.hostname or parsed.netloc or "").lower().replace("www.", "")
                    if netloc and "google." not in netloc:
                        return netloc
                except Exception:
                    pass
            return "unknown"

        def is_wire_entry(link: str, label: str) -> bool:
            combined = f"{link} {label}".lower()
            return any(w in combined for w in wire_keywords)

        def build_hit(search_type: str, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            title = ""
            raw_link = ""
            snippet = ""
            source_label = r.get("source", "")
            thumbnail = r.get("thumbnail") or (r.get("image", {}).get("url") if isinstance(r.get("image"), dict) else None)
            width = None
            height = None

            if search_type == "all":
                search_obj = r.get("search", {})
                title = search_obj.get("title", "")
                raw_link = search_obj.get("href", "")
                snippet = search_obj.get("description", "")
            else:
                title = r.get("title", "")
                raw_link = r.get("link", "")
                snippet = f"Google Lens {search_type} match"
                img_size = r.get("imageSize", {})
                if isinstance(img_size, dict):
                    width = img_size.get("width")
                    height = img_size.get("height")

            if not raw_link:
                return None
            if not title:
                title = source_label or f"Google Lens {search_type} match"

            clean_link = raw_link
            platform = cls._infer_platform(clean_link, source_label)
            domain = infer_domain(clean_link, source_label)

            return {
                "platform": platform,
                "domain": domain,
                "source_label": source_label,
                "account": f"{source_label or platform} (google_lens)",
                "post_url": clean_link,
                "title": title[:160],
                "snippet": snippet[:200] if snippet else f"Discovered via Google Lens ({search_type})",
                "created_utc": None,
                "source": "google_lens_apify",
                "match_type": search_type,
                "thumbnail_url": thumbnail,
                "has_media": True,
                "width": width,
                "height": height,
            }

        hits: List[Dict[str, Any]] = []
        seen_urls = set()

        def add_candidate(search_type: str, r: Dict[str, Any]):
            hit = build_hit(search_type, r)
            if hit:
                # Deduplicate by post_url or thumbnail
                u_key = hit["post_url"]
                t_key = hit["thumbnail_url"]
                if u_key not in seen_urls and (not t_key or t_key not in seen_urls):
                    seen_urls.add(u_key)
                    if t_key:
                        seen_urls.add(t_key)
                    hits.append(hit)

        # 1. Wire domain priority: First ingest any authoritative wire results across ALL buckets
        for stype in ["all", "exact-match", "visual-match"]:
            for search_type, r in bucketed_results[stype]:
                raw_u = r.get("link") or (r.get("search", {}).get("href") if "search" in r else "")
                lbl = r.get("source", "")
                if is_wire_entry(raw_u, lbl):
                    add_candidate(search_type, r)

        # 2. Add visual & exact matches (up to 20 direct visual matches)
        vm_added = 0
        for stype in ["exact-match", "visual-match"]:
            for search_type, r in bucketed_results[stype]:
                if vm_added >= 20 or len(hits) >= max_results:
                    break
                raw_u = r.get("link", "")
                if raw_u not in seen_urls:
                    add_candidate(search_type, r)
                    vm_added += 1

        # 3. Add web article matches (up to 15 articles)
        all_added = 0
        for search_type, r in bucketed_results["all"]:
            if all_added >= 15 or len(hits) >= max_results:
                break
            raw_u = r.get("search", {}).get("href", "") if "search" in r else ""
            if raw_u not in seen_urls:
                add_candidate(search_type, r)
                all_added += 1

        return hits

    @classmethod
    def _infer_platform(cls, url: str, source_label: str) -> str:
        s_lower = source_label.lower()
        if "instagram" in s_lower: return "instagram"
        if "facebook" in s_lower: return "facebook"
        if "twitter" in s_lower or "x.com" in s_lower: return "x"
        if "reddit" in s_lower: return "reddit"
        if "youtube" in s_lower: return "youtube"

        u_lower = url.lower()
        if "instagram.com" in u_lower: return "instagram"
        if "facebook.com" in u_lower: return "facebook"
        if "twitter.com" in u_lower or "x.com" in u_lower: return "x"
        if "reddit.com" in u_lower: return "reddit"
        if "youtube.com" in u_lower or "youtu.be" in u_lower: return "youtube"

        return "news" if any(dom in u_lower for dom in ["hindu", "times", "ndtv", "reuters", "express", "firstpost", "oneindia"]) else "web"
