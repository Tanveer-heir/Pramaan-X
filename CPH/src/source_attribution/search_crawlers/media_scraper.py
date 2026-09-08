"""
Media Scraper and Visual Verifier (§2.4).
Extracts actual media (images, video streams, thumbnails) from discovered post URLs
and validates visual congruence with the query media using CLIP / perceptual hashing.
"""

from typing import List, Dict, Any, Optional
import os
import re
import io
import asyncio
from urllib.parse import urlparse
from PIL import Image
import httpx
from src.common.logger import logger


class MediaScraper:
    """Extracts and verifies actual media attached to discovered web pages and social posts."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    @classmethod
    def extract_media_url_fast(cls, candidate: Dict[str, Any]) -> Optional[str]:
        """
        Extracts media URL directly from URL structure or existing metadata without network call.
        """
        url = candidate.get("post_url") or candidate.get("url") or ""
        
        # 1. Reverse visual search hits already contain direct image links
        if candidate.get("thumbnail_url"):
            return candidate["thumbnail_url"]
        if candidate.get("image"):
            return candidate["image"]

        # 2. YouTube URLs: extract video ID -> high-quality thumbnail
        if "youtube.com" in url or "youtu.be" in url:
            m = re.search(r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})', url)
            if m:
                vid = m.group(1)
                return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"

        # 3. Direct image/video URLs
        lower_url = url.lower()
        if any(lower_url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".mp4"]):
            return url

        # 4. Reddit media hosts in snippet or URL
        if "i.redd.it" in url or "v.redd.it" in url or "i.imgur.com" in url:
            return url

        return None

    @classmethod
    async def scrape_single_page_media(
        cls,
        client: httpx.AsyncClient,
        candidate: Dict[str, Any]
    ) -> Optional[str]:
        """Fetches page HTML and scrapes OpenGraph / Twitter image meta tags and suspect accounts."""
        url = candidate.get("post_url") or candidate.get("url") or ""
        if not url or not url.startswith("http"):
            return None

        # 1. YouTube URLs: fast oEmbed metadata extraction
        if "youtube.com" in url or "youtu.be" in url:
            try:
                oe_url = f"https://www.youtube.com/oembed?url={url}&format=json"
                r = await client.get(oe_url, timeout=2.5)
                if r.status_code == 200:
                    oe_data = r.json()
                    author_name = oe_data.get("author_name")
                    if author_name:
                        channel_handle = f"@{author_name.replace(' ', '')}"
                        candidate["account"] = channel_handle
                        candidate["suspect_account"] = channel_handle
                        logger.info("media_scraper.youtube_channel_identified", channel=channel_handle)
                    if oe_data.get("thumbnail_url"):
                        return oe_data["thumbnail_url"]
            except Exception:
                pass
            return cls.extract_media_url_fast(candidate)

        # Check fast media extraction
        fast_media = cls.extract_media_url_fast(candidate)
        if fast_media and candidate.get("suspect_account"):
            return fast_media

        # Only scrape HTML for open web news and articles
        parsed = urlparse(url)
        if parsed.netloc in ("reddit.com", "www.reddit.com", "x.com", "twitter.com"):
            return fast_media

        try:
            resp = await client.get(url, timeout=3.5)
            if resp.status_code == 200:
                html = resp.text[:30000]  # Read head and body portion
                domain = parsed.netloc or candidate.get("account", "web")

                # 2. Extract Media URL (og:image / twitter:image)
                media_url = fast_media
                if not media_url:
                    m_og = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
                    if not m_og:
                        m_og = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', html, re.I)
                    if m_og:
                        media_url = m_og.group(1).strip()
                    else:
                        m_tw = re.search(r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
                        if not m_tw:
                            m_tw = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']', html, re.I)
                        if m_tw:
                            media_url = m_tw.group(1).strip()

                # 3. Detect Embedded Viral Tweet Links in Fact-Checks & Articles
                m_tw_post = re.search(r'https?://(?:twitter\.com|x\.com)/([a-zA-Z0-9_]{3,25})/status/(\d+)', html)
                if m_tw_post and m_tw_post.group(1).lower() not in ("home", "explore", "search", "i"):
                    author = f"@{m_tw_post.group(1)}"
                    tweet_url = f"https://x.com/{m_tw_post.group(1)}/status/{m_tw_post.group(2)}"
                    candidate["suspect_account"] = author
                    candidate["origin_post_url"] = tweet_url
                    candidate["account"] = f"{author} (via {domain})"
                    logger.info("media_scraper.embedded_tweet_identified", suspect=author, post=tweet_url)

                # 4. Extract Suspect Social Account Mentioned in Article Text
                elif not candidate.get("suspect_account"):
                    suspect_accounts = re.findall(
                        r'(?:(?:post|tweet|screengrab|shared|uploaded|circulated|posted|claimed|by|handle|user)\s+(?:on\s+[\w\.]+\s+)?(?:by\s+)?|via\s+|from\s+)(@[a-zA-Z0-9_]{3,25})',
                        html,
                        re.I
                    )
                    filtered = [a for a in suspect_accounts if a.lower() not in ("@youtube", "@twitter", "@x", "@instagram", "@facebook", "@reddit", "@gmail")]
                    if filtered:
                        orig_acc = filtered[0]
                        candidate["suspect_account"] = orig_acc
                        prev_acc = candidate.get("account", domain)
                        if orig_acc not in prev_acc:
                            candidate["account"] = f"{orig_acc} (via {prev_acc})"
                        logger.info("media_scraper.suspect_account_identified", suspect=orig_acc, via=domain)
                    else:
                        m_creator = re.search(r'<meta[^>]+name=["\']twitter:creator["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
                        if m_creator:
                            creator = m_creator.group(1).strip()
                            if creator.startswith("@") and creator.lower() not in ("@youtube", "@twitter", "@x", "@reddit"):
                                candidate["account"] = f"{creator} ({candidate.get('account', '')})"

                return media_url
        except Exception:
            pass

        return fast_media

    @classmethod
    async def enrich_candidates_with_media(
        cls,
        candidates: List[Dict[str, Any]],
        max_scrape_concurrency: int = 25
    ) -> None:
        """
        Asynchronously enriches candidates with scraped media URLs and suspect accounts.
        Modifies candidate dictionaries in-place.
        """
        logger.info("media_scraper.enrich_start", total=len(candidates))

        # First pass: fast offline media & metadata extraction
        needs_scrape = []
        for c in candidates:
            url = c.get("post_url") or c.get("url") or ""

            # Check direct X post URL
            m_x = re.search(r'(?:twitter\.com|x\.com)/([a-zA-Z0-9_]{1,25})/status/(\d+)', url)
            if m_x and m_x.group(1).lower() not in ("home", "explore", "search", "i"):
                c["platform"] = "x"
                c["account"] = f"@{m_x.group(1)}"
                c["suspect_account"] = f"@{m_x.group(1)}"
                c["origin_post_url"] = f"https://x.com/{m_x.group(1)}/status/{m_x.group(2)}"

            # Extract suspect account from title or snippet text
            if not c.get("suspect_account"):
                text = f"{c.get('title', '')} {c.get('text', '')}"
                m_acc = re.search(
                    r'(?:(?:post|tweet|screengrab|shared|uploaded|circulated|posted|claimed|by|handle|user)\s+(?:on\s+[\w\.]+\s+)?(?:by\s+)?|via\s+|from\s+)(@[a-zA-Z0-9_]{3,25})',
                    text,
                    re.I
                )
                if m_acc:
                    acc = m_acc.group(1)
                    if acc.lower() not in ("@youtube", "@twitter", "@x", "@instagram", "@facebook", "@reddit", "@gmail"):
                        c["suspect_account"] = acc
                        prev = c.get("account", "")
                        if acc not in prev:
                            c["account"] = f"{acc} (via {prev})"

            fast_media = cls.extract_media_url_fast(c)
            if fast_media:
                c["media_url"] = fast_media
                c["has_media"] = True

            # If still missing media OR missing suspect account for an open web/YouTube hit, queue for scrape
            if (not c.get("has_media") or not c.get("suspect_account")) and c.get("platform") in ("open_web", "news", "youtube") and url:
                needs_scrape.append(c)

        # Second pass: async HTTP scraping for top candidates
        scrape_targets = needs_scrape[:max_scrape_concurrency]
        if scrape_targets:
            async with httpx.AsyncClient(headers=cls.HEADERS, follow_redirects=True) as client:
                tasks = [cls.scrape_single_page_media(client, c) for c in scrape_targets]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for c, res in zip(scrape_targets, results):
                    if isinstance(res, str) and res.startswith("http"):
                        c["media_url"] = res
                        c["has_media"] = True

        media_count = sum(1 for c in candidates if c.get("has_media"))
        logger.info("media_scraper.enrich_complete", total_with_media=media_count)

    @classmethod
    def verify_scraped_visuals(
        cls,
        candidates: List[Dict[str, Any]],
        query_image_path: str,
        embedder: Any,
        max_verify: int = 15
    ) -> None:
        """
        Downloads media thumbnails for top candidates and computes visual cosine similarity
        against the query media using CLIP image embeddings.
        Modifies candidate dicts in-place with verified similarity scores.
        """
        if not os.path.exists(query_image_path):
            return

        query_image_vec = embedder.embed_image(query_image_path)
        if query_image_vec is None:
            return

        # Select candidates that have media URLs, prioritized by similarity
        with_media = [c for c in candidates if c.get("media_url")]
        with_media.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        verify_pool = with_media[:max_verify]

        headers = {"User-Agent": cls.HEADERS["User-Agent"]}
        logger.info("media_scraper.verify_start", candidates_to_verify=len(verify_pool))

        for c in verify_pool:
            m_url = c["media_url"]
            try:
                # Fast download of thumbnail / media frame
                resp = httpx.get(m_url, headers=headers, timeout=4.0)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    cand_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    cand_vec = None
                    if embedder._clip_model is not None:
                        import torch
                        inputs = embedder._clip_processor(images=cand_img, return_tensors="pt").to(embedder.device)
                        with torch.no_grad():
                            feat = embedder._clip_model.get_image_features(**inputs)
                            feat = feat / feat.norm(dim=-1, keepdim=True)
                        cand_vec = feat.cpu().numpy()[0].tolist()

                    if cand_vec:
                        raw_sim = sum(a * b for a, b in zip(query_image_vec, cand_vec))
                        c["visual_cosine"] = round(raw_sim, 3)
                        
                        # Visual match threshold for news photo / video frame
                        if raw_sim >= 0.70:
                            c["media_verified"] = True
                            boosted_score = min(0.96, max(c.get("similarity", 0.0), raw_sim * 1.05))
                            c["similarity"] = round(boosted_score, 3)
                            title = c.get("title", "")
                            if not title.startswith("[VERIFIED MEDIA MATCH"):
                                c["title"] = f"[VERIFIED MEDIA MATCH: {c['similarity']:.1%}] {title}"
                            logger.info(
                                "media_scraper.visual_verified",
                                url=c.get("post_url", "")[:60],
                                visual_cosine=raw_sim,
                                final_sim=c["similarity"]
                            )
            except Exception as e:
                logger.debug("media_scraper.verify_failed", url=m_url[:40], error=str(e))
