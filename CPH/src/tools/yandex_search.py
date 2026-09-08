"""
Self-Hosted Yandex Reverse Image Search Tool (§2.4b).
Automates Yandex reverse visual search using Playwright Chromium with stealth headers.
Supports direct local file upload and public image URLs.
Excels at facial identification, uncropped variations, and international syndication.
Zero commercial API keys required.
"""

from typing import List, Dict, Any, Optional
import os
import asyncio
from urllib.parse import quote
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from src.common.logger import logger

class YandexPlaywrightSearchTool:
    """Self-hosted Yandex visual search engine with file upload and anti-bot stealth."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    @classmethod
    async def search(cls, image_path_or_url: str, timeout_sec: int = 25) -> List[Dict[str, Any]]:
        """Unified entrypoint: auto-routes to file upload or URL reverse search."""
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            return await cls.search_by_url(image_path_or_url, timeout_sec=timeout_sec)
        elif os.path.exists(image_path_or_url):
            return await cls.search_by_file(image_path_or_url, timeout_sec=timeout_sec)
        else:
            logger.warn("yandex_search.invalid_target", target=image_path_or_url)
            return []

    @classmethod
    async def search_by_file(cls, image_path: str, timeout_sec: int = 25) -> List[Dict[str, Any]]:
        """Uploads a local image file directly to Yandex Images reverse search."""
        abs_path = os.path.abspath(image_path)
        logger.info("yandex_search.file_upload.started", path=abs_path)
        results = []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=cls.USER_AGENT,
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US"
                )
                page = await context.new_page()

                # Abort heavy assets to speed up execution and reduce captcha footprint
                await page.route("**/*yandex*metrika*", lambda r: r.abort())
                await page.route("**/*.{woff,woff2,ttf,svg}", lambda r: r.abort())

                await page.goto("https://yandex.com/images/", timeout=timeout_sec * 1000)
                await page.wait_for_timeout(1500)

                # Locate file input
                file_input = await page.query_selector("input[type='file']")
                if not file_input:
                    camera_btn = await page.query_selector(".cbir-icon, button[aria-label*='Search by image']")
                    if camera_btn:
                        await camera_btn.click()
                        await page.wait_for_timeout(1000)
                        file_input = await page.query_selector("input[type='file']")

                if not file_input:
                    logger.warn("yandex_search.file_input_not_found")
                    await browser.close()
                    return []

                # Upload file and wait for redirection
                await file_input.set_input_files(abs_path)
                await page.wait_for_timeout(5500)

                html = await page.content()
                await browser.close()

            results = cls._parse_yandex_html(html)
            logger.info("yandex_search.file_upload.completed", count=len(results))
        except Exception as e:
            logger.warn("yandex_search.file_upload.failed", error=str(e))

        return results

    @classmethod
    async def search_by_url(cls, image_url: str, timeout_sec: int = 25) -> List[Dict[str, Any]]:
        """Queries Yandex reverse search using an external image URL."""
        search_url = f"https://yandex.com/images/search?rpt=imageview&url={quote(image_url)}"
        logger.info("yandex_search.url_query.started", url=search_url)
        results = []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=cls.USER_AGENT,
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US"
                )
                page = await context.new_page()

                await page.route("**/*yandex*metrika*", lambda r: r.abort())
                await page.route("**/*.{woff,woff2,ttf,svg}", lambda r: r.abort())

                await page.goto(search_url, timeout=timeout_sec * 1000)
                await page.wait_for_timeout(3500)

                html = await page.content()
                await browser.close()

            results = cls._parse_yandex_html(html)
            logger.info("yandex_search.url_query.completed", count=len(results))
        except Exception as e:
            logger.warn("yandex_search.url_query.failed", error=str(e))

        return results

    @classmethod
    def _parse_yandex_html(cls, html: str) -> List[Dict[str, Any]]:
        """Parses Yandex Cbir rendered DOM for site matches and facial/entity tags."""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        # Extract tags / entity names safely
        tags = []
        for tag_el in soup.find_all(class_=lambda c: c and "Tags-Item" in c):
            txt = tag_el.get_text(strip=True)
            if txt:
                tags.append(txt)

        # Extract matching web pages
        for item in soup.find_all(class_=lambda c: c and any(k in c for k in ["CbirSites", "CbirItem", "SerpItem"])):
            a = item.find("a", href=True)
            if not a:
                continue
            href = a["href"]
            if not href.startswith("http") or "yandex.com" in href or "yandex.ru" in href:
                continue

            title_el = item.find(class_=lambda c: c and any(k in c for k in ["Title", "title", "description"]))
            title_text = title_el.get_text(" ", strip=True) if title_el else a.get_text(" ", strip=True)

            # Categorize platform
            plat = "web"
            if "instagram.com" in href: plat = "instagram"
            elif "twitter.com" in href or "x.com" in href: plat = "x"
            elif "reddit.com" in href: plat = "reddit"
            elif "vk.com" in href: plat = "vk"
            elif "youtube.com" in href or "youtu.be" in href: plat = "youtube"

            snippet_text = f"Tags: {', '.join(tags[:3])}" if tags else "Yandex Visual Match"

            results.append({
                "platform": plat,
                "account": f"yandex_match ({plat})",
                "post_url": href,
                "title": title_text[:140] if title_text else "Visual Match on Web",
                "snippet": snippet_text,
                "created_utc": None,
                "source": "yandex_reverse_image",
                "tags": tags[:5],
                "has_media": True
            })

        return results
