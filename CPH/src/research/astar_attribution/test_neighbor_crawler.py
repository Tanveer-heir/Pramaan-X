"""
Unit tests for VisualNeighborCrawler module (§2.4b / A* Attribution).
"""

import os
import sys
import io
import asyncio
import threading
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.research.astar_attribution.neighbor_crawler import (
    VisualNeighborCrawler,
    NeighborCandidate,
    AUTHORITATIVE_WIRE_DOMAINS,
)


class MockServerHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler serving test images and HTML pages with specific headers."""

    def log_message(self, format, *args):
        pass  # Suppress request logging in test output

    def do_GET(self):
        if self.path.startswith("/valid_photo_2023-04-18.jpg"):
            # Generate a 200x200 valid JPEG
            img = Image.new("RGB", (200, 200), color=(100, 150, 200))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("Last-Modified", "Wed, 19 Apr 2023 10:00:00 GMT")
            self.end_headers()
            self.wfile.write(img_bytes)

        elif self.path.startswith("/small_icon.jpg"):
            # Small 64x64 icon that must be filtered out (<150px)
            img = Image.new("RGB", (64, 64), color=(255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.end_headers()
            self.wfile.write(img_bytes)

        elif self.path.startswith("/favicon.ico"):
            self.send_response(200)
            self.send_header("Content-Type", "image/x-icon")
            self.end_headers()
            self.wfile.write(b"\x00\x00\x01\x00")

        elif self.path == "/test_page.html":
            html = """
            <!DOCTYPE html>
            <html>
            <head><title>Test News Article</title></head>
            <body>
                <h1>Breaking News</h1>
                <img src="/valid_photo_2023-04-18.jpg" alt="Valid Main Photo" width="400" height="300" />
                <img srcset="/valid_photo_2023-04-18.jpg 400w, /large_photo.jpg 800w" alt="Responsive" />
                <a href="/large_photo.jpg">High Resolution JPEG</a>
                <img src="/small_icon.jpg" width="64" height="64" class="icon" />
                <img src="/favicon.ico" class="favicon" />
            </body>
            </html>
            """
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif self.path.startswith("/large_photo.jpg"):
            # 320x240 valid photo
            img = Image.new("RGB", (320, 240), color=(50, 200, 50))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.end_headers()
            self.wfile.write(img_bytes)

        elif self.path.startswith("/yandex_photo.jpg"):
            # 250x250 valid photo from Yandex match
            img = Image.new("RGB", (250, 250), color=(200, 100, 50))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("Last-Modified", "Thu, 20 Apr 2023 12:00:00 GMT")
            self.end_headers()
            self.wfile.write(img_bytes)

        else:
            self.send_response(404)
            self.end_headers()


class MockYandexTool:
    """Mock reverse image search tool returning realistic structured search results."""

    @classmethod
    async def search(cls, image_path: str, timeout_sec: int = 25):
        return [
            {
                "platform": "web",
                "account": "yandex_match (web)",
                "post_url": "http://127.0.0.1:PORT_PLACEHOLDER/large_photo.jpg",
                "title": "Wire Syndication Match",
                "source": "yandex_reverse_image",
                "has_media": True,
            },
            {
                "platform": "web",
                "account": "yandex_match (web)",
                "post_url": "http://127.0.0.1:PORT_PLACEHOLDER/test_page.html",
                "title": "Article page with photo",
                "source": "yandex_reverse_image",
                "has_media": True,
            },
        ]


class TestVisualNeighborCrawler(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Start local HTTP server on an OS-assigned free port
        cls.server = HTTPServer(("127.0.0.1", 0), MockServerHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.crawler = VisualNeighborCrawler(cache_dir="data/cache/astar_test")

    def test_authoritative_domain_check(self):
        """Test wire service domain authority detection."""
        self.assertTrue(self.crawler.is_authoritative_domain("reuters.com"))
        self.assertTrue(self.crawler.is_authoritative_domain("www.reuters.com"))
        self.assertTrue(self.crawler.is_authoritative_domain("pictures.reuters.com"))
        self.assertTrue(self.crawler.is_authoritative_domain("apnews.com"))
        self.assertTrue(self.crawler.is_authoritative_domain("gettyimages.com"))
        self.assertTrue(self.crawler.is_authoritative_domain("pti.in"))
        self.assertTrue(self.crawler.is_authoritative_domain("nytimes.com"))
        self.assertFalse(self.crawler.is_authoritative_domain("randomblog.xyz"))
        self.assertFalse(self.crawler.is_authoritative_domain("example.com"))

    def test_timestamp_extraction(self):
        """Test timestamp parsing from headers and regex in URLs/filenames."""
        headers = {"last-modified": "Wed, 19 Apr 2023 10:00:00 GMT"}
        ts = self.crawler.extract_timestamp(headers, "http://example.com/img.jpg")
        self.assertIsNotNone(ts)
        self.assertIn("2023-04-19", ts)

        # Regex URL fallback
        ts_regex = self.crawler.extract_timestamp({}, "http://example.com/uploads/2021/11/05/photo.jpg")
        self.assertEqual(ts_regex, "2021-11-05T00:00:00Z")

        ts_compact = self.crawler.extract_timestamp({}, "http://example.com/news_20220815_hq.jpg")
        self.assertEqual(ts_compact, "2022-08-15T00:00:00Z")

    async def test_download_and_ingest_valid_image(self):
        """Test downloading a valid image, caching to sha256.jpg, and computing PDQ hash."""
        url = f"{self.base_url}/valid_photo_2023-04-18.jpg"
        candidate = await self.crawler.download_and_ingest(url, source_hop="test_hop")

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.width, 200)
        self.assertEqual(candidate.height, 200)
        self.assertEqual(candidate.source_hop, "test_hop")
        self.assertEqual(candidate.domain, f"127.0.0.1")
        self.assertFalse(candidate.is_authoritative_wire)
        self.assertIsNotNone(candidate.timestamp)
        self.assertTrue(os.path.exists(candidate.local_path))
        self.assertTrue(candidate.local_path.endswith(".jpg"))

        # Meta PDQ 256-bit hash must be a 64-char hex string
        self.assertEqual(len(candidate.pdq_hex), 64)
        int(candidate.pdq_hex, 16)  # Verify valid hex

    async def test_download_and_ingest_filters_small_icon(self):
        """Test that images < 150px are filtered out."""
        small_url = f"{self.base_url}/small_icon.jpg"
        candidate = await self.crawler.download_and_ingest(small_url)
        self.assertIsNone(candidate)

    async def test_download_and_ingest_filters_favicon(self):
        """Test that favicons are filtered out."""
        favicon_url = f"{self.base_url}/favicon.ico"
        candidate = await self.crawler.download_and_ingest(favicon_url)
        self.assertIsNone(candidate)

    async def test_scrape_page_images(self):
        """Test DOM scraping of <img> src, srcset, and <a> links while filtering small icons."""
        page_url = f"{self.base_url}/test_page.html"
        images = await self.crawler.scrape_page_images(page_url)

        self.assertGreater(len(images), 0)
        # Should include the valid photo and the high resolution link
        self.assertTrue(any("valid_photo" in u for u in images))
        self.assertTrue(any("large_photo" in u for u in images))
        # Should NOT include favicon or small icon
        self.assertFalse(any("favicon" in u for u in images))

    async def test_get_neighbors_end_to_end(self):
        """Test get_neighbors with parent_url DOM scraping and Yandex reverse search."""
        # Create dynamic mock tool with server port
        class DynamicMockYandex:
            port = self.port

            @classmethod
            async def search(cls, image_path: str, timeout_sec: int = 25):
                return [
                    {
                        "platform": "web",
                        "post_url": f"http://127.0.0.1:{cls.port}/yandex_photo.jpg",
                        "title": "Match Large Photo",
                        "source": "yandex_reverse_image",
                        "has_media": True,
                    }
                ]

        crawler = VisualNeighborCrawler(
            cache_dir="data/cache/astar_test",
            yandex_tool=DynamicMockYandex,
        )

        parent_page = f"{self.base_url}/test_page.html"
        dummy_query_image = os.path.join(self.crawler.cache_dir, "test_query.jpg")
        os.makedirs(self.crawler.cache_dir, exist_ok=True)
        Image.new("RGB", (200, 200), color=(10, 20, 30)).save(dummy_query_image)

        neighbors = await crawler.get_neighbors(
            candidate_image_path=dummy_query_image,
            parent_url=parent_page,
        )

        self.assertGreater(len(neighbors), 0)
        hops = [n.source_hop for n in neighbors]
        self.assertIn("page_dom_scrape", hops)
        self.assertIn("yandex_similar", hops)

        for n in neighbors:
            self.assertIsInstance(n, NeighborCandidate)
            self.assertGreaterEqual(n.width, 150)
            self.assertGreaterEqual(n.height, 150)
            self.assertEqual(len(n.pdq_hex), 64)
            self.assertTrue(os.path.exists(n.local_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
