import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

from unittest.mock import patch

def test_source_attribution_endpoint_mock():
    print("[*] Testing POST /api/v1/source-attribution with mock engine...")
    mock_result = {
        "target": {"domain": "local", "resolution": [1080, 720]},
        "target_media": {"domain": "local", "resolution": [1080, 720]},
        "patient_zero": {
            "domain": "bloomberg.com",
            "probability": 0.40,
            "media_url": "https://bloomberg.com/photo.jpg",
            "source_page_url": "https://bloomberg.com/article",
            "match_type": "exact_clone",
            "resolution": [1920, 1080],
            "evidence": "Exact clone via PDQ hash"
        },
        "candidate_sources": [
            {
                "domain": "bloomberg.com",
                "probability": 0.40,
                "media_url": "https://bloomberg.com/photo.jpg",
                "match_type": "exact_clone"
            }
        ],
        "graph_path": "data/graphs/astar_lineage_tree.html",
        "graph_html_path": "data/graphs/astar_lineage_tree.html",
        "iterations": 5,
        "execution_time_sec": 12.3
    }
    with patch("src.research.astar_attribution.astar_engine.AStarVisualAttributionEngine.trace_origin", return_value=mock_result):
        response = client.post(
            "/api/v1/source-attribution",
            json={"media_path": "data/sample_media/test.webp"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "target_media" in data
        assert "patient_zero" in data
        assert "candidate_sources" in data
        assert "graph_html_path" in data
        assert data["patient_zero"]["domain"] == "bloomberg.com"
        assert data["patient_zero"]["probability"] == 0.40
        print("[+] Mock POST /api/v1/source-attribution PASSED!")

        # Also test alias route /source-attribution
        alias_response = client.post(
            "/source-attribution",
            json={"media_path": "data/sample_media/test.webp"}
        )
        assert alias_response.status_code == 200
        print("[+] Mock POST /source-attribution alias PASSED!")

if __name__ == "__main__":
    test_source_attribution_endpoint_mock()
    print("[+] All API contract tests passed successfully!")
