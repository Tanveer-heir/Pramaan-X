"""
Unit tests for CPH Multi-Service Gateway Fan-Out (/api/v1/investigate).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from src.api.main import app

client = TestClient(app)

def test_gateway_fanout_with_mocked_services():
    print("[*] Testing POST /api/v1/investigate with mocked microservices...")
    
    mock_ai_resp = MagicMock()
    mock_ai_resp.status_code = 200
    mock_ai_resp.json.return_value = {
        "media_type": "image",
        "prediction": "LIKELY_AUTHENTIC",
        "confidence": "HIGH",
        "probability": 0.15,
        "findings": []
    }
    
    mock_prnu_resp = MagicMock()
    mock_prnu_resp.status_code = 200
    mock_prnu_resp.json.return_value = {
        "ok": True,
        "result": {
            "device": {"label": "OnePlus 12R", "method": "prnu", "confidence": 0.96},
            "metadata": {"make": "OnePlus", "model": "CPH2609"}
        }
    }
    
    mock_source_res = {
        "target_media": {"domain": "local", "resolution": [1080, 720]},
        "patient_zero": {
            "domain": "bloomberg.com",
            "probability": 0.40,
            "media_url": "https://bloomberg.com/photo.jpg",
            "match_type": "exact_clone"
        },
        "candidate_sources": [
            {"domain": "bloomberg.com", "probability": 0.40}
        ],
        "graph_html_path": "data/graphs/astar_lineage_tree.html",
        "iterations": 4,
        "execution_time_sec": 8.5
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=[mock_ai_resp, mock_prnu_resp]), \
         patch("src.research.astar_attribution.astar_engine.AStarVisualAttributionEngine.trace_origin_async", new_callable=AsyncMock, return_value=mock_source_res):
        
        response = client.post(
            "/api/v1/investigate",
            json={"media_path": "data/sample_media/test.webp", "case_id": "TEST-CASE-001"}
        )
        assert response.status_code == 200
        data = response.json()
        
        print("[+] Unified Gateway Report Received!")
        print("    -> Case ID:", data.get("case_id"))
        print("    -> AI Content Detection:", data["ai_content_detection"].get("prediction"))
        print("    -> PRNU Device Forensics:", data["prnu_sensor_forensics"].get("result", {}).get("device", {}).get("label"))
        print("    -> Patient Zero Domain:", data["source_attribution"]["patient_zero"].get("domain"))
        
        assert data["case_id"] == "TEST-CASE-001"
        assert "ai_content_detection" in data
        assert "prnu_sensor_forensics" in data
        assert "source_attribution" in data
        assert data["source_attribution"]["patient_zero"]["domain"] == "bloomberg.com"

def test_gateway_resilience_when_services_offline():
    print("[*] Testing Gateway resilience when AI & PRNU services are offline...")
    
    mock_source_res = {
        "target_media": {"domain": "local"},
        "patient_zero": {"domain": "washingtonpost.com", "probability": 0.55},
        "candidate_sources": [],
        "graph_html_path": "data/graphs/astar_lineage_tree.html"
    }
    
    # Simulate connection error (offline microservices)
    with patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused")), \
         patch("src.research.astar_attribution.astar_engine.AStarVisualAttributionEngine.trace_origin_async", new_callable=AsyncMock, return_value=mock_source_res):
        
        response = client.post(
            "/api/v1/investigate",
            json={"media_path": "data/sample_media/test.webp"}
        )
        assert response.status_code == 200
        data = response.json()
        
        print("[+] Resilient response received despite offline sibling services!")
        print("    -> AI status:", data["ai_content_detection"]["status"])
        print("    -> PRNU status:", data["prnu_sensor_forensics"]["status"])
        print("    -> Patient Zero still isolated:", data["source_attribution"]["patient_zero"]["domain"])
        
        assert data["ai_content_detection"]["status"] == "offline"
        assert data["prnu_sensor_forensics"]["status"] == "offline"
        assert data["source_attribution"]["patient_zero"]["domain"] == "washingtonpost.com"

if __name__ == "__main__":
    test_gateway_fanout_with_mocked_services()
    test_gateway_resilience_when_services_offline()
    print("\n[+] All Multi-Service Gateway Fan-Out tests passed successfully!")
