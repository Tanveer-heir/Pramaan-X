import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    print("[+] /health PASSED:", data)

def test_graphs_endpoint():
    res = client.get("/graphs/astar")
    print(f"[*] /graphs/astar status: {res.status_code}")
    # 200 if file exists, 404 if not run yet
    assert res.status_code in (200, 404)
    print("[+] /graphs/astar response handled properly")

if __name__ == "__main__":
    test_health()
    test_graphs_endpoint()
    print("[+] Basic endpoint tests passed successfully!")
