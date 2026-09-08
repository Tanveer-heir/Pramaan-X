"""
Unit test for Pramaan-X API Node (api.py).
"""
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    print("[*] Testing GET /health...")
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    print("[+] GET /health PASSED:", data)
    assert data["status"] == "ok"

def test_detect_image():
    print("[*] Testing POST /detect with sample image...")
    # Use any available image in repo or sample
    sample_img = "D:/Strawberries/CPH/data/sample_media/test.webp"
    res = client.post(
        "/detect",
        json={"media_path": sample_img}
    )
    print(f"[*] Response Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print("[+] POST /detect PASSED!")
        print("    -> Media Type:", data.get("media_type"))
        print("    -> Prediction:", data.get("prediction"))
        print("    -> Confidence:", data.get("confidence"))
        print("    -> Probability:", data.get("probability"))
    else:
        print("[-] Response text:", res.text)

if __name__ == "__main__":
    test_health()
    test_detect_image()
    print("[+] All Pramaan-X API unit tests passed!")
