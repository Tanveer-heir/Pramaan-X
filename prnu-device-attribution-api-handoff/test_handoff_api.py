"""
Verify PRNU Attribution API endpoints.
"""
import urllib.request
import json
import time

def test_api():
    base_url = "http://127.0.0.1:8002"
    
    # 1. Test /health
    print("[*] Testing GET /health...")
    req = urllib.request.Request(f"{base_url}/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        print("[+] GET /health PASSED:", data.get("model_trained"))

    # 2. Test POST /analyse with JSON payload
    print("[*] Testing POST /analyse with JSON payload...")
    payload = json.dumps({
        "image_path": "dataset/device_A_synthetic_oneplus/original_test/test_000.jpg"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/analyse",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        result = json.loads(resp.read().decode("utf-8"))
        print("[+] POST /analyse PASSED!")
        dev = result.get("result", {}).get("device_attribution", {})
        print("    -> Predicted Device:", dev.get("prediction"))
        print("    -> Confidence:", dev.get("confidence"))
        print("    -> Primary Method:", dev.get("primary_method"))
        assert dev.get("prediction") == "device_A_synthetic_oneplus"

    print("\n[+] All PRNU Handoff API tests passed successfully!")

if __name__ == "__main__":
    test_api()
