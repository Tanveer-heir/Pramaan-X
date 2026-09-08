# 📷 Node 2: Camera Device Attribution & PRNU Sensor Noise Forensics
### Chandigarh Police Hackathon (Track §2) — Physical Hardware Attribution Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI/Uvicorn](https://img.shields.io/badge/API-FastAPI%20%7C%20REST-009688.svg)](#)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](#)
[![Forensics](https://img.shields.io/badge/Method-Sensor%20PRNU%20Noise-red.svg)](#)

A high-reliability digital forensics microservice engineered for the **Chandigarh Police Hackathon**. This service attributes suspect media to specific smartphone camera hardware using a dual-channel forensic pipeline: **Cryptographic Hardware EXIF Metadata** and **Photo-Response Non-Uniformity (PRNU) Sensor Noise Fingerprinting**.

---

## 🏛️ System Architecture

`
                                  [ SUSPECT IMAGE ]
                           (Local File Path or Binary Upload)
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   DUAL-CHANNEL DEVICE ANALYSIS   │
                         └────────────────┬────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
                  ▼                                               ▼
     ┌────────────────────────┐                     ┌────────────────────────┐
     │  CHANNEL 1: METADATA   │                     │   CHANNEL 2: SENSOR    │
     │   EXIF & JPEG Headers  │                     │       PRNU NOISE       │
     ├────────────────────────┤                     ├────────────────────────┤
     │ • Make / Model tags    │                     │ • Wavelet Noise Filter │
     │ • Software / OS Build  │                     │ • High-pass Residual   │
     │ • Lens & Exposure info │                     │ • 2D Cross-Correlation│
     │ • Fast (< 5ms check)   │                     │ • Hardware Fingerprint │
     └───────────┬────────────┘                     └───────────┬────────────┘
                 │                                              │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                         ┌─────────────────────────────────┐
                         │   FORENSIC ARBITRATION ENGINE   │
                         ├─────────────────────────────────┤
                         │ 1. Metadata check (EXIF match)  │
                         │ 2. Offline PRNU baseline match  │
                         │ 3. Multi-reference Lab match    │
                         │ 4. Graceful fallback resolution │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │    STRUCTURED JSON DOSSIER      │
                         │ (Device, Method, Confidence, UI)│
                         └─────────────────────────────────┘
`

---

## 🌐 Communication Contracts & REST API Reference

Default Base URL: http://localhost:8002 (or container internal: http://prnu-forensics:8002)

### 1. System Health & Liveness Probe
Inspects service liveness, model readiness, and the active dataset.

* **Endpoint**: GET /health or GET /status or GET /api/status
* **Response (HTTP 200)**:
`json
{
  "ready": true,
  "model_trained": true,
  "images": 84,
  "devices": 3,
  "device_labels": [
    "device_A_synthetic_oneplus",
    "device_B_synthetic_oneplus",
    "device_C_synthetic_oneplus"
  ],
  "dataset_layout": {
    "device_A_synthetic_oneplus": {
      "references": 12,
      "tests": 4,
      "facebook": 4,
      "instagram": 4,
      "reddit": 4
    }
  }
}
`

---

### 2. Device Attribution Analysis (Main Endpoint)
Analyzes an image and returns the hardware device attribution. Supports **both JSON payloads (path/URL)** and **multipart file uploads**.

* **Endpoints**: 
  - POST /analyse
  - POST /analyze
  - POST /api/analyze
  - POST /api/analyse

#### Option A: JSON Payload (Used by Gateway & Internal Services)
* **Headers**: Content-Type: application/json
* **Request Body**:
`json
{
  "media_path": "dataset/device_A_synthetic_oneplus/original_test/test_000.jpg"
}
`
*(Accepts keys: media_path, ile_path, image_path, or remote image_url / url)*

#### Option B: Multipart Form Upload (Used by Web Frontends)
* **Headers**: Content-Type: multipart/form-data
* **Form Field**: image (Binary JPEG, PNG, or WEBP file)

#### Standard Response Shape (HTTP 200):
`json
{
  "ok": true,
  "result": {
    "device": {
      "label": "OnePlus 12R",
      "method": "prnu",
      "confidence": 0.96
    },
    "device_attribution": {
      "prediction": "OnePlus 12R",
      "display_name": "OnePlus 12R",
      "primary_method": "prnu",
      "confidence": 0.96,
      "evidence": [
        "PRNU correlation score 0.7412 matches baseline fingerprint for OnePlus 12R."
      ]
    },
    "metadata": {
      "make": "OnePlus",
      "model": "CPH2609",
      "software": "Android 14",
      "exif_present": true,
      "dimensions": [4096, 3072],
      "color_space": "sRGB"
    },
    "rankings": [
      {
        "device": "OnePlus 12R",
        "correlation": 0.7412,
        "p_value": 0.0001
      },
      {
        "device": "Samsung Galaxy S23",
        "correlation": 0.0821,
        "p_value": 0.421
      }
    ],
    "evidence": [
      "PRNU sensor noise match established against pre-computed device baseline (correlation: 0.7412, confidence: 0.96)."
    ]
  }
}
`

#### Attribution Method Values & Meanings:
| Method Value | Meaning for Investigator | Recommended UI Badge |
|:---|:---|:---|
| prnu | Microscopic sensor noise correlated with physical hardware baseline. | 🟢 **PRNU Sensor Verified** (Green) |
| metadata | Attributed via cryptographic EXIF camera hardware tags. | 🔵 **Camera EXIF Verified** (Blue) |
| weak_features | Heuristic / compression quantization table match. | 🟡 **Tentative Heuristic** (Amber) |
| inconclusive | Metadata stripped and sensor noise unindexed or below threshold. | ⚪ **Inconclusive** (Gray) |

---

### 3. PRNU Fingerprint Lab (Ad-Hoc Hardware Matching)
Uploads multiple known reference photos from a seized suspect phone and matches an unseen query image in real-time.

* **Endpoint**: POST /api/prnu-match or POST /prnu-match
* **Headers**: Content-Type: multipart/form-data
* **Form Fields**:
  - eferences: Multiple JPEG/PNG photos taken by the suspect phone (e.g. 10–20 photos).
  - query: 1 unseen query image to verify.
  - eference_label (Optional query parameter): e.g. ?reference_label=Seized_iPhone_15_Pro

#### Response Shape (HTTP 200):
`json
{
  "ok": true,
  "result": {
    "device": {
      "label": "Seized_iPhone_15_Pro",
      "method": "prnu",
      "confidence": 0.92
    },
    "match": {
      "status": "match",
      "correlation": 0.684,
      "references_used": 12,
      "threshold": 0.15
    },
    "evidence": [
      "Query image residual correlates (0.684) with averaged reference fingerprint across 12 reference frames."
    ]
  }
}
`

---

## 🎨 Frontend Developer Implementation Guide

### TypeScript Interfaces
`	ypescript
export interface DeviceResult {
  label: string;
  method: 'prnu' | 'metadata' | 'weak_features' | 'inconclusive';
  confidence: number; // 0.0 to 1.0
}

export interface CameraMetadata {
  make?: string;
  model?: string;
  software?: string;
  exif_present: boolean;
  dimensions?: [number, number];
}

export interface DeviceAttributionResponse {
  ok: boolean;
  result: {
    device: DeviceResult;
    device_attribution: {
      prediction: string;
      display_name: string;
      primary_method: string;
      confidence: number;
      evidence: string[];
    };
    metadata: CameraMetadata;
    rankings?: Array<{
      device: string;
      correlation: number;
      p_value?: number;
    }>;
    evidence: string[];
    warning?: string;
  };
  error?: string;
}
`

### UI Display Guidelines for Frontend:
1. **Device Identification Card**:
   - Primary Title: Display esult.device.label.
   - Confidence Gauge: Render esult.device.confidence * 100 as a progress bar or percentage meter.
   - Method Tag: Render distinct color-coded badges based on esult.device.method:
     - prnu $\rightarrow$ Green badge (#10B981)
     - metadata $\rightarrow$ Blue badge (#3B82F6)
     - inconclusive $\rightarrow$ Gray badge (#6B7280)
2. **Hardware EXIF Panel**:
   - If esult.metadata.exif_present == true, display Make (esult.metadata.make), Model (esult.metadata.model), and Software.
   - If exif_present == false, display alert banner: *"Camera metadata was stripped by social platform; physical PRNU sensor noise matching engaged."*
3. **Evidence Audit Trail**:
   - Render esult.evidence as a bulleted checklist for courtroom affidavits.

---

## 🚀 How to Run

### Direct Python (1-Second Instant Start)
`powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Start FastAPI/HTTP server on port 8002
python -m uvicorn prnu_attribution.api:app --host 0.0.0.0 --port 8002
`

### Docker
`powershell
docker compose up --build
`
*(Runs container cph-prnu-forensics on http://localhost:8002)*

### Run Automated Unit Test
`powershell
python test_handoff_api.py
`
*(Asserts GET /health and POST /analyse return HTTP 200 with 95%+ confidence)*
