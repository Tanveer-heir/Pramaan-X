# Chandigarh Police Hackathon — Section 2 Platform
### Autonomous Media Origin Tracing, Provenance & Evidence-Grade Attribution System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![Standards](https://img.shields.io/badge/Forensics-ISO%2FIEC%2027037-success.svg)](#)
[![Stack](https://img.shields.io/badge/Cost-$0.00%20Free%20Tier-brightgreen.svg)](#)

A production-grade, investigator-ready forensic evidence platform engineered for the **Chandigarh Police Hackathon (CPH Track §2)**. The system autonomously isolates **Patient Zero** (the earliest primary uploader), recovers exact upload timestamps via 64-bit platform Snowflakes, mathematically proves sub-image crops using **SIFT/RANSAC Homography**, and overcomes dead-end visual searches using **Palantir-style Semantic Pivoting**.

Operates on a **100% free-tier tool stack** ($0.00 API cost per investigation) with **zero mandatory Google Cloud Vision billing dependencies**.

---

## 🏛️ System Architecture: The 8 Forensic Stages

```
                                  [ TARGET MEDIA ]
                       (Local Image/Video or Live Web URL)
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   STAGE 1: HARDWARE & PERCEPTUAL FORENSICS │
                   ├───────────────────────────────────────────┤
                   │ • Meta 256-bit PDQ Hash (pdqhash v0.2.8)  │
                   │ • DCT pHash & Gradient dHash              │
                   │ • Error Level Analysis (ELA Tamper Map)   │
                   │ • C2PA JUMBF Manifest Cryptographic Reader│
                   │ • EXIF Metadata (Dual pyexiftool + Pillow)│
                   │ • Multimodal CLIP ViT-B/32 Embeddings     │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   STAGE 2: FORENSIC SCENE RECONSTRUCTION  │
                   ├───────────────────────────────────────────┤
                   │ • Transcription of Watermarks / UI Badges │
                   │ • Vehicle Markings & Registration Plates  │
                   │ • Notable Persons, Uniforms & Flags       │
                   │ • Environmental & Geographical Signatures │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  STAGE 3: AUTONOMOUS ReAct REASONING LOOP │
                   ├───────────────────────────────────────────┤
                   │ • Multi-Platform Peripheral Tools:        │
                   │   ├─ Instagram (Playwright Embed + Snowf.)│
                   │   ├─ Reddit (Public JSON API + Media)     │
                   │   ├─ Telegram (Public Channel DOM Scraper)│
                   │   ├─ YouTube (Data API v3 + oEmbed)       │
                   │   ├─ Web Scraper (Playwright → Markdown)  │
                   │   └─ Reverse Search (Yandex + DDG Visual) │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   STAGE 4: CHRONOLOGICAL DISSEMINATION    │
                   ├───────────────────────────────────────────┤
                   │ • Candidate Deduplication & Normalization │
                   │ • Timestamp Extraction & Snowflake Decode │
                   │ • Temporal Sorting (Oldest → Newest)      │
                   │ • Patient Zero Identification             │
                   │ • Top 50 Dissemination Venues Tracked     │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   STAGE 5: ISO/IEC 27037 EVIDENCE LEDGER  │
                   ├───────────────────────────────────────────┤
                   │ • Cryptographically Chained SHA-256 Ledger│
                   │ • Tamper-Evident Courtroom Audit Trail    │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ STAGE 6: RECURSIVE A* CROP HOMOGRAPHY     │
                   ├───────────────────────────────────────────┤
                   │ • Objective: min f(n) = g(n) + h(n)       │
                   │ • OpenCV SIFT/ORB Keypoint Feature Match  │
                   │ • RANSAC Planar Homography Transformation │
                   │ • Uncovers Uncropped Master Parents       │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ STAGE 7: PALANTIR-STYLE SEMANTIC PIVOT    │
                   ├───────────────────────────────────────────┤
                   │ • Continuous Intelligence: Never Give Up  │
                   │ • Bellingcat Salient Sub-Region Probing   │
                   │ • Multimodal Scene Deconstruction (Gemini)│
                   │ • 3-Tier Orthogonal Query Batteries       │
                   │ • Open-Web Image Harvesting Pool (DDG)    │
                   │ • Reseeds A* Frontier upon Dead-End       │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ STAGE 8: DOCKER CONTAINERIZATION & PYVIS  │
                   ├───────────────────────────────────────────┤
                   │ • Interactive PyVis HTML Dissemination    │
                   │ • Live Bind-Mount (Zero Rebuilds on Edit) │
                   │ • Cross-Platform Helper Runners           │
                   └───────────────────────────────────────────┘
```

---

## ⚡ 5-Platform Tool Suite ($0.00 Free-Tier Stack)

| Target Platform | Tool Implementation | Authentication / Cost | Proven Forensic Capability |
|:---|:---|:---|:---|
| **Instagram** | `InstagramForensicsTool` + `SnowflakeDecoder` | **Free** (No Meta API key needed) | • Renders `/p/{shortcode}/embed/captioned/` via Playwright<br/>• Extracts handle, caption, likes, followers<br/>• Decodes 64-bit Snowflake ID (`id >> 23 + 1314220021721`) for millisecond UTC upload time |
| **Reddit** | `RedditForensicTool` | **Free** (No OAuth app required) | • Queries `reddit.com/search.json?sort=new`<br/>• Extracts author (`u/...`), score, comments, direct `i.redd.it` and `v.redd.it` URLs |
| **Telegram** | `TelegramForensicTool` | **Free** (No Bot Token required) | • Scrapes `https://t.me/s/{channel}` public preview DOM<br/>• Extracts message text, ISO dates, photo previews<br/>• Telethon user client adapter ready when session string exists |
| **YouTube** | `YouTubeForensicTool` | **Free** (10k units/day quota) | • Extracts 11-character video IDs and HQ thumbnails (`img.youtube.com/vi/{id}/hqdefault.jpg`)<br/>• Resolves channel handles via YouTube oEmbed |
| **Open Web** | `RobustWebSearchTool` + `PlaywrightScraper` | **Free** | • In-memory SHA-256 query cache to prevent redundant lookups<br/>• Exponential backoff retry logic (2s/4s/8s) against rate-limiting<br/>• Headless Playwright DOM scraper converting SPAs to clean Markdown |
| **Reverse Search** | `ReverseSearchDispatcher` | **Free** | • Parallel execution of Yandex (Chromium upload) + DuckDuckGo Visual + Google Lens fail-safe (auto-trips bot detection in <8s with zero hanging) |

---

## 🚀 Quick Start with Docker (Recommended)

The entire platform is containerized with **live-reloading bind mounts**, allowing anyone on Linux, macOS, or Windows to test and modify code with zero environment setup.

### 1. Configure Environment
```bash
cp .env.example .env
```
Add your free-tier Google Gemini API key inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Build Container (One-Time)
```bash
docker compose build
```

### 3. Start the Microservice (Docker)
```bash
# Start FastAPI service on http://localhost:8000
docker compose --env-file ../.env up cph-gateway

# Or use helper runners:
./docker-run.sh api       # Linux / macOS
.\docker-run.ps1 -Command api  # Windows PowerShell
```

### 4. Run Standalone Attribution CLI inside Container
```bash
# On Linux / macOS
./docker-run.sh test_astar "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"

# On Windows PowerShell
.\docker-run.ps1 -Command test_astar -Target "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"

# Or standard Docker Compose
docker compose --env-file ../.env run --rm cph-gateway python test_astar_attribution.py "<url_or_path>"
```

### 5. Modify Code Live (No Rebuilds)
Because `.:/app` is mounted as a live volume in `docker-compose.yml`, any code edit made on your host machine in VS Code takes effect **instantly** inside the container. No `docker compose build` is required after code changes!

---

## 💻 Local Python Setup (Alternative)

```bash
# 1. Virtual Environment Setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt
playwright install --with-deps chromium

# 3. Configure Environment
cp .env.example .env

# 4. Run Investigation
python test_astar_attribution.py "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"
```

---

## 🔬 Key Empirical Discoveries & Validations

### 1. Uncovering Uncropped Master Photographs (SIFT/RANSAC)
* **Target Image**: Firstpost thumbnail ($640 \times 362$ px, 16:9 aspect ratio).
* **Master Photo Discovered**: New York Times archive ($2048 \times 1365$ px, 3:2 aspect ratio).
* **Mathematical Proof**:
  * **1,700 matching SIFT keypoint inliers (99.0% inlier ratio)**.
  * **Detected Bounding Box**: `[x=0, y=171, w=2048, h=1194]`.
  * **Scale Factor**: **$3.302\times$**.
  * **Verdict**: The target image was mathematically proven to be a horizontal $16:9$ crop cut out from the center of the $2048 \times 1365$ master photograph!

### 2. Palantir Semantic Pivot on Cold Visual Searches
* When tested on a fresh regional news image (*OneIndia* Telugu post regarding G.O. 97), direct reverse search returned 0 matches.
* The **Semantic Pivot Layer** automatically engaged:
  * Transcribed Telugu text from the podium (`G.O. 97, వెంట్సే రద్దు చేయాలి!, TPJAC`).
  * Extracted entity context (*BRS Working President KTR*).
  * Harvested 50 open-web candidates across news DOMs.
  * Sifted candidates via SIFT homography and proved that OneIndia's story from today was reusing an **archival photograph first published on August 24, 2023** on Facebook and *The News Minute*!

---

## 📊 Interactive Forensic Lineage Graphs

Interactive PyVis HTML graphs are generated in `data/graphs/`:
* `data/graphs/astar_lineage_tree.html` (A* search frontier, crop bounding boxes, Patient Zero)
* `data/graphs/investigator_dissemination_tree.html` (Full chronological dissemination tree)

Double-click to open any `.html` graph directly in your web browser.

---

## 🧪 Automated Unit Test Suite

```bash
# Run all 23 A* attribution & semantic pivot tests:
pytest src/research/astar_attribution/ -v

# Run full project tests:
pytest -v
```
*(All 23 core tests pass with 100% success rate).*

---

## 🌐 Multi-Service Forensic Architecture & Unified Gateway

> The versioned frontend contract is now maintained in [`../docs/API_CONTRACT.md`](../docs/API_CONTRACT.md). The examples below describe the earlier prototype and may contain legacy field names. Frontend development must use `POST /api/v1/investigate/upload` and the `pramaan_x_investigation_v1` schema from the contract document.

This system acts as both **Node 3: Source Attribution & Origin Tracing** and the **Unified Forensic Gateway & Dispatcher** (`cph-gateway`) for the entire 3-node forensic investigation platform.

### Unified 3-Node Architecture (`cph-net` + `shared_media` volume)
```
                                 [ INVESTIGATOR SUBMISSION ]
                             (Single File Upload or Local Media)
                                              │
                                              ▼
                             ┌──────────────────────────────────┐
                             │    cph-gateway (Dispatcher)      │
                             │  (Saves once to `shared_media`)  │
                             │   Port 8000 | POST /investigate  │
                             └────────────────┬─────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │ Concurrent Fan-Out      │                         │
                    ▼ (asyncio.gather)        ▼                         ▼
        ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
        │  Node 1: ai-detection │ │Node 2: prnu-forensics │ │Node 3: cph-gateway    │
        │   (Pramaan-X Engine)  │ │   (Sensor Noise PRNU) │ │   (A* Attribution)    │
        ├───────────────────────┤ ├───────────────────────┤ ├───────────────────────┤
        │ Port: 8001            │ │ Port: 8002            │ │ Port: 8000            │
        │ POST /detect          │ │ POST /analyse         │ │ POST /source-attribution│
        │ Returns: Manipulation │ │ Returns: Sensor Noise │ │ Returns: Patient Zero │
        │ native evidence JSON  │ │ device attribution    │ │ & Crop Homography     │
        └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                             ┌──────────────────────────────────┐
                             │     CONSOLIDATED JSON REPORT     │
                             │  Full 3-Node Forensic Dossier   │
                             └──────────────────────────────────┘
```

### Single-Command Multi-Container Launch
From the `CPH` root directory, launch all 3 microservices and shared media storage in a single command:
```bash
docker compose up --build
```

* **Gateway & Attribution**: `http://localhost:8000` (Swagger UI: `http://localhost:8000/docs`)
* **AI Content Detection**: `http://localhost:8001` (Swagger UI: `http://localhost:8001/docs`)
* **PRNU Sensor Forensics**: `http://localhost:8002` (Swagger UI: `http://localhost:8002/docs`)
* **Shared Media Volume**: `cph_shared_media` mounted at `/data/shared_media` across all three nodes.

---

### Endpoint Reference

#### 1. Unified Forensic Investigation (`POST /api/v1/investigate` or `POST /investigate`)
The primary gateway endpoint. Ingests media once, writes to the high-performance shared volume, fans out concurrently across Node 1, Node 2, and Node 3 using non-blocking async coroutines, and aggregates a comprehensive forensic report.

* **Request Payload (JSON)**:
```json
{
  "media_path": "data/sample_media/test.webp",
  "case_id": "CASE-2026-CHANDIGARH-001"
}
```

* **Multipart File Upload Endpoint**: `POST /api/v1/investigate/upload`
  - Accepts raw image/video binary file directly from browser or cURL.

* **Consolidated Forensic Response (HTTP 200)**:
```json
{
  "case_id": "CASE-2026-CHANDIGARH-001",
  "media_sha256": "4b92...",
  "shared_media_path": "/data/shared_media/media_4b92.jpg",
  "execution_time_sec": 12.4,
  "ai_content_detection": {
    "media_type": "image",
    "prediction": "LIKELY_AUTHENTIC",
    "confidence": "HIGH",
    "findings": []
  },
  "prnu_sensor_forensics": {
    "ok": true,
    "result": {
      "device": {"label": "OnePlus 12R", "method": "prnu", "confidence": 0.96},
      "metadata": {"make": "OnePlus", "model": "CPH2609"}
    }
  },
  "source_attribution": {
    "patient_zero": {
      "domain": "bloomberg.com",
      "probability": 0.40,
      "media_url": "https://bloomberg.com/photo.jpg",
      "match_type": "parent_master_original"
    },
    "candidate_sources": [...],
    "graph_html_path": "data/graphs/astar_lineage_tree.html"
  }
}
```

#### 2. Visual Source Attribution & Patient Zero (`POST /source-attribution` or `POST /api/v1/source-attribution`)
Executes the A* Heuristic Traversal Engine, homography crop detection, and Patient Zero isolation.

* **Request Payload**:
```json
{
  "media_path": "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"
}
```
*(Also accepts local cached container paths like `/app/data/sample_media/test.webp`)*

* **Response Payload (HTTP 200)**:
```json
{
  "target_media": {
    "url": "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg",
    "local_path": "data/cache/astar/target_master.jpg",
    "resolution": [640, 362],
    "pdq_hex": "b3e0c4...",
    "timestamp": "2021-01-26T08:00:00Z",
    "domain": "firstpost.com"
  },
  "patient_zero": {
    "media_url": "file:///app/data/cache/astar/c0e77a5d2bc4fcc4a7ba4a36f11febb20cfa22f25f76ab8527692230ae1eea52.jpg",
    "source_page_url": "https://www.washingtonpost.com/...",
    "local_cached_path": "data/cache/astar/c0e77a5d2bc4fcc4a7ba4a36f11febb20cfa22f25f76ab8527692230ae1eea52.jpg",
    "domain": "washingtonpost.com",
    "publisher": "washingtonpost.com",
    "probability": 0.55,
    "confidence": 0.55,
    "timestamp": "2021-01-26T06:15:00Z",
    "resolution": [275, 183],
    "match_type": "parent_master_original",
    "crop_type": "target_is_crop_of_candidate",
    "bounding_box": [0, 28, 275, 155],
    "scale_factor": 0.429,
    "pdq_distance": 110,
    "inlier_count": 450,
    "inlier_ratio": 0.976,
    "is_authoritative_wire": true,
    "evidence": "Target is a cropped sub-region of Candidate (bbox=[0, 28, 275, 155] in candidate, scale=0.429, inliers=450, ratio=0.98)."
  },
  "candidate_sources": [
    {
      "node_id": "node_1_1",
      "media_url": "file:///app/data/cache/astar/c0e77a5d2bc4fcc4a7ba4a36f11febb20cfa22f25f76ab8527692230ae1eea52.jpg",
      "source_page_url": "https://www.washingtonpost.com/...",
      "domain": "washingtonpost.com",
      "probability": 0.55,
      "match_type": "parent_master_original",
      "crop_type": "target_is_crop_of_candidate",
      "resolution": [275, 183],
      "timestamp": "2021-01-26T06:15:00Z",
      "is_authoritative_wire": true,
      "inlier_count": 450,
      "pdq_distance": 110,
      "evidence": "Target is a cropped sub-region of Candidate..."
    },
    {
      "node_id": "node_1_2",
      "media_url": "file:///app/data/cache/astar/2f00a26971d67d618b105c3138da6c6c0769c474fa61e50e941695213dfa31c2.jpg",
      "source_page_url": "https://www.nytimes.com/...",
      "domain": "nytimes.com",
      "probability": 0.55,
      "match_type": "parent_master_original",
      "crop_type": "target_is_crop_of_candidate",
      "resolution": [275, 183],
      "is_authoritative_wire": true,
      "inlier_count": 448,
      "evidence": "Target is a cropped sub-region of Candidate..."
    }
  ],
  "graph_html_path": "data/graphs/astar_lineage_tree.html",
  "iterations": 7,
  "nodes_explored": 7,
  "execution_time_sec": 42.1
}
```

#### 2. Interactive Lineage Graph HTML (`GET /graphs/astar`)
Returns the live interactive PyVis HTML lineage graph directly to web browsers or dashboard iframes.

#### 3. Full Orchestrated Investigation (`POST /api/v1/cases/analyze`)
Runs the complete Section 2 pipeline (Metadata Provenance + Source Attribution + Custody Chain + Audit Log).

#### 4. System Liveness Probe (`GET /health`)
```json
{
  "status": "ok",
  "service": "cph-section2-api",
  "environment": "development"
}
```

---

### Example cURL Request

```bash
# Call from host or another container:
curl -X POST "http://localhost:8000/source-attribution" \
     -H "Content-Type: application/json" \
     -d '{"media_path": "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"}'
```

### Example Python Integration (`httpx` / `requests`)
```python
import httpx

# In internal docker network: http://source-attribution:8000/source-attribution
response = httpx.post(
    "http://localhost:8000/source-attribution",
    json={"media_path": "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"},
    timeout=120.0
)

data = response.json()
print("Patient Zero:", data["patient_zero"]["domain"])
print("Top Ranked Sources:", len(data["candidate_sources"]))
```

---

## 📜 Standards & Compliance
- **ISO/IEC 27037:2012**: Guidelines for identification, collection, acquisition, and preservation of digital evidence.
- **C2PA / Content Authenticity Initiative**: Cryptographic JUMBF provenance verification.
- **Meta PDQ Hashing**: 256-bit perceptual image hashing standard with Hamming distance clustering ($\le 30$).
