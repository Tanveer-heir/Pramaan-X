# CPH Section 2: Docker Setup & Execution Guide

This repository is fully containerized. Anyone on macOS, Linux, or Windows can run, test, and modify the entire **Origin Attribution & Forensic Intelligence Engine** with zero local environment setup (no manual Python, Playwright, ExifTool, or OpenCV installation needed).

---

## 🚀 Quick Start (3 Steps)

### 1. Set Up Environment
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Add your free-tier Google Gemini API key inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Build the Docker Image
```bash
docker compose build
```
*(This installs Python 3.11, ExifTool, headless OpenCV, Playwright Chromium, C2PA, and Meta PDQ hash libraries in a cached container layer).*

### 3. Run Investigation on Any Target Image
Run the **A\* Recursive Visual Attribution Engine**:
```bash
# On Linux / macOS
docker compose run --rm agent python test_astar_attribution.py "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"

# On Windows PowerShell
.\docker-run.ps1 -Command test_astar -Target "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"
```

Or run the full **ReAct Forensic Agent**:
```bash
docker compose run --rm agent python run_agent.py "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"
```

---

## 🛠️ How to Modify Code Live (Zero Rebuilds Needed)

The `docker-compose.yml` mounts the current repository directory as a live volume:
```yaml
volumes:
  - .:/app
```
**You can open any file in VS Code, PyCharm, or your favorite editor on your host machine, make edits, and save.**

When you run:
```bash
docker compose run --rm agent python test_astar_attribution.py "<target>"
```
your code changes are reflected **instantly** inside the container. You do **not** need to re-run `docker compose build` unless you modify `requirements.txt` or `Dockerfile`.

---

## 📊 Viewing Interactive HTML Lineage Trees

When an investigation completes, the engine exports interactive PyVis HTML lineage graphs:
- `data/graphs/astar_lineage_tree.html`
- `data/graphs/investigator_dissemination_tree.html`

Because `./data/graphs` is mapped to your host filesystem, you can double-click and open these `.html` files directly in your host web browser (Chrome, Safari, Edge, Firefox) to explore the interactive visual dissemination tree, node tooltips, and sub-crop bounding boxes.

---

## 🧪 Running Automated Unit Tests

Run the full automated forensic test suite inside the container:
```bash
# Run all 23 A* attribution & semantic pivot tests:
docker compose run --rm agent pytest src/research/astar_attribution/ -v

# Run entire repository tests:
docker compose run --rm agent pytest -v
```

---

## 💻 Interactive Container Shell

To inspect the environment or debug interactively:
```bash
docker compose run --rm agent bash
```

Inside the container shell, you have access to:
- `python` (Python 3.11 with all ML and CV libraries)
- `playwright` (headless Chromium)
- `exiftool` (metadata forensics)
- All project source code under `/app`

---

## 🌐 Launching the REST API & Dashboard

To start the FastAPI web service with live auto-reload:
```bash
docker compose up api
```
Then navigate in your host browser to:
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Interactive API: `http://localhost:8000/redoc`

---

## 📋 Convenient Shortcut Scripts

We have provided cross-platform shortcut runners:

### On Linux / macOS (`docker-run.sh`):
```bash
chmod +x docker-run.sh

./docker-run.sh build                                     # Build Docker image
./docker-run.sh test_astar "<url_or_path>"                # Run A* attribution
./docker-run.sh run_agent "<url_or_path>"                 # Run ReAct agent
./docker-run.sh test                                      # Run test suite
./docker-run.sh shell                                     # Interactive bash shell
./docker-run.sh api                                       # Start FastAPI service
```

### On Windows PowerShell (`docker-run.ps1`):
```powershell
.\docker-run.ps1 -Command build
.\docker-run.ps1 -Command test_astar -Target "<url_or_path>"
.\docker-run.ps1 -Command run_agent -Target "<url_or_path>"
.\docker-run.ps1 -Command test
.\docker-run.ps1 -Command shell
.\docker-run.ps1 -Command api
```
