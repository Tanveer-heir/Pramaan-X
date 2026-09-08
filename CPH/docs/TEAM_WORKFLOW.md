# Team Collaboration & Git Workflow Guide

> **Project**: Chandigarh Police Hackathon — Section 2 Platform  
> **Team Size**: 3 Engineers  
> **Rule**: GitHub-driven development from Day 1.

---

## 1. Workstream Ownership Matrix

| Part | Component | Module Directory | Assigned Owners | Current Status |
| :--- | :--- | :--- | :--- | :--- |
| **Part 1** | **Source Attribution & Origin Tracing**<br>Reverse search (Google/Yandex), Reddit/YouTube/X crawlers, LLM description & query expansion, semantic re-ranking, account-level social graph. | `src/source_attribution/` | **Lead (You) + Teammate 1** | **Active Sprint** |
| **Part 2** | **Fingerprinting**<br>PDQ perceptual hashing, Hamming clustering, PRNU sensor noise, platform/compression quantization tables. | `src/fingerprinting/` | **Teammate 2** | Scaffolded (Ready for next sprint) |
| **Part 3** | **Metadata & Provenance**<br>C2PA Content Credentials, SynthID slot, EXIF tamper consistency, ISO/IEC 27037 hash-chained chain-of-custody ledger. | `src/metadata_provenance/` | **Teammate 2** | Scaffolded (Ready for next sprint) |
| **Shared** | **Canonical Schemas, API & Orchestration**<br>Pydantic schemas, FastAPI backend, Docker compose, test suite. | `src/common/`<br>`src/pipeline/`<br>`src/api/` | **Shared / Lead** | Core Complete |

---

## 2. Git Branching Strategy

```text
main (Protected — Production & Demo Ready)
  │
  ├── feature/source-attribution        <-- Worked on by Lead + Teammate 1
  │     ├── sub-branches if needed (e.g. feature/crawlers-reddit, feature/social-graph)
  │
  ├── feature/fingerprinting-pdq-prnu   <-- Worked on by Teammate 2
  │
  └── feature/metadata-c2pa-custody     <-- Worked on by Teammate 2
```

### Golden Rules:
1. **Never push directly to `main`**. All changes arrive via Pull Requests (PRs).
2. **Contract Stability**: Do not modify `src/common/schemas.py` without alignment from all 3 members.
3. **Run tests before pushing**: Ensure `pytest` passes with 0 errors before opening a PR.

---

## 3. How to Share This Repo on GitHub (One-Time Setup by Lead)

Run these commands in your terminal to publish to GitHub:

### Option A: Using GitHub CLI (`gh`)
```bash
# Authenticate if needed
gh auth login

# Create private (or public) repository on GitHub and push
gh repo create cph-section2-platform --private --source=. --remote=origin --push
```

### Option B: Using GitHub Web UI
1. Go to [github.com/new](https://github.com/new) and create an empty repository named `cph-section2-platform` (do **not** check "Add README" or ".gitignore").
2. Link and push your local repo:
```bash
git remote add origin https://github.com/jashndx/cph-section-2.git
git branch -M main
git push -u origin main
```
3. In GitHub repo settings $\rightarrow$ **Collaborators**, invite your 2 teammates.

---

## 4. Teammate Onboarding (Share this with Teammates)

When your teammates clone the repo:

```bash
# 1. Clone repository
git clone https://github.com/jashndx/cph-section-2.git
cd cph-section-2

# 2. Create and activate Python virtual environment (Python 3.10+)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .

# 4. Copy configuration template
cp .env.example .env

# 5. Run tests to verify setup
pytest
```

---

## 5. Daily Development Workflow

```bash
# 1. Switch to your feature branch and pull latest changes
git checkout -b feature/source-attribution
git pull origin main

# 2. Make your targeted code edits in src/source_attribution/
# 3. Test locally
pytest tests/test_source_attribution.py

# 4. Commit using Conventional Commits
git add src/source_attribution/
git commit -m "feat(attribution): implement reddit crawler and semantic reranker"

# 5. Push to GitHub
git push -u origin feature/source-attribution

# 6. Open a Pull Request into main on GitHub for review
```
