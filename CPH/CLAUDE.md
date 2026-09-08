# CLAUDE.md — Strict Engineering Standards & Project Context

> **Mandate**: All work in this repository must strictly adhere to the skills library in `skills/` (`.agent/skills/`) to operate at par with Anthropic's Claude Code standard of precision, autonomy, and verification.

---

## 1. Core Operating Principles
- **Explore Before Editing**: Never modify a file before reading it and checking related usages (`skills/codebase-exploration`).
- **Surgical Changes**: Use minimal diffs. Never rewrite entire files when editing specific functions (`skills/claude-code-core`).
- **Plan Complex Work**: For non-trivial or multi-file tasks, enter Plan Mode, detail milestones, and run an adversarial review before execution (`skills/architectural-planner`).
- **Verification is Mandatory**: "Looks done" is not accepted. Always verify via automated tests, linters, or standalone verification scripts (`skills/verification-engineering`).
- **Zero-Guesswork Debugging**: Follow the 5-step root-cause protocol: Reproduce -> Isolate -> Hypothesize -> Prove -> Surgical Fix (`skills/systematic-debugging`).
- **Safe Terminal Operations**: Use non-interactive flags, observe Windows PowerShell syntax, and guard against destructive commands (`skills/safe-terminal-ops`).
- **Containerization Discipline**: Use multi-stage builds, non-root users, layer caching, and healthchecks (`skills/docker-containerization`).
- **Robust Backend Architecture**: Strict Pydantic contracts, Alembic migrations, async queues for long-running jobs, and RBAC (`skills/backend-management`).
- **AI/ML Rigor**: Resilient device fallback (CUDA/CPU), inference mode memory hygiene, PDQ perceptual hashing, and quantitative evaluation (`skills/ai-ml-engineering`).

---

## 2. Project Architecture & Pipeline
This project implements **Section 2: Provenance, Origin Tracing & Platform Features**, strictly adhering to Figure 1 of Track 4 Documentation:

```text
Evidence Object (Section 1)
       │
       ▼
1. Provenance Check (C2PA, EXIF metadata consistency, PRNU device fingerprint, platform/compression fingerprint)
       │
       ▼
2. Origin Tracing (PDQ perceptual hash, external reverse search / Yandex / Google Vision, near-duplicate clustering, account-level attribution)
       │
       ▼
3. Investigator Dashboard & Evidence-Grade Report (Automated alerting, ISO-inspired hash-chained audit log, access control, scalability)
```

### Invariants:
- Never alter Section 1's manipulation verdict; provenance & origin tracing run downstream or side-by-side.
- Account-level attribution sits inside Origin Tracing, not as a disconnected module.
- Maintain tamper-evident hash-chained custody logs for all evidence artifacts.

---

## 3. Feedback & Memory Loop
Whenever a mistake is corrected or a project-specific constraint is clarified, immediately update this `CLAUDE.md` to permanently prevent regression (`skills/living-memory-claude`).

---

## 4. Key Constraints & Learned Architectural Invariants
- **Zero-Cost LLM Budget**: Limit LLM vision calls to strictly 1 call per run (max 300 tokens). Never poll or call LLM iteratively for candidate re-ranking.
- **Two-Stage Contextual Discovery**:
  - *Stage 1*: Visual reverse search discovers verified news headlines and wire photos.
  - *Context Enrichment*: `EventContextExtractor` isolates canonical entities (e.g. *Red Fort*, *Kisan Andolan*, *26 January 2021*).
  - *Stage 2*: Multi-query fan-out targets historical origin posts across Web, Reddit, YouTube, and X.
- **Calibrated Dual-Anchor Embedding**:
  - Raw CLIP cosine ranges are narrow ($0.16 - 0.32$ cross-modal; $0.45 - 0.55$ text-text baseline).
  - Piecewise linear calibration maps true incident candidates to $\ge 80\%$ similarity and suppresses unrelated noise to $\le 35\%$.
- **Windows Python 3.13 Runtime Hygiene**:
  - Keep `USE_TF=0` and `USE_TORCH=1` to avoid Protobuf descriptor conflicts between TensorFlow and Transformers.
  - Rely exclusively on PyTorch for CLIP embeddings (`CLIPModel`, `CLIPProcessor`).

