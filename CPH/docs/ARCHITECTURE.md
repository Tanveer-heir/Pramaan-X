# Section 2 Architecture & Data Flow

> **Domain**: Chandigarh Police Hackathon — Provenance Check, Origin Tracing & Evidence Reporting  
> **Standard**: Strictly follows Figure 1 of the Track 4 Documentation and `section-2-readme (1).md`.

---

## 1. Pipeline Sequence Diagram

```text
               +-------------------------------------------+
               | Evidence Object from Section 1 (Detector) |
               +-------------------------------------------+
                                     │
                                     ▼
                   STAGE 1: PROVENANCE CHECK (§2.3)
      ┌──────────────────┬──────────────────┬──────────────────┐
      │ C2PA Credential  │ EXIF Metadata    │ PRNU Camera      │ Platform Recomp.
      │ (c2patool)       │ Consistency      │ Fingerprint      │ Signature
      └──────────────────┴──────────────────┴──────────────────┘
                                     │
                                     ▼
                   STAGE 2: ORIGIN TRACING (§2.4)
      ┌─────────────────────────────────────┬──────────────────────────────────┐
      │ Internal Perceptual Matching        │ External Reverse Search          │
      │ (PDQ hash + Hamming Distance < 30)  │ (Google Cloud Vision Web Detect) │
      └─────────────────────────────────────┴──────────────────────────────────┘
                                     │
                                     ▼
               STAGE 3: ACCOUNT-LEVEL SOURCE ATTRIBUTION (§2.4d)
      ┌────────────────────────────────────────────────────────────────────────┐
      │ 1. Describe Media (Vision LLM)                                         │
      │ 2. Generate Platform Queries (Reddit / YouTube / X)                    │
      │ 3. Fan-out Search Crawl                                                │
      │ 4. Semantic Re-ranking (Text Embeddings + Similarity Scoring)          │
      │ 5. Account-Level Social Graph + Dissemination Graph (NetworkX)         │
      │ 6. Age-Dependent Temporal Weighting (§2.4e)                            │
      └────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
          STAGE 4: TAMPER-EVIDENT EVIDENCE REPORT & CUSTODY (§2.5)
      ┌────────────────────────────────────────────────────────────────────────┐
      │ ISO/IEC 27037 Hash-Chained Audit Ledger (prev_entry_hash chaining)     │
      │ Merged Evidence Object: Detection + Provenance + Origin + Attribution   │
      └────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
         STAGE 5: INVESTIGATOR DASHBOARD & EXPORT (Features 6–10)
      ┌────────────────────────────────────────────────────────────────────────┐
      │ FastAPI REST API / Single-Page Dashboard / PDF Evidence Export / RBAC  │
      └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

```text
src/
├── common/
│   ├── config.py              # Environment variables and settings
│   ├── schemas.py             # Canonical Pydantic schemas (§2.3, §2.4, §2.5)
│   └── logger.py              # Structured telemetry logger
│
├── source_attribution/        # [Lead + Teammate 1]
│   ├── reverse_search/        # Google Vision & Yandex connectors
│   ├── search_crawlers/       # Reddit, YouTube, X search crawlers
│   ├── llm/                   # Media description & platform query generator
│   ├── reranking/             # Semantic embedding re-ranker
│   ├── social_graph/          # NetworkX account & dissemination graphs
│   └── pipeline.py            # Master Source Attribution pipeline
│
├── fingerprinting/            # [Teammate 2]
│   ├── pdq/                   # PDQ hashing & Hamming distance clustering
│   ├── prnu/                  # Sensor noise camera attribution
│   ├── compression/           # Quantization table platform signatures
│   └── pipeline.py            # Master Fingerprinting pipeline
│
├── metadata_provenance/       # [Teammate 2]
│   ├── c2pa/                  # C2PA content credentials
│   ├── exif/                  # EXIF consistency & tampering inspector
│   ├── custody/               # ISO-inspired hash-chained audit ledger
│   └── pipeline.py            # Master Provenance pipeline
│
├── pipeline/
│   └── orchestrator.py        # End-to-end integration orchestrator
│
└── api/
    └── main.py                # FastAPI endpoints & case management
```
