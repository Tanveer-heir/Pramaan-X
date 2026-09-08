# Section 2: Provenance, Origin Tracing & Platform Features (v1)

> **Alignment note:** the official problem statement PDF's Figure 1 specifies a single
> linear pipeline: `Media Ingestion → AI Detection Engine → Provenance Check → Origin
> Tracing → Investigator Dashboard & Evidence Report`. This section is built directly
> around the **Provenance Check** and **Origin Tracing** boxes from that canonical
> diagram — not as a parallel track running alongside Section 1, but as the two stages
> that run after detection and before the final report. Use this document's structure,
> not any earlier ASCII/parallel-track sketch, as the source of truth for how the two
> sections connect.
>
> This section also folds in Expected Features 6–10 from the problem statement
> (Investigator Dashboard, Automated Alerting, Evidence-Grade Reporting, Security &
> Access Control, Scalability & Model Updatability) since they're platform-level
> concerns that sit downstream of both Section 1 and Section 2's outputs.

## 2.1 What it does

Given the structured evidence object Section 1 produces (see `Final_Section1.md`
§1.15), this module independently answers three further questions:

1. **Is there a valid, checkable authenticity signal for this file?** (Provenance Check)
2. **Where did this file first appear, and how has it spread since?** (Origin Tracing)
3. **How does an investigator actually see, act on, and trust all of this?** (Dashboard,
   alerting, reporting, security, scalability — Expected Features 6–10)

Nothing in this section changes or overrides Section 1's manipulation verdict. Both
verdicts are reported side by side in the final evidence report, exactly as Section 1
§1.13 already reserves a line for: `Provenance evidence: Handled independently by
Section 2`.

---

## 2.2 High-level architecture

```text
                  Evidence object from Section 1 (§1.15)
                                   |
                          PROVENANCE CHECK
                                   |
        +---------------+---------------+---------------+
        |               |               |               |
   C2PA / watermark  EXIF/metadata   PRNU device     Platform/compression
   credential check   consistency    fingerprint      fingerprint
        |               |               |               |
        +---------------+-------+-------+---------------+
                                 |
                     Provenance evidence object
                                 |
                          ORIGIN TRACING
                                 |
                  +--------------+--------------+
                  |                             |
         PDQ perceptual hash            Google Vision Web
         (internal dataset match)        Detection (external
                  |                       reverse search) / Yandex
                  +--------------+--------------+
                                 |
              Near-duplicate clustering (Hamming distance)
                                 |
              Nearest-predecessor dissemination graph
                                 |
                                 |         LLM media description
                                 |                 |
                                 |         Per-platform keyword gen
                                 |    (Reddit / X / Telegram / IG / FB / YouTube)
                                 |                 |
                                 |         Top-N search results/platform
                                 |                 |
                                 |         Semantic re-ranking (embeddings)
                                 |                 |
                                 |         Suspect-account shortlist
                                 |                 |
                                 |         Account-level social graph
                                 |                 |
                    (weighted by content age — §2.4d/e)
                                 |
                 Origin + propagation + account-attribution object
                                 |
                 +---------------------------------+
                 |                                 |
        Aggregation with Section 1's       ISO-inspired hash-chained
          evidence object (§1.15)          chain-of-custody log
                 |                          (per-stage audit, §2.5)
                 +---------------------------------+
                                 |
              INVESTIGATOR DASHBOARD & EVIDENCE REPORT
              (§2.6 — Features 6-10: dashboard, alerting,
               evidence-grade export, access control, scaling)
```

This matches the PDF's Figure 1 box order exactly: Provenance Check happens before
Origin Tracing, and both happen after detection, converging into one dashboard/report
stage — not two independent tracks. Account-level attribution and the social graph sit
inside Origin Tracing as a fourth workstream alongside PDQ matching and reverse search,
not as a separate box, since they answer the same underlying question ("where did this
spread from, and who's behind it").

---

## 2.3 Provenance Check

### Purpose

Corroborate or contradict Section 1's manipulation verdict using signals that are
independent of "does this look AI-generated" — a genuine video can still be
undocumented, stripped of metadata, or attributable to no known device, and a genuine
video's *provenance* being weak is itself useful evidence.

### 2.3a Credential check

* **C2PA Content Credentials** — validate via `c2patool`/`c2pa-rs`. Present/valid,
  present/invalid, or absent. Absence is neutral, not a manipulation signal, since most
  content in circulation was never signed.
* **SynthID watermark** — as of this writing, Google's public verification is
  waitlist-gated and not usable inside the hackathon window. Scope this as **roadmap**:
  the credential check must have a slot for it (`synthid_status: "not_checked"` in the
  output schema below), so no restructuring is needed if/when access opens up.

### 2.3b Metadata consistency

* Read embedded EXIF (camera model, GPS, timestamp, software tag) via `exiftool`.
* Flag inconsistency, not just absence — e.g. a "software tag" naming a known editing
  tool, or a timestamp that postdates the file's earliest-seen date from Origin Tracing
  (§2.4). Report which case applies rather than a single boolean.

### 2.3c PRNU device fingerprint

* Extract sensor noise via wavelet denoising (`PyWavelets`) and compare against
  reference fingerprints using `polimi-ispl/prnu-python`.
* Requires a reference set (≥~20-50 images from the same device) to compare against —
  meaningful primarily in an investigative context where a suspect device is available,
  or against your own mock dataset devices for the demo. Report `NOT_APPLICABLE` when no
  reference fingerprint exists, following the same convention Section 1 uses for missing
  modalities (§1.14) rather than a misleading neutral score.
* This reference set isn't static — confirmed device attributions grow it over time
  rather than staying pinned to the initial mock-dataset sizes above (see the living-
  system loop in Feature 10).

### 2.3d Platform/compression fingerprint

* Read quantization tables and compression signature via `exiftool`/`piexif`.
* Different platforms (WhatsApp, Instagram, etc.) recompress uploads with distinct,
  fairly consistent signatures — this hints at *which platform's pipeline* last touched
  a file, independent of which device captured it. Report as a best-guess platform label
  with confidence, not a hard claim.
* The signature database this compares against is likewise refreshed over time rather
  than frozen at build time — platforms change their recompression pipelines
  periodically, and the living-system loop (Feature 10) is what keeps this from going
  stale.

### Output of this stage

```json
{
  "provenance": {
    "c2pa": { "present": false, "valid": null },
    "synthid": { "status": "not_checked" },
    "metadata": { "consistent": true, "notes": [] },
    "prnu": { "status": "NOT_APPLICABLE", "reason": "no_reference_fingerprint" },
    "platform_fingerprint": { "predicted_platform": "whatsapp", "confidence": 0.71 }
  }
}
```

---

## 2.4 Origin Tracing

### Purpose

Identify the earliest known instance of a piece of content and map how it has spread —
matching PDF Expected Features 4 ("Reverse Search & Origin Identification") and 5
("Cross-Platform Propagation Tracing"), which the official diagram treats as one
combined "Origin Tracing" box.

### 2.4a Perceptual fingerprinting (internal)

* **PDQ hash** (`pdqhash` package, Meta's open-source hash) — replaces plain pHash for
  better resilience to crop/recompress/watermark transformations. Video: sample ~1
  frame/sec, hash each sampled frame.
* Compare against everything previously ingested (your mock/demo dataset, standing in
  for a real platform's prior ingestions). Hamming distance < ~10/64 bits (adjust
  threshold for PDQ's bit-length) = match.
* This internal dataset is meant to keep growing rather than stay fixed at the mock-
  dataset size in §2.7 — every new ingestion joins the pool it's matched against, and
  investigator-confirmed matches are weighted more heavily than unconfirmed ones (see
  the living-system loop in Feature 10).

### 2.4b Reverse search (external)

* **Google Cloud Vision — Web Detection** for the image case (confirmed active, 1,000
  free units/month). Bing Visual Search is not usable — Microsoft retired it in August
  2025, so it is intentionally absent from this architecture.
* Video: no reliable free live video-search API exists. Extract representative
  keyframes and run each through the image pipeline instead — state this limitation
  explicitly in the report rather than implying full video search.

### 2.4c Clustering & dissemination graph

* Group all matches under the distance threshold into one cluster.
* Sort by timestamp; earliest = inferred origin candidate.
* Build a directed graph (`networkx` + `pyvis`, or a lightweight D3/vis.js frontend):
  nodes = individual instances (platform, account, timestamp), edges = each item linked
  to its closest-in-time, most-similar predecessor.

### 2.4d Account-level attribution (LLM-guided cross-platform search)

Previously scoped as a stretch/roadmap item ("no viable technique at hackathon scale,
requires platform API partnership" — see the old §2.8). Reframed here as a search-and-
rank pipeline that doesn't need platform partnership, only public search surfaces:

1. **Describe** — run the media through a vision-language LLM to produce a text
   description (salient objects, setting, people/action, any visible text/logos). This
   reuses the same "one modality, one description" pattern as §2.3d's platform-label
   step, just applied to content rather than compression artifacts.
2. **Expand into per-platform queries** — prompt an LLM to turn that description into a
   small set of candidate search queries, phrased per platform's search conventions
   (hashtag-style for Instagram, short keyword for X, natural-language title-style for
   Reddit/YouTube).
3. **Search** — run those queries against Reddit, X/Twitter, Instagram, Facebook, and
   YouTube. Realistic access varies a lot by platform: Reddit's API and YouTube Data API
   are usable within hackathon constraints; X's search API sits behind a paid tier;
   Instagram/Facebook search access is the most restricted (Graph API is scoped to a
   business's own content, not open search) — scope these two as **best-effort /
   roadmap** rather than a guaranteed source, and say so explicitly in the report rather
   than silently returning empty results.
4. **Take the first N results per platform** (e.g. N≈20–50, tunable).
5. **Semantic re-rank** — embed each candidate result (its caption/post text/thumbnail
   description) and the original media's description into the same embedding space,
   rank by similarity, keep the top-k. This is the step that turns "N noisy keyword
   hits" into "a short, defensible shortlist" — report it as a ranked shortlist with
   confidence scores, never a single definitive account, matching the same
   non-committal framing used for PRNU (§2.3c) and the platform fingerprint (§2.3d).
6. Candidate accounts from the top-k results feed the account-level social graph
   (below) and the combined evidence report (§2.5) as `account_attribution`.

**Google Trends** — considered but scoped as a secondary corroboration signal, not a
search/attribution engine, since it returns aggregate topic interest, not accounts or
URLs. Its use here is narrow: cross-check whether search interest in the media's
inferred keywords/topic spiked around the `earliest_candidate` timestamp from §2.4c,
which either supports or weakens that timestamp's credibility. Optional, not
must-have.

### 2.4e Temporal weighting of search sources

Not all origin/attribution sources are equally useful at every point in a piece of
content's life, so the platform-search (2.4d) and reverse-search (2.4b) results are
combined with an age-dependent weighting rather than a fixed one:

* **New content** (short elapsed time between ingestion and the earliest-seen signal
  from §2.4c) — weight fast, platform-native search higher: Reddit, X, and Telegram
  surface newly-posted content well before general crawlers index it.
* **Older content** (longer elapsed time / already-indexed) — weight general reverse
  image/video search higher: Google Vision Web Detection and **Yandex** reverse image
  search (added alongside Google here — strong at face/visually-similar matching and
  at surfacing non-English-indexed reposts) become more reliable once a crawler has had
  time to index the propagated copies, while platform-native keyword search gets
  noisier as the content spreads and mutates.

This weighting feeds a combined confidence score on `origin_tracing` +
`account_attribution` rather than acting as a hard cutoff — both source families are
always queried, just blended differently.

### 2.4f Account-level social graph

A second graph alongside the existing content-instance dissemination graph (§2.4c):
nodes are **accounts** (not individual posts), edges are repost/share/mention
relationships inferred from the §2.4d semantic matches. Built with the same
`networkx`/`pyvis` (or D3/vis.js) approach so it can sit next to the propagation graph
in the dashboard (Feature 6) as a second, complementary view — "how the file moved"
versus "who moved it."

### Output of this stage

```json
{
  "origin_tracing": {
    "internal_matches": [
      { "instance_id": "demo_007", "platform": "instagram_sim", "timestamp": "..." }
    ],
    "external_matches": [
      { "url": "...", "source": "google_vision_web_detection", "date_found": "..." },
      { "url": "...", "source": "yandex_reverse_image", "date_found": "..." }
    ],
    "earliest_candidate": { "instance_id": "demo_001", "confidence": "moderate" },
    "dissemination_graph": "graph_id_ref_or_inline_json",
    "account_attribution": {
      "media_description": "LLM-generated text description",
      "search_queries": { "reddit": ["..."], "x": ["..."], "youtube": ["..."] },
      "shortlist": [
        { "platform": "reddit", "account": "...", "similarity": 0.83, "post_url": "..." }
      ],
      "sources_not_searched": ["instagram", "facebook"],
      "google_trends_corroboration": { "checked": true, "spike_aligned_with_earliest_candidate": null },
      "temporal_weighting": { "content_age": "new", "platform_search_weight": 0.7, "reverse_search_weight": 0.3 },
      "social_graph": "graph_id_ref_or_inline_json"
    }
  }
}
```

---

## 2.5 Combined evidence report (feeds Investigator Dashboard)

Section 1's output (§1.15) and this section's `provenance` + `origin_tracing` objects
(including `account_attribution`) are merged into one report, plus a chain-of-custody
block modeled loosely on ISO/IEC 27037 digital-evidence handling principles
(identification → collection → acquisition → preservation, each step attributable and
time-ordered) rather than a single opaque log line:

```json
{
  "detection": { "...": "Section 1 §1.15 output, unmodified" },
  "provenance": { "...": "§2.3 output" },
  "origin_tracing": { "...": "§2.4 output, including account_attribution" },
  "chain_of_custody": {
    "content_hash_sha256": "...",
    "ingested_at": "...",
    "pipeline_version": "section1-v3 + section2-v1",
    "audit_log": [
      { "stage": "ingested", "timestamp": "...", "input_hash": "...", "output_hash": "..." },
      { "stage": "detection_complete", "timestamp": "...", "model_version": "...", "input_hash": "...", "output_hash": "..." },
      { "stage": "provenance_complete", "timestamp": "...", "input_hash": "...", "output_hash": "..." },
      { "stage": "origin_tracing_complete", "timestamp": "...", "input_hash": "...", "output_hash": "..." },
      { "stage": "report_generated", "timestamp": "...", "input_hash": "...", "output_hash": "..." },
      { "stage": "investigator_confirmed", "timestamp": "...", "investigator_id": "...", "written_back_to": ["pdq_dataset", "prnu_reference_set"] }
    ]
  }
}
```

That last entry is the audit trail's link to the living-system loop (Feature 10) — it's
the record of *which* datasets a given case was allowed to influence, by *whom*, so the
feedback loop is itself auditable rather than a silent background process.

Each `audit_log` entry also carries a `prev_entry_hash` (hash-chained, ledger-style —
each entry's hash covers the previous entry) so any retroactive edit to an earlier
stage breaks the chain and is detectable, rather than relying on the log simply not
being tampered with. This is what makes the merge of Section 1's verdict and Section
2's provenance/origin/attribution findings into one report defensible as a single
forensic artifact rather than two reports stapled together: the chain itself is the
record of *how* the two parts were combined, not just *that* they were.

This single object is what Features 6–10 below are built on top of.

---

## 2.6 Expected Features 6–10 (platform-level requirements)

These are named explicitly in the problem statement but aren't detection or
attribution logic — they're what turns the pipeline above into something an
investigator can actually use. Addressing here since they sit downstream of both
sections' outputs.

### Feature 6 — Investigator Dashboard

* One screen per case: verdict (from §2.5) + suspicious-segment timeline (Section 1) +
  provenance status (§2.3) + content dissemination graph (§2.4c) + account-level social
  graph and suspect shortlist (§2.4d/f) shown together, not as separate disconnected
  views.
* Case management: list of submitted media, status (queued/processing/complete), and
  the ability to revisit a completed analysis without re-running it.
* Minimum viable for hackathon: a single-page app hitting the aggregation API, polling
  job status, rendering the five panels above from the §2.5 JSON.

### Feature 7 — Automated Alerting

* Trigger condition: high-confidence manipulation (`detection.overall.confidence` above
  a threshold) involving a monitored keyword/account/public figure.
* Hackathon scope: this genuinely requires a live-monitoring pipeline (watching
  incoming content against a watchlist) that's out of reach in 10 days without real
  platform access. Recommend scoping this as **architected but not built** — define the
  trigger schema and where it would hook into §2.5's output, demo it by manually
  submitting a "monitored" test case and showing the alert fire, rather than building
  real-time monitoring infrastructure.

### Feature 8 — Evidence-Grade Reporting

* Export §2.5's combined JSON as a rendered PDF: verdict, provenance findings,
  propagation timeline, methodology notes (including the explicit "mock dataset used
  for demo, live platform access required for production" framing), full audit trail.
* This is the natural home for the "state the mock-dataset limitation explicitly" advice
  from the original Section 2 review — it becomes a visible field in the exported
  report, not just something said out loud in the pitch.

### Feature 9 — Security & Access Control

* Authentication + role-based access control (investigator vs. admin) on the API layer.
* Audit logging of who accessed which case, when — this is a natural extension of the
  `chain_of_custody.audit_log` field already in §2.5, just extended to include user
  identity per action, not only pipeline stages.
* Hackathon scope: basic auth (e.g. JWT-based) with two roles is sufficient to
  demonstrate the concept; full RBAC granularity is a "production roadmap" line in the
  pitch, not a build target.

### Feature 10 — Scalability & Model Updatability

* Proposing a **Spring Boot** service as the secure, robust backend for the
  investigator-facing layer (auth/RBAC, case management, aggregation API, report
  export) — this is a change from treating FastAPI as the whole backend. The ML/CV
  workstreams (Section 1 detection, §2.3 provenance, §2.4 origin tracing +
  attribution) stay as independently-deployable Python services behind that layer, so
  the stack is polyglot by design rather than a single monolith: Spring Boot for the
  API/security surface, Python for the model-heavy stages, talking over REST/a
  message queue (Redis or similar per Section 3's queue foundations) rather than
  direct calls. Flag this explicitly against Section 3's earlier FastAPI-only framing
  so the two documents don't quietly disagree.
* **Containerization + microservices**: each stage (detection, provenance, origin
  tracing, account attribution, dashboard/API) ships as its own container, so any one
  stage (e.g. the account-attribution search fan-out, which is the most I/O-heavy)
  can be scaled independently of the others rather than scaling the whole pipeline.
  Docker Compose is enough to demo this at hackathon scale; note Kubernetes as the
  production path rather than building it in the 10-day window.
* Model updatability: version every model artifact (`model_metadata.model_version` in
  Section 1's schema, `pipeline_version` in §2.5) so a retrained detector can be
  deployed without breaking the ability to compare against historical case results.

#### Living-system loop — continuous fingerprint & model updating

The point above covers *versioning* a model swap; this covers *where the updated
data/models come from* — the system shouldn't be frozen at whatever it shipped with
for the demo. The loop:

1. **Trigger** — an investigator confirms (or rejects) a case's verdict from the
   dashboard (Feature 6). Only an authenticated confirmation counts, not a raw
   submission, tying this directly to Feature 9's RBAC so the training data can't be
   poisoned by an unauthenticated upload.
2. **Write-back** — a confirmed case feeds the specific datasets it touched:
   * PDQ internal dataset (§2.4a) — the confirmed instance is retained and its future
     matches are weighted above unconfirmed ones.
   * PRNU reference set (§2.3c) — confirmed device-attributed images are added to that
     device's reference set, so the fingerprint sharpens with use instead of staying
     at the initial ~20–50 image seed.
   * Platform/compression signature database (§2.3d) — confirmed platform labels
     refresh the signature set, tracking platforms' recompression pipelines as they
     change rather than assuming they're static.
   * Account-attribution re-ranker (§2.4d) — confirmed shortlist hits (and confirmed
     misses) become labeled pairs for periodically recalibrating the semantic
     re-ranking step.
   * Section 1's detection model — confirmed true/false positives accumulate into a
     retraining set for its next version.
3. **Refresh cadence** — batched/scheduled (e.g. nightly), not live per-case
   retraining: a single confirmation shouldn't immediately move a production model or
   reference set, since that's a drift/poisoning risk with only one data point.
4. **Pinning** — every dataset/model refresh produces a new version id (per the
   versioning point above), and an already-generated report stays pinned to the
   version it was produced against. Re-scoring a case against a newer version is an
   explicit "re-analyze" action, never something that happens silently underneath an
   existing report.
* Hackathon scope: build the write-back path itself (a confirm action that appends to
  a `confirmed_examples` store per §2.5's schema) and the version-pinning scheme;
  actual scheduled retraining can be demoed as a manual "trigger refresh" action
  rather than a live cron job, and stated as such in the pitch — the goal is to show
  the loop exists end-to-end, not to run continuous retraining in a 10-day window.

---

## 2.7 Mock/demo dataset

Per the team's decision to trim this for a testing-scope build (Jashan & Anand to help
finalize numbers): keep the full production-shaped spec below as the target for the
final demo, but for early pipeline testing, cut each device's set down to roughly a
quarter — **~12-15 boring-batch shots, ~5 real pics, 1 real video per device** — enough
to validate PRNU extraction and the clustering pipeline actually work end-to-end before
investing in the full 200-220 item collection effort.

**Full spec (final demo target), unchanged from the earlier review:**
Three devices (varying brands), each with:
* **Boring batch (50 pics)** — sky (no sun in frame), blank wall, a few slightly
  out-of-focus shots, normal auto mode, varied lighting — this is the clean PRNU
  reference material, not part of any repost simulation.
* **Real pics (20)** — a few with people/faces, a few outdoor/scenic, a couple with
  text/signs/posters.
* **Real videos (1-2, 10-30 sec)** — one talking, one just a scene.

Only the **real pics/videos** (not the boring batch) go through the platform-simulation
step (`ffmpeg` recompression at platform-typical bitrates, `exiftool` metadata
stripping, fabricated timestamps/platform/account tags) to build the repost chains that
§2.4's clustering and dissemination graph get tested against.

---

## 2.8 Hackathon Scope

### Must-have

* C2PA credential check (`c2patool`)
* EXIF/metadata consistency check
* PDQ perceptual hashing + internal clustering
* Google Cloud Vision Web Detection reverse search (image case)
* Nearest-predecessor dissemination graph (`networkx`/`pyvis`)
* Chain-of-custody hash + timestamp + audit log
* Combined evidence report (§2.5) merging Section 1 + Section 2 outputs
* Minimal investigator dashboard (Feature 6) rendering all four panels
* Basic auth + role separation (Feature 9, scoped-down)

### Strong additions

* PRNU device fingerprinting (`prnu-python`) against mock dataset devices
* Platform/compression fingerprinting (`exiftool`/`piexif`)
* Evidence-grade PDF export (Feature 8)
* Video reverse search via keyframe extraction
* **Account-level attribution** (§2.4d): LLM description → per-platform keyword
  search (Reddit + YouTube first, given API access; X/Telegram if budget allows) →
  semantic re-ranking → suspect shortlist + account-level social graph (§2.4f)
* Yandex reverse image search alongside Google Vision Web Detection (§2.4e)
* Age-based weighting between platform search and reverse search (§2.4e)
* Spring Boot backend + containerized/microservice deployment (Feature 10)
* **Living-system write-back path** (Feature 10): investigator-confirm action that
  appends to `confirmed_examples` and updates the PDQ/PRNU/platform-signature stores,
  plus the version-pinning scheme so existing reports never silently re-score

### Stretch goals / explicit roadmap items (state as such in the pitch, don't attempt)

* Instagram/Facebook search within account attribution — access is the most
  restricted of the five platforms (§2.4d); demo with Reddit/YouTube/X and note the
  other two as roadmap rather than silently omitting them
* SynthID watermark verification — currently waitlist-gated
* Live automated alerting (Feature 7) — requires real-time platform monitoring access
* Full RBAC granularity (Feature 9)
* Kubernetes-based orchestration — Docker Compose demonstrates the microservice split
  at hackathon scale; Kubernetes is the stated production path, not a build target
* Actual scheduled/automatic model retraining — demo the write-back path and a manual
  "trigger refresh" instead; a real cron-driven retraining cycle is a production
  concern, not a 10-day build target

---


