"""
Interactive Demo & Test Runner for Source Attribution (§2.4).
Usage:
    python run_demo.py
    python run_demo.py path/to/your/image.jpg
"""

import os
import sys

# Ensure UTF-8 stdout/stderr on Windows to avoid cp1252 charmap encoding crashes with emojis/multilingual text
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Prevent HuggingFace Transformers from probing TensorFlow (which causes Protobuf mismatch)
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import asyncio
import webbrowser
from src.source_attribution.pipeline import SourceAttributionPipeline

async def main():
    # 1. Determine media path
    if len(sys.argv) > 1:
        media_path = sys.argv[1]
    else:
        media_path = "data/sample_media/sample_protest_rally.jpg"

    if not os.path.exists(media_path):
        print(f"[!] Warning: File '{media_path}' not found. Using fallback demo.")

    print("\n" + "=" * 75)
    print("  CHANDIGARH POLICE HACKATHON -- SOURCE ATTRIBUTION LIVE TEST")
    print("=" * 75)
    print(f"[*] Analyzing Media File: {media_path}")

    # 2. Execute pipeline
    pipeline = SourceAttributionPipeline()
    print("[*] Running Vision Description & Cross-Platform Search Fan-Out...")
    evidence = await pipeline.execute(media_path=media_path)

    # 3. Print Results
    attr = evidence.account_attribution
    print("\n--- [1] FORENSIC MEDIA DESCRIPTION ---")
    if hasattr(pipeline.descriptor, 'llm_succeeded') and pipeline.descriptor.llm_succeeded:
        print("  [Source: GEMINI 2.5 FLASH (LLM)]")
    else:
        print("  [Source: VISUAL-FIRST FALLBACK (LLM unavailable — derived from reverse image search)]")
    print(attr.media_description)

    # Display Stage 1 Discovered Incident Profile
    ev_profile = attr.identified_event or {}
    if ev_profile:
        print("\n--- [2] TWO-STAGE DISCOVERY: VERIFIED REAL-WORLD EVENT PROFILE ---")
        print(f"  Identified Event : {ev_profile.get('event_name', 'N/A').upper()}")
        print(f"  Historical Date  : {ev_profile.get('event_date', 'N/A')}")
        print(f"  Event Confidence : {ev_profile.get('confidence', 0.0):.1%}")
        entities = ev_profile.get('key_entities', [])
        if entities:
            print(f"  Key Entities     : {', '.join(entities)}")
        print(f"  Event Context    : {ev_profile.get('event_summary', 'N/A')}")

    print("\n--- [3] EVENT-SPECIFIC MULTI-QUERY EXPANSION (TEMPORAL FILTERS) ---")
    for platform, q_list in attr.search_queries.items():
        print(f"  [{platform.upper():<14}] ({len(q_list)} queries):")
        for q in q_list:
            print(f"    - \"{q}\"")

    print("\n--- [4] INFERRED PRIMARY ORIGIN (PATIENT ZERO) ---")
    earliest = evidence.earliest_candidate
    if earliest:
        print(f"  Primary Account  : {earliest.account or earliest.instance_id or 'Unknown'}")
        print(f"  Platform         : {earliest.platform.upper() if earliest.platform else 'UNKNOWN'}")
        print(f"  Published Time   : {earliest.timestamp} (UTC)")
        print(f"  Post URL / Source: {earliest.url or 'N/A'}")
        print(f"  Attribution Conf : {earliest.confidence.upper()}")
        if earliest.title:
            print(f"  Title / Context  : {earliest.title}")
    else:
        print("  No earliest origin candidate isolated.")

    # 5. First 50 Published Places & Propagation Timeline
    print("\n--- [5] FIRST 50 PUBLISHED PLACES & PROPAGATION TIMELINE ---")
    pubs = evidence.first_50_publications
    if pubs:
        header = f"  {'Rank':<5} | {'Elapsed':<8} | {'Role':<18} | {'Platform':<10} | {'Account / Attribution':<32} | {'Sim':<5} | {'URL / Post'}"
        print(header)
        print("  " + "-" * 128)
        for p in pubs:
            role_badge = p.role.replace("_", " ").upper()
            url_snippet = p.post_url or (p.title[:45] if p.title else "N/A")
            acc_str = (p.account[:29] + "..") if len(p.account) > 31 else p.account
            print(f"  #{p.rank:<4} | {p.elapsed_time:<8} | {role_badge:<18} | {p.platform:<10} | {acc_str:<32} | {p.similarity:.1%} | {url_snippet}")
    else:
        print("  No publication records available.")

    # 6. Active Live Circulation & Dissemination Metrics
    circ = evidence.circulation_summary or {}
    print("\n--- [6] ACTIVE LIVE CIRCULATION & DISSEMINATION METRICS ---")
    print(f"  Total Candidates Scanned : {circ.get('total_candidates_scanned', 'N/A')}")
    print(f"  Verified Publications    : {circ.get('verified_publications_tracked', 'N/A')}")
    print(f"  First 50 Tracked Venues  : {circ.get('first_50_places_count', 'N/A')}")
    print(f"  Dissemination Span       : {circ.get('total_span_hours', 'N/A')} hours")
    print(f"  Dissemination Velocity   : {circ.get('dissemination_velocity_posts_per_hour', 'N/A')} posts/hour")
    platforms_breakdown = circ.get("active_platforms_breakdown", {})
    if platforms_breakdown:
        breakdown_str = ", ".join(f"{k}: {v}" for k, v in platforms_breakdown.items())
        print(f"  Platform Distribution   : {breakdown_str}")

    print("\n--- [7] TOP SUSPECT SHORTLIST (MEDIA CONGRUENCE & ACCOUNT ATTRIBUTION) ---")
    for idx, c in enumerate(attr.shortlist[:8], 1):
        if c.post_text and "[VERIFIED VISUAL MATCH" in c.post_text:
            tag = "[VERIFIED VISUAL MATCH]"
        elif c.has_media:
            tag = "[MEDIA ATTACHED]       "
        else:
            tag = "[TEXT REPORT]          "
        print(f"  #{idx} {tag} [{c.platform.upper():<10}] Account: {c.account:<24} | Sim: {c.similarity:.1%}")
        if c.post_url:
            print(f"       Direct URL: {c.post_url}")

    print("\n--- [8] TEMPORAL WEIGHTING & PLATFORM DYNAMICS ---")
    tw = attr.temporal_weighting
    print(f"  Content Age Category : {tw.content_age.upper()}")
    print(f"  Platform Search Weight: {tw.platform_search_weight:.0%}")
    print(f"  Reverse Search Weight : {tw.reverse_search_weight:.0%}")

    print("\n--- [9] INTERACTIVE 50-NODE FORENSIC GRAPHS (HTML) ---")
    social_graph_path = os.path.abspath("data/graphs/social_graph.html")
    dissem_graph_path = os.path.abspath("data/graphs/dissemination_graph.html")
    print(f"  Account Social Graph     : {social_graph_path}")
    print(f"  50-Node Dissemination Tree: {dissem_graph_path}")

    # Automatically open the dissemination graph in the browser
    if os.path.exists(dissem_graph_path):
        print("\n[*] Opening interactive 50-Node Dissemination Graph in your default web browser...")
        try:
            webbrowser.open(f"file://{dissem_graph_path}")
        except Exception:
            pass

    print("\n" + "=" * 75)
    print("  TEST COMPLETED SUCCESSFULLY")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
