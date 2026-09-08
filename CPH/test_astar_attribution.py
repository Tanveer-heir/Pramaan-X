"""
Standalone CLI Test Harness for A* Visual Attribution Engine (§2.4b).
Chandigarh Police Hackathon - Visual Media Forensics

Usage:
    python test_astar_attribution.py "<target_image_url_or_path>"
"""

import os
import sys
import json

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.research.astar_attribution.astar_engine import AStarVisualAttributionEngine


def main():
    default_url = "https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg"
    target_input = sys.argv[1] if len(sys.argv) > 1 else default_url

    print("=" * 80)
    print("       A* HEURISTIC TRAVERSAL ENGINE - VISUAL MEDIA ATTRIBUTION")
    print("            Chandigarh Police Hackathon - Track §2.4b")
    print("=" * 80)
    print(f"[*] Target Input Media: {target_input}\n")

    engine = AStarVisualAttributionEngine(
        output_graph_path="data/graphs/astar_lineage_tree.html"
    )

    print("[*] Launching A* Visual Graph Traversal...")
    results = engine.trace_origin(target_input)

    # 1. Target Media Details & PDQ Hash
    target = results["target"]
    print("\n" + "-" * 80)
    print(" [1] TARGET MEDIA DETAILS & PERCEPTUAL HASH")
    print("-" * 80)
    print(f"  - Source URL / Path: {target['url']}")
    print(f"  - Local Cache Path : {target['local_path']}")
    print(f"  - Resolution       : {target['resolution'][0]} x {target['resolution'][1]} px")
    print(f"  - Publication Time : {target['timestamp'] or 'Unknown'}")
    print(f"  - Meta PDQ 256-bit : {target['pdq_hex']}")
    print(f"  - Source Domain    : {target['domain']}")

    # 2. Priority Queue Exploration Progress
    print("\n" + "-" * 80)
    print(" [2] PRIORITY QUEUE EXPLORATION PROGRESS & LINEAGE PATH")
    print("-" * 80)
    print(f"  Total Iterations    : {results['iterations']}")
    print(f"  Nodes Explored      : {results['nodes_explored']}")
    print(f"  Frontier Queue Size : {results['nodes_frontier']}")
    print(f"  Execution Time      : {results['execution_time_sec']} seconds")
    print("\n  Lineage Path (Query -> Patient Zero):")
    for idx, hop in enumerate(results["lineage_path"]):
        print(
            f"    Hop #{idx}: [{hop['role'].upper()}] {hop['domain']} "
            f"({hop['resolution'][0]}x{hop['resolution'][1]}) | "
            f"f={hop['f_score']:.1f} | Match: {hop['crop_type']}"
        )

    # 2.5. Palantir Semantic Pivot Layer
    pivot_info = results.get("semantic_pivot", {})
    print("\n" + "-" * 80)
    print(" [2.5] PALANTIR-STYLE SEMANTIC PIVOT & CONTINUOUS INTELLIGENCE LAYER")
    print("-" * 80)
    if pivot_info.get("activated"):
        print("  Status             : [ACTIVATED] Overcame visual dead-end via multimodal scene deconstruction.")
        decon = pivot_info.get("deconstruction") or {}
        ocr_list = decon.get("ocr_text", [])
        print(f"  Extracted OCR      : {', '.join(ocr_list) if ocr_list else 'None detected'}")
        print(f"  Event Hypothesis   : {decon.get('event_hypothesis', 'N/A')}")
        queries = decon.get("query_battery", [])
        print(f"  Query Batteries    : ({len(queries)} batteries)")
        for q_idx, q in enumerate(queries, 1):
            print(f"    - Battery #{q_idx} : {q}")
        print(f"  Harvested Images   : {pivot_info.get('harvested_count', 0)} candidates downloaded from open web")
        print(f"  Sifted Candidates  : {pivot_info.get('candidates_count', 0)} qualified & injected into A* Priority Queue")
    else:
        print("  Status             : [STANDBY] Direct reverse visual search yielded direct authoritative master matches.")

    # 3. Partial Match Ledger
    ledger = results["partial_match_ledger"]
    print("\n" + "-" * 80)
    print(f" [3] PARTIAL MATCH LEDGER ({len(ledger)} visual match/crop events recorded)")
    print("-" * 80)
    if not ledger:
        print("  No partial matches recorded in ledger.")
    else:
        for idx, entry in enumerate(ledger, 1):
            bbox_str = f"[{', '.join(map(str, entry['bounding_box']))}]" if entry['bounding_box'] else "None"
            print(f"  Match #{idx}:")
            print(f"    - URL          : {entry['url']}")
            print(f"    - Resolution   : {entry['resolution'][0]} x {entry['resolution'][1]} px")
            print(f"    - Crop Type    : {entry['crop_type']}")
            print(f"    - Bounding Box : {bbox_str}")
            print(f"    - Scale Factor : {entry['scale_factor']:.3f}x")
            print(f"    - SIFT Inliers : {entry['inlier_count']} (ratio: {entry['inlier_ratio']:.1%})")
            print(f"    - PDQ Distance : {entry['pdq_distance']} bits")
            print(f"    - Timestamp    : {entry['timestamp'] or 'Unknown'}")
            print(f"    - Details      : {entry['details']}")

    # 4. Isolated Primary Origin (Patient Zero)
    pz = results["patient_zero"]
    print("\n" + "=" * 80)
    print(" [4] ISOLATED PRIMARY ORIGIN (PATIENT ZERO)")
    print("=" * 80)
    if not pz:
        print("  [!] No external parent origin discovered. Query image is root/isolated.")
    else:
        print(f"  - Direct Media URL  : {pz.get('media_url') or pz.get('url')}")
        print(f"  - Source Web Page   : {pz.get('source_page_url') or 'N/A'}")
        print(f"  - Local Cached Path : {pz.get('local_cached_path') or pz.get('local_path')}")
        print(f"  - Origin Domain     : {pz['domain']}")
        print(f"  - Wire Authority    : {'YES (Authoritative Press/Wire)' if pz['is_authoritative_wire'] else 'No'}")
        print(f"  - Master Resolution : {pz['resolution'][0]} x {pz['resolution'][1]} px (Target: {target['resolution'][0]}x{target['resolution'][1]})")
        print(f"  - Earliest Timestamp: {pz['timestamp'] or 'Unknown'}")
        print(f"  - Relationship      : {pz.get('match_type') or pz.get('crop_type')}")
        if pz['bounding_box']:
            print(f"  - Target Crop BBox  : [{', '.join(map(str, pz['bounding_box']))}] inside Master")
        print(f"  - Homography Inliers: {pz['inlier_count']} (ratio: {pz['inlier_ratio']:.1%})")
        print(f"  - A* Traversal Cost : f={pz['f_score']:.1f}")
        print(f"  - Attribution Prob  : {pz['probability'] * 100.0:.1f}%")
        print(f"  - Forensic Evidence : {pz.get('evidence') or pz.get('details')}")

    # 5. Ranked Candidate Sources with Probabilities
    candidate_sources = results.get("candidate_sources", [])
    print("\n" + "-" * 80)
    print(f" [5] RANKED CANDIDATE SOURCES WITH PROBABILITIES ({len(candidate_sources)} candidates evaluated)")
    print("-" * 80)
    if not candidate_sources:
        print("  No candidate sources qualified.")
    else:
        for c_idx, c in enumerate(candidate_sources[:10], 1):
            is_pz = " [PATIENT ZERO]" if pz and c["media_url"] == pz.get("media_url") else ""
            m_url = c['media_url']
            if m_url.startswith("data:image/"):
                m_url = f"[Embedded Base64 Thumbnail - {len(m_url)} bytes]"
            p_url = c['source_page_url']
            if len(p_url) > 100 and "google.com/goto" in p_url:
                p_url = f"{c['domain']} (via Google Lens redirect)"
            print(f"  Rank #{c_idx}{is_pz}:")
            print(f"    - Probability    : {c['probability'] * 100.0:.1f}%")
            print(f"    - Media URL      : {m_url}")
            print(f"    - Web Page URL   : {p_url}")
            print(f"    - Domain         : {c['domain']} {'(Wire Authority)' if c['is_authoritative_wire'] else ''}")
            print(f"    - Resolution     : {c['resolution'][0]} x {c['resolution'][1]} px")
            print(f"    - Relationship   : {c['match_type']}")
            print(f"    - Publication    : {c['timestamp'] or 'Unknown'}")
            print(f"    - Evidence       : {c['evidence']}")

    # 6. Generated PyVis HTML Graph
    print("\n" + "-" * 80)
    print(" [6] INTERACTIVE PYVIS LINEAGE GRAPH")
    print("-" * 80)
    abs_graph = os.path.abspath(results["graph_path"])
    print(f"  - Interactive Lineage Tree Saved to: {abs_graph}")
    print(f"  - File URI: file:///{abs_graph.replace(os.sep, '/')}")
    print("=" * 80)


if __name__ == "__main__":
    main()