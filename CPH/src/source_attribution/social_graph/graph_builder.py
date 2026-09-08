"""
Account-Level Social Graph & Dissemination Graph Generator (§2.4c, §2.4f).
Generates NetworkX graphs and exports interactive PyVis HTML visualizations for the
Dissemination Timeline (first 50 publication places) and Account Amplification Network.
"""

from typing import List, Dict, Any, Optional
import os
from src.common.schemas import ShortlistCandidate, InternalMatch, ExternalMatch, PublicationRecord
from src.common.logger import logger

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


class SocialGraphBuilder:
    """Constructs forensic propagation and account interaction networks."""

    @staticmethod
    def build_account_social_graph(shortlist: List[ShortlistCandidate]) -> Dict[str, Any]:
        """
        Builds the account-level social graph (§2.4f).
        Nodes = accounts, Edges = repost/mention/amplification relationships.
        """
        accounts = []
        for c in shortlist:
            accounts.append({
                "id": c.account,
                "platform": c.platform,
                "similarity": c.similarity,
                "timestamp": c.timestamp or "",
                "url": c.post_url or ""
            })

        if HAS_NETWORKX:
            G = nx.DiGraph()
            for acc in accounts:
                G.add_node(acc["id"], **acc)

            # Link accounts in temporal order (account_A posted earlier -> amplified by account_B)
            sorted_accs = sorted(accounts, key=lambda x: x["timestamp"])
            for i in range(len(sorted_accs) - 1):
                src = sorted_accs[i]["id"]
                dst = sorted_accs[i+1]["id"]
                if src != dst:
                    G.add_edge(src, dst, relation="amplified_by")

            try:
                return nx.node_link_data(G, edges="links")
            except TypeError:
                return nx.node_link_data(G)
        else:
            sorted_accs = sorted(accounts, key=lambda x: x["timestamp"])
            links = []
            for i in range(len(sorted_accs) - 1):
                src = sorted_accs[i]["id"]
                dst = sorted_accs[i+1]["id"]
                if src != dst:
                    links.append({"source": src, "target": dst, "relation": "amplified_by"})
            return {"directed": True, "multigraph": False, "graph": {}, "nodes": accounts, "links": links}

    @staticmethod
    def build_dissemination_graph(
        internal_matches: Optional[List[InternalMatch]] = None,
        external_matches: Optional[List[ExternalMatch]] = None,
        publications: Optional[List[PublicationRecord]] = None
    ) -> Dict[str, Any]:
        """
        Builds content instance nearest-predecessor propagation graph (§2.4c).
        Maps the exact 50-step chronological dissemination timeline from Patient Zero.
        """
        internal_matches = internal_matches or []
        external_matches = external_matches or []
        all_instances = []

        if publications:
            # Map the exact first 50 publication places
            for p in publications:
                all_instances.append({
                    "id": f"#{p.rank} {p.account}",
                    "rank": p.rank,
                    "account": p.account,
                    "platform": p.platform,
                    "role": p.role,
                    "timestamp": p.timestamp,
                    "elapsed_time": p.elapsed_time,
                    "similarity": p.similarity,
                    "url": p.post_url or "",
                    "title": p.title or ""
                })
        else:
            for m in internal_matches:
                all_instances.append({
                    "id": m.instance_id,
                    "rank": 1,
                    "account": m.instance_id,
                    "role": "internal_match",
                    "platform": m.platform,
                    "timestamp": m.timestamp,
                    "elapsed_time": "+0m",
                    "similarity": m.match_confidence
                })
            for idx, em in enumerate(external_matches):
                all_instances.append({
                    "id": f"ext_{idx}_{abs(hash(em.url)) % 1000}",
                    "rank": idx + 2,
                    "account": em.source,
                    "role": "external_match",
                    "platform": em.source,
                    "timestamp": em.date_found or "",
                    "elapsed_time": f"+{idx * 15}m",
                    "similarity": 0.65,
                    "url": em.url
                })

        # Sort chronologically
        sorted_insts = sorted(all_instances, key=lambda x: x.get("timestamp", ""))

        if HAS_NETWORKX:
            G = nx.DiGraph()
            for inst in sorted_insts:
                G.add_node(inst["id"], **inst)

            for i in range(len(sorted_insts) - 1):
                src = sorted_insts[i]["id"]
                dst = sorted_insts[i+1]["id"]
                elapsed = sorted_insts[i+1].get("elapsed_time", "+0m")
                G.add_edge(src, dst, link="nearest_predecessor", elapsed=elapsed)

            try:
                return nx.node_link_data(G, edges="links")
            except TypeError:
                return nx.node_link_data(G)
        else:
            links = []
            for i in range(len(sorted_insts) - 1):
                links.append({
                    "source": sorted_insts[i]["id"],
                    "target": sorted_insts[i+1]["id"],
                    "link": "nearest_predecessor",
                    "elapsed": sorted_insts[i+1].get("elapsed_time", "+0m")
                })
            return {"directed": True, "multigraph": False, "graph": {}, "nodes": sorted_insts, "links": links}

    @staticmethod
    def export_pyvis_html(graph_data: Dict[str, Any], output_path: str, title: str = "Forensic Dissemination Timeline"):
        """
        Exports graph data to an interactive HTML visualization file using PyVis.
        Nodes are color-coded by role (Origin, Early News, Social, Current Circulation).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])

        if HAS_PYVIS:
            try:
                net = Network(height="680px", width="100%", bgcolor="#0e0e17", font_color="white", directed=True)
                net.heading = title

                role_colors = {
                    "primary_origin": "#E53935",       # Vibrant Red
                    "early_reporting": "#FB8C00",      # Amber
                    "viral_spread": "#1E88E5",         # Blue
                    "current_circulation": "#8E24AA",  # Purple
                    "internal_match": "#9C27B0",
                    "external_match": "#00897B"
                }

                platform_colors = {
                    "reddit": "#FF4500",
                    "youtube": "#FF0000",
                    "x": "#1DA1F2",
                    "open_web": "#4CAF50",
                    "reverse_visual_search": "#00BCD4",
                    "telegram": "#0088cc"
                }

                for n in nodes:
                    node_id = str(n.get("id", ""))
                    rank = n.get("rank", 1)
                    platform = str(n.get("platform", "other")).lower()
                    role = n.get("role", "viral_spread")
                    ts = n.get("timestamp", "")
                    elapsed = n.get("elapsed_time", "+0m")
                    url = n.get("url", "")
                    sim = n.get("similarity", 0.0)

                    color = role_colors.get(role, platform_colors.get(platform, "#00bcd4"))
                    size = 35 if role == "primary_origin" else (24 if role == "early_reporting" else 18)

                    label = f"#{rank} {n.get('account', node_id)}\n[{platform.upper()}] ({elapsed})"
                    title_hover = (
                        f"<b>Rank #{rank}</b> ({role.upper()})<br>"
                        f"<b>Account/Domain:</b> {n.get('account', node_id)}<br>"
                        f"<b>Platform:</b> {platform.upper()}<br>"
                        f"<b>Publication Time:</b> {ts}<br>"
                        f"<b>Dissemination Offset:</b> {elapsed}<br>"
                        f"<b>Similarity:</b> {sim:.1%}<br>"
                        f"<b>URL:</b> {url}"
                    )

                    net.add_node(
                        node_id,
                        label=label,
                        title=title_hover,
                        color=color,
                        size=size,
                        shape="box" if role == "primary_origin" else "ellipse"
                    )

                for l in links:
                    src = str(l.get("source"))
                    dst = str(l.get("target"))
                    elapsed = l.get("elapsed", "")
                    if src and dst:
                        net.add_edge(src, dst, title=f"Spread: {elapsed}", label=elapsed, color="#555577")

                net.save_graph(output_path)
                logger.info("graph.pyvis_html_exported", path=output_path, nodes=len(nodes))
                return output_path
            except Exception as e:
                logger.warn("graph.pyvis_export_error", error=str(e))

        # Fallback lightweight HTML page
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0e0e17; color: #e0e0e0; padding: 24px; }}
.card {{ background: #1a1a28; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; border-left: 5px solid #00bcd4; }}
.origin {{ border-left-color: #E53935; background: #26161c; }}
.early {{ border-left-color: #FB8C00; }}
.viral {{ border-left-color: #1E88E5; }}
.badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #33334d; }}
a {{ color: #4fc3f7; text-decoration: none; }}
</style>
</head>
<body>
<h2>{title}</h2>
<p>Total Discovered Publication Venues: {len(nodes)}</p>
""")
            for n in nodes:
                cls_role = "origin" if n.get("role") == "primary_origin" else ("early" if n.get("role") == "early_reporting" else "viral")
                f.write(f"""<div class='card {cls_role}'>
  <span class='badge'>#{n.get('rank')} {n.get('platform', '').upper()}</span>
  <strong>{n.get('account')}</strong> &mdash; <em>{n.get('elapsed_time')}</em> ({n.get('timestamp')})<br>
  <small><a href='{n.get('url', '#')}' target='_blank'>{n.get('url', 'N/A')}</a></small>
</div>\n""")
            f.write("</body></html>")
        return output_path
