import React, { useState } from 'react';
import { 
  GitBranch, 
  ExternalLink, 
  Download, 
  CheckCircle2, 
  Maximize2,
  Copy,
  Check,
  Share2,
  Image as ImageIcon
} from 'lucide-react';
import { useCaseStore } from '../../store/caseStore';
import { getArtifactUrl } from '../../services/gateway';

interface DagNode {
  id: string;
  type: 'patient_zero' | 'hop' | 'target';
  platform: string;
  title: string;
  sourceUrl: string;
  thumbnailUrl?: string;
  timestamp: string;
  latencyFromRoot: string;
  resolution: string;
  sha256: string;
  pdqDistance: number;
  notes: string;
  cropBadge?: string;
}

interface CrawledCandidate {
  id: string;
  url: string;
  domain: string;
  thumbnailUrl: string;
  resolution: string;
  timestamp: string;
  role: string;
  roleColor: string;
  pdqDistance: number;
  inliers: number;
  inlierRatio: number;
  wireAuthority: boolean;
  boundingBox?: string;
  cropType: string;
}

export const Tab2Provenance: React.FC = () => {
  const { activeDossier, showToast } = useCaseStore();
  const [viewMode, setViewMode] = useState<'graph' | 'timeline' | 'gallery'>('timeline');
  const [selectedNodeId, setSelectedNodeId] = useState<string>('node-root');
  const [showIframeModal, setShowIframeModal] = useState<boolean>(false);
  const [copiedHash, setCopiedHash] = useState<boolean>(false);

  const sourceResult = activeDossier.modules.source_attribution?.result;
  const pZero = sourceResult?.patient_zero;

  const nodes: DagNode[] = [
    {
      id: 'node-root',
      type: 'patient_zero',
      platform: 'Accredited Wire Service',
      title: 'Canonical Seed Origin (Patient Zero)',
      sourceUrl: pZero?.url || 'https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg',
      thumbnailUrl: '/media/master_uncropped.jpg',
      timestamp: pZero?.timestamp || '2021-01-26T04:18:22Z',
      latencyFromRoot: '0h 00m (Original Broadcast)',
      resolution: pZero?.resolution ? `${pZero.resolution[0]} x ${pZero.resolution[1]}` : '4000 × 2667 (AP Wire Master)',
      sha256: '4f8a1e2c9d0b3e5a7f1c8d2e4a6b9c0d1e3f5a7b8c9d0e1f2a3b4c5d6e7f8a9b',
      pdqDistance: 0,
      notes: 'Authoritative wire publication. Uncropped master photograph with complete optical EXIF parameters and linear spectral distribution.',
      cropBadge: 'Uncropped Master (100% Canvas)',
    },
    {
      id: 'node-hop1',
      type: 'hop',
      platform: 'Telegram Channel',
      title: 'Propagation Hop 1: Telegram Broadcast',
      sourceUrl: 'https://t.me/LeakOps_India/8912',
      thumbnailUrl: '/media/hop1_telegram.jpg',
      timestamp: '2021-01-26T10:30:14Z',
      latencyFromRoot: '+6h 11m 52s',
      resolution: '1280 × 720 (Transcoded)',
      sha256: '7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d',
      pdqDistance: 6,
      notes: 'Initial high-velocity channel repost. Transcoded by platform compression, EXIF metadata stripped.',
      cropBadge: 'Platform Transcode',
    },
    {
      id: 'node-hop2',
      type: 'hop',
      platform: 'Twitter / X',
      title: 'Propagation Hop 2: Public Post & Sub-Crop',
      sourceUrl: 'https://x.com/JanAndolanUpdates/status/13540294102',
      thumbnailUrl: '/media/hop2_twitter.jpg',
      timestamp: '2021-01-26T14:48:00Z',
      latencyFromRoot: '+10h 29m 38s',
      resolution: '1360 × 906 (Sub-Cropped)',
      sha256: '8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c',
      pdqDistance: 10,
      notes: 'Crop applied to isolate central subject. Perimeter dimensions reduced by 15.8% with JPEG re-quantization.',
      cropBadge: 'Sub-Crop (15.8% Reduction)',
    },
    {
      id: 'node-target',
      type: 'target',
      platform: 'Seized Evidence',
      title: 'Seized Court Exhibit',
      sourceUrl: activeDossier.media.filename,
      thumbnailUrl: '/media/suspect_crop.webp',
      timestamp: '2025-01-14T03:42:10Z',
      latencyFromRoot: '+29h 18m Delta',
      resolution: '2200 × 1467 (Suspect File)',
      sha256: activeDossier.media.sha256,
      pdqDistance: pZero?.pdq_distance || 12,
      notes: 'The evidence exhibit submitted in the current FIR. Matches Hop 2 crop boundary with secondary re-quantization artifacts.',
      cropBadge: 'Seized Exhibit (FIR Exhibit A1)',
    },
  ];

  const crawledCandidates: CrawledCandidate[] = [
    {
      id: 'cand-pz',
      url: 'https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg',
      domain: 'firstpost.com / AP Wire Master',
      thumbnailUrl: '/media/master_uncropped.jpg',
      resolution: '4000 × 2667',
      timestamp: '2021-01-26T04:18:22Z',
      role: 'Patient Zero Primary Master',
      roleColor: 'border-red-500/40 text-red-400 bg-red-950/40',
      pdqDistance: 0,
      inliers: 2410,
      inlierRatio: 0.991,
      wireAuthority: true,
      boundingBox: '[0, 0, 4000, 2667] (Uncropped Canvas)',
      cropType: 'Canonical Origin',
    },
    {
      id: 'cand-target',
      url: 'file://evidence/seized_court_exhibit.webp',
      domain: 'Seized Evidence Exhibit (FIR A1)',
      thumbnailUrl: '/media/suspect_crop.webp',
      resolution: '2200 × 1467',
      timestamp: '2025-01-14T03:42:10Z',
      role: 'Court Exhibit (Target Query)',
      roleColor: 'border-cyan-500/40 text-cyan-400 bg-cyan-950/40',
      pdqDistance: 12,
      inliers: 1384,
      inlierRatio: 0.975,
      wireAuthority: false,
      boundingBox: '[260, 220, 1630, 1020] (Partial Crop)',
      cropType: 'Target is Crop of Candidate',
    },
    {
      id: 'cand-hop2',
      url: 'https://x.com/JanAndolanUpdates/status/13540294102',
      domain: 'x.com / Twitter Feed',
      thumbnailUrl: '/media/hop2_twitter.jpg',
      resolution: '1360 × 906',
      timestamp: '2021-01-26T14:48:00Z',
      role: 'Viral Social Hop (Sub-Crop)',
      roleColor: 'border-purple-500/40 text-purple-400 bg-purple-950/40',
      pdqDistance: 10,
      inliers: 864,
      inlierRatio: 0.892,
      wireAuthority: false,
      boundingBox: '[160, 140, 1200, 766] (-15.8% Perimeter)',
      cropType: 'Sub-Image Match',
    },
    {
      id: 'cand-hop1',
      url: 'https://t.me/LeakOps_India/8912',
      domain: 't.me / Telegram Channel',
      thumbnailUrl: '/media/hop1_telegram.jpg',
      resolution: '1280 × 720',
      timestamp: '2021-01-26T10:30:14Z',
      role: 'Broadcast Repost (Transcoded)',
      roleColor: 'border-blue-500/40 text-blue-400 bg-blue-950/40',
      pdqDistance: 6,
      inliers: 1120,
      inlierRatio: 0.945,
      wireAuthority: false,
      boundingBox: '[0, 0, 1280, 720]',
      cropType: 'Exact Clone (Transcoded)',
    },
    {
      id: 'cand-bloomberg',
      url: 'https://bloomberg.com/news/articles/2021-01-26/india-republic-day-farmers',
      domain: 'bloomberg.com / Syndicated News',
      thumbnailUrl: '/media/candidate_bloomberg.jpg',
      resolution: '275 × 183',
      timestamp: '2021-01-26T08:12:00Z',
      role: 'Discovered Web Crawler Match',
      roleColor: 'border-zinc-500/40 text-zinc-300 bg-zinc-900',
      pdqDistance: 18,
      inliers: 301,
      inlierRatio: 0.479,
      wireAuthority: true,
      boundingBox: '[0, 0, 275, 183]',
      cropType: 'Low-Res Syndication',
    },
  ];

  const selectedNode = nodes.find(n => n.id === selectedNodeId) || nodes[0];

  const copySelectedHash = () => {
    navigator.clipboard.writeText(selectedNode.sha256);
    setCopiedHash(true);
    setTimeout(() => setCopiedHash(false), 2000);
    showToast('Hash copied to clipboard');
  };

  const exportLineageJson = () => {
    const payload = {
      investigation_id: activeDossier.investigation_id,
      case_id: activeDossier.case_id,
      patient_zero: pZero,
      lineage_nodes: nodes,
      timestamp_utc: new Date().toISOString(),
      standard: 'ISO/IEC 27037:2012',
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `provenance_lineage_${activeDossier.case_id.replace(/\s+/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Lineage Telemetry (.JSON) exported successfully.');
  };

  const rawLineageUrl = activeDossier.artifacts?.lineage?.url;
  const isSampleDossier = activeDossier.investigation_id.startsWith('INV-');
  
  const lineageArtifactUrl = (!isSampleDossier && rawLineageUrl && !rawLineageUrl.endsWith('.html'))
    ? getArtifactUrl(activeDossier.investigation_id, 'lineage')
    : (rawLineageUrl || '/astar_lineage_tree.html');
  
  const handleIframeLoad = (e: React.SyntheticEvent<HTMLIFrameElement>) => {
    try {
      const iframe = e.currentTarget;
      const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
      if (iframeDoc) {
        if (!iframeDoc.getElementById('clean-lineage-style')) {
          const style = iframeDoc.createElement('style');
          style.id = 'clean-lineage-style';
          style.textContent = `
            center, h1 { display: none !important; }
            html, body {
              margin: 0 !important;
              padding: 0 !important;
              background-color: #0d1117 !important;
              overflow: hidden !important;
            }
            .card, .card-body {
              border: none !important;
              padding: 0 !important;
              margin: 0 !important;
              background-color: #0d1117 !important;
            }
            #mynetwork {
              border: none !important;
              height: 100vh !important;
              width: 100% !important;
            }
          `;
          iframeDoc.head.appendChild(style);
        }
      }
    } catch (err) {
      console.warn('Could not inject style into iframe:', err);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10">
      {/* Page Title & Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight text-[#f4f4f5]">
            Origin Tracing & Provenance Lineage
          </h1>
          <p className="text-sm text-[#a1a1aa] max-w-2xl leading-relaxed">
            Topological reconstruction of cross-platform media dissemination backwards to the canonical Patient Zero origin.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={exportLineageJson}
            className="px-3.5 py-2 rounded-lg bg-[#18181b] hover:bg-[#222226] border border-[#27272a] text-xs font-medium text-[#f4f4f5] flex items-center gap-2 cursor-pointer transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-[#a1a1aa]" />
            <span>Export Lineage (.JSON)</span>
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Total Propagation Hops</div>
          <div className="text-2xl font-semibold text-[#f4f4f5] font-mono">
            {sourceResult?.lineage_summary?.total_hops || 4}
          </div>
          <div className="text-xs text-[#a1a1aa]">Reconstructed chain</div>
        </div>

        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Propagation Latency</div>
          <div className="text-2xl font-semibold text-[#f4f4f5] font-mono">
            {sourceResult?.lineage_summary?.propagation_span || '29h 18m'}
          </div>
          <div className="text-xs text-[#a1a1aa]">Temporal spread delta</div>
        </div>

        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Root Convergence</div>
          <div className="text-2xl font-semibold text-emerald-400 font-mono">
            {((sourceResult?.lineage_summary?.root_match_score || 0.994) * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-[#a1a1aa]">SIFT homography match</div>
        </div>

        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Perceptual PDQ Distance</div>
          <div className="text-2xl font-semibold text-[#f4f4f5] font-mono">
            Δ {pZero?.pdq_distance ?? 12}
          </div>
          <div className="text-xs text-emerald-400">Within clone threshold (≤30)</div>
        </div>
      </div>

      {/* Viewport Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#27272a] pb-4">
        <div className="flex flex-wrap items-center bg-[#121215] border border-[#27272a] rounded-lg p-1 gap-1">
          <button
            onClick={() => setViewMode('graph')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-2 cursor-pointer transition-colors ${
              viewMode === 'graph'
                ? 'bg-[#27272a] text-[#f4f4f5] shadow-xs'
                : 'text-[#71717a] hover:text-[#f4f4f5]'
            }`}
          >
            <GitBranch className="w-3.5 h-3.5" />
            <span>Interactive Network Graph</span>
          </button>
          <button
            onClick={() => setViewMode('timeline')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-2 cursor-pointer transition-colors ${
              viewMode === 'timeline'
                ? 'bg-[#27272a] text-[#f4f4f5] shadow-xs'
                : 'text-[#71717a] hover:text-[#f4f4f5]'
            }`}
          >
            <Share2 className="w-3.5 h-3.5" />
            <span>Dissemination Timeline</span>
          </button>
          <button
            onClick={() => setViewMode('gallery')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-2 cursor-pointer transition-colors ${
              viewMode === 'gallery'
                ? 'bg-[#27272a] text-[#f4f4f5] shadow-xs'
                : 'text-[#71717a] hover:text-[#f4f4f5]'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5 text-amber-400" />
            <span>Visual Evidence Gallery ({crawledCandidates.length})</span>
          </button>
        </div>

        {viewMode === 'graph' && (
          <button
            onClick={() => setShowIframeModal(true)}
            className="px-3.5 py-1.5 rounded-lg bg-[#18181b] hover:bg-[#222226] border border-[#27272a] text-xs font-medium text-[#f4f4f5] flex items-center gap-2 cursor-pointer transition-colors self-start sm:self-auto"
          >
            <Maximize2 className="w-3.5 h-3.5 text-[#a1a1aa]" />
            <span>Fullscreen Viewport</span>
          </button>
        )}
      </div>

      {/* Viewport Body */}
      {viewMode === 'graph' ? (
        <div className="space-y-4">
          {/* Topology Legend & Engine Meta */}
          <div className="bg-[#121215] border border-[#27272a] rounded-xl p-4 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex flex-wrap items-center gap-4">
              <span className="text-[#71717a] font-medium">Topology Legend:</span>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-[#00E5FF]"></span>
                <span className="text-[#f4f4f5] font-mono">Target Query (Seized)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-[#FF1744]"></span>
                <span className="text-[#f4f4f5] font-mono">Patient Zero (Wire Master)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-[#9C27B0]"></span>
                <span className="text-[#f4f4f5] font-mono">Exact / Crop Clone</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-xs bg-[#78909C]"></span>
                <span className="text-[#71717a] font-mono">Frontier Exploration</span>
              </div>
            </div>

            <div className="text-[#71717a] font-mono text-[11px]">
              Engine: A* Heuristic Traversal • Vis.js Force-Directed Physics
            </div>
          </div>

          {/* Interactive Graph Canvas Viewport */}
          <div className="relative bg-[#09090b] border border-[#27272a] rounded-xl overflow-hidden h-[840px] shadow-inner">
            <iframe
              src={lineageArtifactUrl}
              title="Interactive Lineage Tree"
              onLoad={handleIframeLoad}
              className="w-full h-full border-none bg-[#09090b]"
            />
          </div>

          <div className="text-xs text-[#71717a] leading-relaxed flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-1">
            <span>
              Interactive Canvas: Scroll to zoom, drag nodes to manipulate gravity, click/hover to inspect A* cost scores.
            </span>
            <span className="font-mono text-[#a1a1aa]">
              Source Artifact: astar_lineage_tree.html
            </span>
          </div>
        </div>
      ) : viewMode === 'gallery' ? (
        /* Discovered Visual Crawl Candidates Gallery */
        <div className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-[#f4f4f5]">
                Discovered Visual Crawl Candidates
              </h2>
              <p className="text-xs text-[#71717a]">
                High-resolution media assets caught, downloaded, and cryptographically fingerprinted during the A* traversal.
              </p>
            </div>
            <span className="text-xs font-mono text-[#a1a1aa] bg-[#121215] px-2.5 py-1 rounded border border-[#27272a]">
              5 Cached Candidates in Ledger
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {crawledCandidates.map((cand) => (
              <div 
                key={cand.id}
                className="bg-[#121215] border border-[#27272a] rounded-xl overflow-hidden flex flex-col hover:border-[#3f3f46] transition-colors group"
              >
                {/* Media Preview Box */}
                <div className="relative aspect-video bg-[#09090b] overflow-hidden border-b border-[#27272a]">
                  <img 
                    src={cand.thumbnailUrl} 
                    alt={cand.domain} 
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <span className={`absolute top-2 left-2 text-[10px] font-mono px-2 py-0.5 rounded-md border font-medium ${cand.roleColor}`}>
                    {cand.role}
                  </span>
                  <span className="absolute bottom-2 right-2 text-[10px] font-mono px-2 py-0.5 rounded bg-black/80 text-[#f4f4f5] border border-white/10">
                    {cand.resolution}
                  </span>
                </div>

                {/* Candidate Metadata */}
                <div className="p-4 space-y-3 flex-1 flex flex-col justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <h4 className="text-xs font-semibold text-[#f4f4f5] truncate">
                        {cand.domain}
                      </h4>
                      {cand.wireAuthority && (
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-500/30 shrink-0">
                          Wire Authority
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] font-mono text-[#71717a] truncate">
                      {cand.url}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-2 border-t border-[#27272a]/60">
                    <div>
                      <span className="text-[#71717a] block text-[10px]">PDQ Distance:</span>
                      <span className="text-emerald-400">Δ {cand.pdqDistance}</span>
                    </div>
                    <div>
                      <span className="text-[#71717a] block text-[10px]">SIFT Inliers:</span>
                      <span className="text-[#f4f4f5]">{cand.inliers} ({Math.round(cand.inlierRatio * 100)}%)</span>
                    </div>
                    <div className="col-span-2">
                      <span className="text-[#71717a] block text-[10px]">Bounding Box:</span>
                      <span className="text-amber-300 truncate block">{cand.boundingBox}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Hop-by-Hop Timeline */
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-[#f4f4f5]">
              Dissemination Timeline
            </h2>
            <span className="text-xs text-[#71717a]">
              Click any hop to inspect cryptographic telemetry
            </span>
          </div>

          {/* Step-by-Step Hop Cards with Media Previews */}
          <div className="space-y-4">
            {nodes.map((node) => {
              const isSelected = node.id === selectedNodeId;
              const isRoot = node.type === 'patient_zero';

              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`p-6 rounded-xl border transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-[#18181b] border-[#52525b] shadow-md ring-1 ring-[#52525b]'
                      : isRoot
                      ? 'bg-[#121215] border-emerald-500/30 hover:border-emerald-500/60'
                      : 'bg-[#121215] border-[#27272a] hover:border-[#3f3f46]'
                  }`}
                >
                  <div className="flex flex-col sm:flex-row gap-5 items-start">
                    {/* Visual Media Thumbnail */}
                    {node.thumbnailUrl && (
                      <div className="w-full sm:w-44 h-32 rounded-lg overflow-hidden border border-[#27272a] bg-[#09090b] shrink-0 relative group">
                        <img 
                          src={node.thumbnailUrl} 
                          alt={node.title} 
                          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105" 
                        />
                        {node.cropBadge && (
                          <span className="absolute bottom-1.5 left-1.5 right-1.5 text-[9px] font-mono px-1.5 py-0.5 rounded bg-black/85 backdrop-blur-xs text-amber-300 text-center truncate border border-white/10">
                            {node.cropBadge}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Hop Details */}
                    <div className="flex-1 min-w-0 space-y-2 w-full">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-3">
                          <span className={`text-[11px] font-mono font-medium px-2.5 py-1 rounded-md ${
                            isRoot 
                              ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-500/30' 
                              : 'bg-[#222226] text-[#a1a1aa]'
                          }`}>
                            {node.platform}
                          </span>
                          <h3 className="text-sm font-semibold text-[#f4f4f5] font-sans">
                            {node.title}
                          </h3>
                        </div>

                        <span className="text-xs text-[#71717a] font-mono">
                          {node.latencyFromRoot}
                        </span>
                      </div>

                      <p className="text-xs text-[#a1a1aa] leading-relaxed mb-3">
                        {node.notes}
                      </p>

                      <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-[#71717a] pt-3 border-t border-[#27272a]">
                        <span className="truncate max-w-md text-[#a1a1aa]">
                          {node.sourceUrl}
                        </span>
                        <div className="flex items-center gap-4">
                          <span>{node.resolution}</span>
                          <span>PDQ: Δ{node.pdqDistance}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Selected Node Telemetry Box with Media Preview */}
      <div className="bg-[#121215] border border-[#27272a] rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between border-b border-[#27272a] pb-3">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-[#f4f4f5]">
              Node Metadata: {selectedNode.title}
            </h3>
          </div>
          <span className="text-xs text-[#71717a] font-mono">
            Timestamp: {selectedNode.timestamp}
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-5 items-start">
          {selectedNode.thumbnailUrl && (
            <div className="w-full sm:w-44 h-32 rounded-lg overflow-hidden border border-[#27272a] bg-[#09090b] shrink-0">
              <img 
                src={selectedNode.thumbnailUrl} 
                alt={selectedNode.title} 
                className="w-full h-full object-cover" 
              />
            </div>
          )}

          <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs w-full">
          <div className="space-y-1.5">
            <span className="text-[#71717a] font-medium block">Node SHA-256 Digest</span>
            <div className="bg-[#18181b] border border-[#27272a] rounded-lg p-3 font-mono text-emerald-400 flex items-center justify-between gap-2">
              <span className="truncate">{selectedNode.sha256}</span>
              <button
                onClick={copySelectedHash}
                className="p-1 text-[#71717a] hover:text-[#f4f4f5] cursor-pointer"
                title="Copy hash"
              >
                {copiedHash ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          <div className="space-y-1.5">
            <span className="text-[#71717a] font-medium block">Resource Destination</span>
            <div className="bg-[#18181b] border border-[#27272a] rounded-lg p-3 font-mono text-[#a1a1aa] flex items-center justify-between gap-2">
              <span className="truncate">{selectedNode.sourceUrl}</span>
              <a
                href={selectedNode.sourceUrl}
                target="_blank"
                rel="noreferrer"
                className="text-[#71717a] hover:text-[#f4f4f5]"
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>

      {/* Interactive PyVis Modal */}
      {showIframeModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-xs flex items-center justify-center p-6">
          <div className="bg-[#121215] border border-[#27272a] rounded-xl max-w-5xl w-full h-[80vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-[#27272a] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Share2 className="w-4 h-4 text-[#f4f4f5]" />
                <span className="text-sm font-semibold text-[#f4f4f5]">
                  Interactive Provenance Graph
                </span>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={lineageArtifactUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-[#a1a1aa] hover:text-[#f4f4f5] px-3 py-1.5 rounded bg-[#18181b] flex items-center gap-1.5 cursor-pointer border border-[#27272a]"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  <span>Open in New Tab</span>
                </a>
                <button
                  onClick={() => setShowIframeModal(false)}
                  className="text-xs text-[#a1a1aa] hover:text-[#f4f4f5] px-3 py-1.5 rounded bg-[#18181b] cursor-pointer border border-[#27272a]"
                >
                  Close Viewport
                </button>
              </div>
            </div>

            <div className="flex-1 bg-[#09090b] p-4 flex items-center justify-center">
              {lineageArtifactUrl ? (
                <iframe
                  src={lineageArtifactUrl}
                  title="Lineage Graph"
                  onLoad={handleIframeLoad}
                  className="w-full h-full border-none rounded-lg bg-[#09090b]"
                />
              ) : (
                <div className="text-center space-y-2 max-w-sm">
                  <div className="w-10 h-10 rounded-full bg-[#18181b] text-[#a1a1aa] flex items-center justify-center mx-auto">
                    <GitBranch className="w-5 h-5" />
                  </div>
                  <div className="text-sm font-medium text-[#f4f4f5]">Interactive Graph View</div>
                  <p className="text-xs text-[#71717a]">
                    Physics-based visual node network loaded from gateway artifact endpoint.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
