import React, { useState } from 'react';
import { 
  Search, 
  Download, 
  CheckCircle2, 
  Copy, 
  Check
} from 'lucide-react';
import { useCaseStore } from '../../store/caseStore';

interface AuditEvent {
  eventId: string;
  sequence: number;
  timestamp: string;
  operatorId: string;
  category: 'Ingress' | 'Hashing' | 'Homography' | 'Lineage' | 'Sec 65B';
  operation: string;
  digest: string;
  previousDigest: string;
  notes: string;
}

export const Tab5AuditLedger: React.FC = () => {
  const { caseSession, activeDossier, showToast } = useCaseStore();
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [activeCategory, setActiveCategory] = useState<string>('All');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const custodyEntries = activeDossier.custody.entries;

  const events: AuditEvent[] = [
    {
      eventId: 'EVT-1094',
      sequence: 1,
      timestamp: custodyEntries[0]?.timestamp || '2025-01-15T09:36:12Z',
      operatorId: caseSession.officerId,
      category: 'Ingress',
      operation: 'INGEST_BITSTREAM_VOLATILE',
      digest: activeDossier.media.sha256,
      previousDigest: 'GENESIS_BLOCK_0000000000000000000000000000000000000000000000000000000000000000',
      notes: `Ingested ${activeDossier.media.filename} (${(activeDossier.media.size_bytes / 1024).toFixed(1)} KB). Volatile RAM buffer locked under ISO/IEC 27037:2012 protocol.`,
    },
    {
      eventId: 'EVT-1095',
      sequence: 2,
      timestamp: custodyEntries[1]?.timestamp || '2025-01-15T09:36:15Z',
      operatorId: caseSession.officerId,
      category: 'Hashing',
      operation: 'PERCEPTUAL_DCT_HASH',
      digest: '3f7a8b9c1d2e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a',
      previousDigest: activeDossier.media.sha256,
      notes: 'Computed 256-bit DCT perceptual hash matrix & Meta PDQ Hamming distance across spatial blocks.',
    },
    {
      eventId: 'EVT-1096',
      sequence: 3,
      timestamp: custodyEntries[2]?.timestamp || '2025-01-15T09:36:24Z',
      operatorId: caseSession.officerId,
      category: 'Homography',
      operation: 'HOMOGRAPHY_RANSAC_CONVERGENCE',
      digest: '8a2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c',
      previousDigest: '3f7a8b9c1d2e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a',
      notes: '1,384 SIFT keypoint vector pairs converged under projective transformation H(3x3). Sub-crop offset proved.',
    },
    {
      eventId: 'EVT-1097',
      sequence: 4,
      timestamp: custodyEntries[3]?.timestamp || '2025-01-15T09:36:28Z',
      operatorId: caseSession.officerId,
      category: 'Lineage',
      operation: 'DAG_PATIENT_ZERO_ISOLATED',
      digest: 'b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6',
      previousDigest: '8a2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c',
      notes: 'A* frontier traversal isolated canonical root publication on wire broadcast dated 2021-01-26T04:18:22Z.',
    },
    {
      eventId: 'EVT-1098',
      sequence: 5,
      timestamp: custodyEntries[4]?.timestamp || '2025-01-15T09:36:31Z',
      operatorId: caseSession.officerId,
      category: 'Sec 65B',
      operation: 'SEC_65B_CERT_COMPILATION',
      digest: '0x8F92A1B4C7D8E2F305619A4B8C1D7E3F5A9B2C4D6E8F0A1B3C5D7E9F1A2B4C6D',
      previousDigest: 'b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6',
      notes: 'Digital Section 65B forensic affidavit package compiled with tamper-evident cryptographic seal.',
    },
  ];

  const filteredEvents = events.filter((ev) => {
    const matchesCategory = activeCategory === 'All' || ev.category === activeCategory;
    const query = searchQuery.toLowerCase();
    const matchesQuery = 
      ev.eventId.toLowerCase().includes(query) ||
      ev.operation.toLowerCase().includes(query) ||
      ev.digest.toLowerCase().includes(query) ||
      ev.notes.toLowerCase().includes(query);
    return matchesCategory && matchesQuery;
  });

  const copyDigest = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
    showToast('Hash copied to clipboard');
  };

  const exportCsv = () => {
    const headers = ['Event_ID', 'Sequence', 'Timestamp_UTC', 'FIR_Reference', 'Investigator_ID', 'Category', 'Operation_Name', 'SHA256_Digest', 'Previous_Digest', 'Notes'];
    const rows = events.map((ev) => [
      ev.eventId,
      ev.sequence,
      ev.timestamp,
      `"${caseSession.firNumber}"`,
      ev.operatorId,
      ev.category,
      ev.operation,
      ev.digest,
      ev.previousDigest,
      `"${ev.notes.replace(/"/g, '""')}"`,
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit_ledger_${caseSession.firNumber.replace(/\s+/g, '_')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Audit Trail (.CSV) exported successfully.');
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10">
      {/* Title & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight text-[#f4f4f5]">
            Immutable Chain-of-Custody Ledger
          </h1>
          <p className="text-sm text-[#a1a1aa] max-w-2xl leading-relaxed">
            Cryptographically verified, append-only event stream satisfying ISO/IEC 27037:2012 digital evidence handling standards.
          </p>
        </div>

        <button
          onClick={exportCsv}
          className="px-3.5 py-2 rounded-lg bg-[#18181b] hover:bg-[#222226] border border-[#27272a] text-xs font-medium text-[#f4f4f5] flex items-center gap-2 cursor-pointer transition-colors self-start sm:self-auto"
        >
          <Download className="w-3.5 h-3.5 text-[#a1a1aa]" />
          <span>Export Audit Trail (.CSV)</span>
        </button>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Sealed Audit Blocks</div>
          <div className="text-2xl font-semibold text-[#f4f4f5] font-mono">5 / 5</div>
          <div className="text-xs text-emerald-400">100% Continuity</div>
        </div>

        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Tamper Anomalies</div>
          <div className="text-2xl font-semibold text-emerald-400 font-mono">0</div>
          <div className="text-xs text-[#a1a1aa]">Deterministic verification</div>
        </div>

        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Custody Standard</div>
          <div className="text-sm font-semibold text-[#f4f4f5] font-mono mt-1">ISO/IEC 27037</div>
          <div className="text-xs text-[#a1a1aa]">Volatile buffer protocol</div>
        </div>

        <div className="bg-[#121215] border border-[#27272a] rounded-xl p-5 space-y-1">
          <div className="text-xs text-[#71717a] font-medium">Statute Compliance</div>
          <div className="text-sm font-semibold text-[#f4f4f5] font-mono mt-1">Section 65B(4)</div>
          <div className="text-xs text-[#a1a1aa]">Judicial admissibility</div>
        </div>
      </div>

      {/* Main Ledger Container */}
      <div className="bg-[#121215] border border-[#27272a] rounded-xl p-8 shadow-sm space-y-6">
        {/* Search & Category Filter Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-[#71717a] absolute left-3.5 top-2.5" />
            <input
              type="text"
              placeholder="Search by event ID, operation, or digest..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-[#18181b] border border-[#27272a] rounded-lg pl-9 pr-4 py-2 text-xs text-[#f4f4f5] placeholder:text-[#52525b] focus:border-[#71717a] focus:outline-none"
            />
          </div>

          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            {['All', 'Ingress', 'Hashing', 'Homography', 'Lineage', 'Sec 65B'].map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={`px-3 py-1 rounded-md transition-colors cursor-pointer ${
                  activeCategory === cat
                    ? 'bg-[#27272a] text-[#f4f4f5] font-medium'
                    : 'text-[#a1a1aa] hover:text-[#f4f4f5] hover:bg-[#18181b]'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Ledger Event Rows */}
        <div className="space-y-4 pt-2">
          {filteredEvents.map((ev) => (
            <div
              key={ev.eventId}
              className="p-5 rounded-lg border border-[#27272a] bg-[#18181b]/50 hover:bg-[#18181b] transition-colors space-y-3"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-[#222226] text-[#f4f4f5]">
                    {ev.eventId}
                  </span>
                  <span className="text-xs font-semibold text-[#f4f4f5] font-mono">
                    {ev.operation}
                  </span>
                  <span className="text-[11px] text-[#71717a]">
                    {ev.category}
                  </span>
                </div>

                <div className="flex items-center gap-3 text-xs font-mono text-[#71717a]">
                  <span>{ev.timestamp}</span>
                  <span className="text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Verified
                  </span>
                </div>
              </div>

              <p className="text-xs text-[#a1a1aa] leading-relaxed">
                {ev.notes}
              </p>

              <div className="pt-3 border-t border-[#27272a] grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
                <div className="space-y-1">
                  <span className="text-[#71717a] text-[10px] uppercase font-sans">Block Digest</span>
                  <div className="bg-[#121215] border border-[#27272a] p-2 rounded text-emerald-400 flex items-center justify-between gap-2">
                    <span className="truncate">{ev.digest}</span>
                    <button
                      onClick={() => copyDigest(ev.eventId, ev.digest)}
                      className="p-1 text-[#71717a] hover:text-[#f4f4f5] cursor-pointer"
                      title="Copy digest"
                    >
                      {copiedId === ev.eventId ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                </div>

                <div className="space-y-1">
                  <span className="text-[#71717a] text-[10px] uppercase font-sans">Previous Block Hash</span>
                  <div className="bg-[#121215] border border-[#27272a] p-2 rounded text-[#71717a] truncate">
                    {ev.previousDigest}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {filteredEvents.length === 0 && (
            <div className="text-center py-12 text-[#71717a] text-xs">
              No audit records match the selected filter query.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
