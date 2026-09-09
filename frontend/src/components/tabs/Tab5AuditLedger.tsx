import React, { useMemo, useState } from 'react';
import { CheckCircle2, Copy, Download, Search } from 'lucide-react';
import { useCaseStore } from '../../store/caseStore';
import { InvestigationEmptyState } from '../InvestigationEmptyState';

const categoryForEvent = (event: string) => {
  const normalized = event.toLowerCase();
  if (normalized.includes('ingest')) return 'Ingress';
  if (normalized.includes('hash')) return 'Hashing';
  if (normalized.includes('homography')) return 'Homography';
  if (normalized.includes('lineage') || normalized.includes('patient') || normalized.includes('source')) return 'Lineage';
  return 'Report';
};

export const Tab5AuditLedger: React.FC = () => {
  const { caseSession, activeDossier, showToast } = useCaseStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const events = useMemo(() => (activeDossier?.custody?.entries || []).map((entry) => ({
    id: `CUSTODY-${String(entry.sequence).padStart(3, '0')}`,
    sequence: entry.sequence,
    timestamp: entry.timestamp,
    category: categoryForEvent(entry.event),
    operation: entry.event,
    digest: entry.entry_hash,
    previousDigest: entry.previous_entry_hash || 'GENESIS',
    inputHash: entry.input_sha256,
    modules: Object.entries(entry.module_statuses || {}).map(([name, status]) => `${name}=${status}`).join(', ') || 'None recorded',
    artifacts: entry.artifact_ids?.join(', ') || 'None recorded',
  })), [activeDossier?.custody?.entries]);

  if (!activeDossier) return <InvestigationEmptyState title="No chain-of-custody investigation loaded" />;

  const filteredEvents = events.filter((event) => {
    const query = searchQuery.toLowerCase();
    const matchesCategory = activeCategory === 'All' || event.category === activeCategory;
    const matchesQuery = [event.id, event.operation, event.digest, event.inputHash, event.modules, event.artifacts].some((value) => value.toLowerCase().includes(query));
    return matchesCategory && matchesQuery;
  });

  const copyDigest = (id: string, value: string) => {
    void navigator.clipboard.writeText(value);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
    showToast('Custody hash copied to clipboard');
  };

  const exportCsv = () => {
    const headers = ['Event_ID', 'Sequence', 'Timestamp_UTC', 'FIR_Reference', 'Investigator_ID', 'Category', 'Operation', 'Entry_Hash', 'Previous_Hash', 'Input_SHA256', 'Module_Statuses', 'Artifact_IDs'];
    const rows = events.map((event) => [event.id, event.sequence, event.timestamp, caseSession.firNumber, caseSession.officerId, event.category, event.operation, event.digest, event.previousDigest, event.inputHash, event.modules, event.artifacts].map((value) => `"${String(value).replaceAll('"', '""')}"`));
    const url = URL.createObjectURL(new Blob([[headers.join(','), ...rows.map((row) => row.join(','))].join('\n')], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `custody_${activeDossier.investigation_id}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    showToast('Current custody ledger exported.');
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end"><div className="space-y-2"><p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">Chain of custody</p><h1 className="text-2xl font-semibold tracking-tight text-gray-950">Gateway custody ledger</h1><p className="text-sm text-gray-600">Only custody entries returned in investigation {activeDossier.investigation_id} are displayed.</p></div><button type="button" onClick={exportCsv} disabled={events.length === 0} className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-300 bg-white px-3.5 py-2 text-xs font-medium text-gray-700 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-40"><Download className="h-4 w-4" /> Export CSV</button></div>

      <div className="grid gap-3 sm:grid-cols-3"><div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm"><div className="text-xs text-gray-500">Returned blocks</div><div className="mt-1 text-2xl font-semibold text-gray-950">{events.length}</div></div><div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm"><div className="text-xs text-gray-500">Hash algorithm</div><div className="mt-1 font-mono text-sm font-semibold text-gray-950">{activeDossier.custody?.hash_algorithm || 'Unavailable'}</div></div><div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm"><div className="text-xs text-gray-500">Investigation status</div><div className="mt-1 text-sm font-semibold text-gray-950">{activeDossier.status}</div></div></div>

      <section className="space-y-5 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center"><div className="relative w-full sm:max-w-md"><Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" /><input type="search" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search event, operation, or hash" className="w-full rounded-lg border border-gray-300 py-2 pl-9 pr-3 text-xs text-gray-950 placeholder:text-gray-400 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-100" /></div><div className="flex flex-wrap gap-1">{['All', 'Ingress', 'Hashing', 'Homography', 'Lineage', 'Report'].map((category) => <button type="button" key={category} onClick={() => setActiveCategory(category)} className={`rounded-md px-2.5 py-1.5 text-xs transition-colors ${activeCategory === category ? 'bg-blue-50 font-semibold text-blue-700' : 'text-gray-600 hover:bg-gray-50 hover:text-gray-950'}`}>{category}</button>)}</div></div>

        {filteredEvents.length === 0 ? <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-8 text-center text-xs text-gray-600">{events.length === 0 ? 'The gateway response contains no custody entries.' : 'No custody entries match the current filter.'}</div> : <div className="space-y-3">{filteredEvents.map((event) => <article key={event.id} className="space-y-3 rounded-lg border border-gray-200 p-4"><div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center"><div className="flex flex-wrap items-center gap-2"><span className="rounded bg-gray-100 px-2 py-1 font-mono text-[11px] font-semibold text-gray-800">{event.id}</span><span className="text-xs font-semibold text-gray-950">{event.operation}</span><span className="rounded-full bg-blue-50 px-2 py-1 text-[11px] text-blue-700">{event.category}</span></div><div className="flex items-center gap-2 text-[11px] text-gray-500"><span>{event.timestamp}</span><span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle2 className="h-3.5 w-3.5" />Returned</span></div></div><div className="grid gap-3 text-[11px] text-gray-600 md:grid-cols-2"><div><span className="text-gray-500">Input SHA-256</span><div className="mt-1 break-all font-mono text-gray-800">{event.inputHash}</div></div><div><span className="text-gray-500">Module statuses</span><div className="mt-1 text-gray-800">{event.modules}</div></div><div><span className="text-gray-500">Artifacts</span><div className="mt-1 text-gray-800">{event.artifacts}</div></div><div><span className="text-gray-500">Previous hash</span><div className="mt-1 break-all font-mono text-gray-800">{event.previousDigest}</div></div></div><div className="flex items-center justify-between gap-3 border-t border-gray-200 pt-3"><div className="min-w-0"><span className="text-[11px] text-gray-500">Entry hash</span><div className="truncate font-mono text-xs text-gray-900">{event.digest}</div></div><button type="button" onClick={() => copyDigest(event.id, event.digest)} className="shrink-0 rounded-md border border-gray-200 p-2 text-gray-500 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700" title="Copy entry hash" aria-label="Copy entry hash">{copiedId === event.id ? <CheckCircle2 className="h-4 w-4 text-emerald-700" /> : <Copy className="h-4 w-4" />}</button></div></article>)}</div>}
      </section>
    </div>
  );
};
