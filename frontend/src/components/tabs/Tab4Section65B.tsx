import React from 'react';
import { Download, FileText } from 'lucide-react';
import { useCaseStore } from '../../store/caseStore';
import { getArtifactUrl } from '../../services/gateway';
import { InvestigationEmptyState } from '../InvestigationEmptyState';

const asRecord = (value: unknown): Record<string, unknown> | null => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
const textValue = (value: unknown, fallback = 'Unavailable') => typeof value === 'string' && value.trim() ? value : fallback;
const humanize = (value: unknown) => textValue(value).toLowerCase().replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

export const Tab4Section65B: React.FC = () => {
  const { caseSession, activeDossier, dossierSource, showToast } = useCaseStore();

  if (!activeDossier) return <InvestigationEmptyState title="No court report investigation loaded" />;

  const detection = asRecord(activeDossier.modules.detection.result);
  const assessment = asRecord(detection?.assessment);
  const device = asRecord(activeDossier.modules.device_attribution.result);
  const source = asRecord(activeDossier.modules.source_attribution.result);
  const patientZero = asRecord(source?.patient_zero);
  const latestCustodyEntry = activeDossier.custody?.entries.at(-1);

  const exportReport = () => {
    const payload = {
      report_type: 'investigation_record_export',
      generated_at_utc: new Date().toISOString(),
      certifying_officer: caseSession,
      investigation_record: activeDossier,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `investigation_report_${activeDossier.investigation_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
    showToast('Current investigation report exported.');
  };

  const moduleRows = [
    ['Detection', activeDossier.modules.detection],
    ['Device attribution', activeDossier.modules.device_attribution],
    ['Source attribution', activeDossier.modules.source_attribution],
  ] as const;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div className="space-y-2"><p className="text-xs font-semibold uppercase tracking-[0.18em] text-red-700">Court report</p><h1 className="text-2xl font-semibold tracking-tight text-gray-950">Section 65B / Section 63 evidence report</h1><p className="text-sm text-gray-600">This view is generated from the currently selected investigation record.</p></div>
        <button type="button" onClick={exportReport} className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-300 bg-white px-3.5 py-2 text-xs font-medium text-gray-700 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"><Download className="h-4 w-4" /> Export current record</button>
      </div>

      <section className="rounded-xl border border-gray-300 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between gap-3 border-b border-gray-200 pb-4"><div><div className="flex items-center gap-2"><FileText className="h-5 w-5 text-red-700" /><h2 className="text-base font-semibold text-gray-950">Forensic examination report</h2></div><p className="mt-1 text-xs text-gray-500">Investigation {activeDossier.investigation_id} · {activeDossier.status}</p></div><span className="rounded-full border border-gray-200 bg-gray-50 px-2 py-1 text-[11px] font-semibold text-gray-700">{dossierSource === 'live' ? 'Gateway response' : 'Explicit sample'}</span></div>

        <div className="mt-5 grid gap-4 text-xs sm:grid-cols-2">
          <div><span className="text-gray-500">Court / jurisdiction</span><div className="mt-1 font-medium text-gray-950">{caseSession.judicialCourt}</div></div>
          <div><span className="text-gray-500">FIR / case</span><div className="mt-1 font-mono font-medium text-gray-950">{activeDossier.case_id}</div></div>
          <div><span className="text-gray-500">Police station</span><div className="mt-1 font-medium text-gray-950">{caseSession.policeStation}</div></div>
          <div><span className="text-gray-500">Certifying officer</span><div className="mt-1 font-medium text-gray-950">{caseSession.officerInCharge} ({caseSession.officerId})</div></div>
        </div>

        <div className="mt-6 space-y-3 border-t border-gray-200 pt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-700">Evidence identity</h3>
          <div className="grid gap-3 rounded-lg border border-gray-200 bg-gray-50 p-4 text-xs sm:grid-cols-2"><div><span className="text-gray-500">Filename</span><div className="mt-1 break-all font-medium text-gray-950">{activeDossier.media.filename}</div></div><div><span className="text-gray-500">Media</span><div className="mt-1 font-medium uppercase text-gray-950">{activeDossier.media.media_type} · {(activeDossier.media.size_bytes / 1024).toFixed(1)} KB</div></div><div className="sm:col-span-2"><span className="text-gray-500">SHA-256</span><div className="mt-1 break-all font-mono text-gray-950">{activeDossier.media.sha256}</div></div></div>
        </div>

        <div className="mt-6 space-y-3 border-t border-gray-200 pt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-700">Technical findings</h3>
          <div className="overflow-x-auto rounded-lg border border-gray-200"><table className="w-full min-w-[640px] border-collapse text-xs"><tbody>
            <tr><th className="w-1/3 border-b border-gray-200 bg-gray-50 p-3 text-left font-semibold text-gray-700">Detection verdict</th><td className="border-b border-gray-200 p-3 text-gray-900">{activeDossier.modules.detection.status === 'COMPLETED' ? humanize(detection?.selected_class || detection?.classification || assessment?.label || detection?.assessment) : `${activeDossier.modules.detection.status}: ${textValue(activeDossier.modules.detection.message || activeDossier.modules.detection.reason)}`}</td></tr>
            <tr><th className="w-1/3 border-b border-gray-200 bg-gray-50 p-3 text-left font-semibold text-gray-700">Device attribution</th><td className="border-b border-gray-200 p-3 text-gray-900">{activeDossier.modules.device_attribution.status === 'COMPLETED' ? `${textValue(device?.attributed_device)} · ${humanize(device?.attribution_method)}` : `${activeDossier.modules.device_attribution.status}: ${textValue(activeDossier.modules.device_attribution.message || activeDossier.modules.device_attribution.reason)}`}</td></tr>
            <tr><th className="w-1/3 border-b border-gray-200 bg-gray-50 p-3 text-left font-semibold text-gray-700">Patient-zero candidate</th><td className="border-b border-gray-200 p-3 text-gray-900">{activeDossier.modules.source_attribution.status === 'COMPLETED' ? textValue(patientZero?.domain || patientZero?.publisher || patientZero?.url, 'No candidate returned') : `${activeDossier.modules.source_attribution.status}: ${textValue(activeDossier.modules.source_attribution.message || activeDossier.modules.source_attribution.reason)}`}</td></tr>
            <tr><th className="bg-gray-50 p-3 text-left font-semibold text-gray-700">Latest custody hash</th><td className="break-all p-3 font-mono text-gray-900">{latestCustodyEntry?.entry_hash || 'Unavailable'}</td></tr>
          </tbody></table></div>
        </div>

        <div className="mt-6 space-y-2 border-t border-gray-200 pt-5"><h3 className="text-xs font-semibold uppercase tracking-wide text-gray-700">Module status</h3>{moduleRows.map(([label, module]) => <div key={label} className="flex flex-col justify-between gap-1 rounded-lg border border-gray-200 px-3 py-2 text-xs sm:flex-row sm:items-center"><span className="font-medium text-gray-900">{label}</span><span className="text-gray-600">{module.status} · {module.duration_sec !== undefined ? `${module.duration_sec.toFixed(2)}s` : 'duration unavailable'}{module.reason ? ` · ${module.reason}` : ''}</span></div>)}</div>

        {activeDossier.warnings && activeDossier.warnings.length > 0 && <div className="mt-6 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900"><span className="font-semibold">Warnings:</span> {activeDossier.warnings.join(' ')}</div>}

        {activeDossier.artifacts && Object.keys(activeDossier.artifacts).length > 0 && <div className="mt-6 flex flex-wrap gap-2 border-t border-gray-200 pt-5">{Object.entries(activeDossier.artifacts).map(([key, artifact]) => <a key={key} href={dossierSource === 'live' ? getArtifactUrl(activeDossier.investigation_id, key) : artifact.url} target="_blank" rel="noreferrer" className="rounded-md border border-gray-200 px-3 py-2 text-xs font-medium text-blue-700 hover:border-blue-300 hover:bg-blue-50">Open {key}</a>)}</div>}
      </section>
    </div>
  );
};
