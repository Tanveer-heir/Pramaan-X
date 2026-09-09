import React from 'react';
import { AlertTriangle, CheckCircle2, ExternalLink, FileDown, Info, XCircle } from 'lucide-react';
import { getArtifactUrl } from '../../services/gateway';
import type { InvestigationRecord, ModuleEnvelope } from '../../types/contract';

type UnknownRecord = Record<string, unknown>;

interface InvestigationOverviewProps {
  investigation: InvestigationRecord;
  dossierSource: 'live' | 'sample' | null;
}

const asRecord = (value: unknown): UnknownRecord | null => (
  value && typeof value === 'object' && !Array.isArray(value) ? value as UnknownRecord : null
);

const textValue = (value: unknown, fallback = 'Unavailable') => (
  typeof value === 'string' && value.trim() ? value : fallback
);

const numberValue = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : null;

const scalarValue = (value: unknown, fallback = 'Unavailable') => (
  typeof value === 'number' && Number.isFinite(value) ? String(value) : textValue(value, fallback)
);

const percent = (value: unknown) => {
  const number = numberValue(value);
  if (number === null) return 'Unavailable';
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
};

const humanize = (value: unknown) => textValue(value).toLowerCase().replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

const moduleMessage = (module: ModuleEnvelope<unknown>) => textValue(module.message || module.reason, 'No additional details returned.');

const classLabel = (value: unknown) => {
  const labels: Record<string, string> = {
    REAL: 'Authentic',
    VISUAL_MANIPULATION: 'Visual manipulation',
    AUDIO_MANIPULATION: 'Audio manipulation',
    AUDIO_VISUAL_MANIPULATION: 'Audio-visual manipulation',
  };
  return typeof value === 'string' && labels[value] ? labels[value] : humanize(value);
};

const statusStyles: Record<string, string> = {
  COMPLETED: 'border-emerald-200 bg-emerald-50 text-emerald-800',
  NOT_APPLICABLE: 'border-gray-200 bg-gray-50 text-gray-700',
  PARTIAL: 'border-amber-200 bg-amber-50 text-amber-800',
  FAILED: 'border-red-200 bg-red-50 text-red-800',
  TIMEOUT: 'border-red-200 bg-red-50 text-red-800',
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const isGood = status === 'COMPLETED';
  const isUnavailable = status === 'FAILED' || status === 'TIMEOUT';
  const Icon = isGood ? CheckCircle2 : isUnavailable ? XCircle : Info;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-1 text-[11px] font-semibold ${statusStyles[status] || 'border-gray-200 bg-gray-50 text-gray-700'}`}>
      <Icon className="h-3.5 w-3.5" />
      {humanize(status)}
    </span>
  );
};

const ModuleHeader: React.FC<{ title: string; module: ModuleEnvelope<unknown> }> = ({ title, module }) => (
  <div className="flex flex-col justify-between gap-2 border-b border-gray-200 pb-3 sm:flex-row sm:items-center">
    <div>
      <h2 className="text-sm font-semibold text-gray-950">{title}</h2>
      <p className="mt-0.5 text-xs text-gray-500">
        {module.duration_sec !== undefined ? `${module.duration_sec.toFixed(2)} seconds` : 'Duration unavailable'}
      </p>
    </div>
    <StatusBadge status={module.status} />
  </div>
);

const TechnicalDetails: React.FC<{ value: unknown }> = ({ value }) => (
  <details className="rounded-lg border border-gray-200 bg-gray-50 text-xs text-gray-700">
    <summary className="cursor-pointer px-3 py-2 font-medium text-gray-700">View technical details</summary>
    <pre className="max-h-80 overflow-auto border-t border-gray-200 p-3 font-mono text-[11px] leading-5 text-gray-600">{JSON.stringify(value, null, 2)}</pre>
  </details>
);

const EmptyModuleResult: React.FC<{ module: ModuleEnvelope<unknown> }> = ({ module }) => (
  <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600">
    <span className="font-medium text-gray-900">{module.status === 'NOT_APPLICABLE' ? 'Not applicable.' : 'No result returned.'}</span>{' '}
    {moduleMessage(module)}
  </div>
);

const DetectionPanel: React.FC<{ investigation: InvestigationRecord }> = ({ investigation }) => {
  const module = investigation.modules.detection as ModuleEnvelope<unknown>;
  const result = asRecord(module.result);
  const assessment = asRecord(result?.assessment);
  const isVideo = investigation.media.media_type === 'video';
  const label = isVideo ? result?.selected_class || result?.classification : assessment?.label || result?.assessment;
  const confidence = assessment?.confidence_level || result?.confidence_level || result?.confidence;
  const findings = Array.isArray(result?.visual_findings) ? result.visual_findings.filter((item): item is string => typeof item === 'string') : [];
  const branchEvidence = asRecord(result?.branch_evidence);
  const summarySemantics = investigation.summary?.detection_score_semantics || '';
  const calibrated = result?.calibrated === true
    || typeof result?.calibration === 'string'
    || /calibrat/i.test(summarySemantics);
  const scores = asRecord(result?.softmax_probabilities);
  const warningItems = [
    ...((Array.isArray(result?.warnings) ? result.warnings : []).filter((item): item is string => typeof item === 'string')),
    ...((Array.isArray(result?.limitations) ? result.limitations : []).filter((item): item is string => typeof item === 'string')),
  ];

  return (
    <section className="space-y-4 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <ModuleHeader title="Detection" module={module} />
      {!result ? <EmptyModuleResult module={module} /> : (
        <>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border border-blue-100 bg-blue-50/50 p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Verdict</div>
              <div className="mt-1 text-lg font-semibold text-gray-950">{isVideo ? classLabel(label) : humanize(label)}</div>
              <div className="mt-1 text-xs text-gray-600">{textValue(assessment?.summary || result?.classification, 'No summary returned.')}</div>
            </div>
            <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Confidence</div>
              <div className="mt-1 text-lg font-semibold text-gray-950">{humanize(confidence)}</div>
              <div className="mt-1 text-xs text-gray-600">Only values explicitly returned by the gateway are shown.</div>
            </div>
          </div>

          {scores && (
            <div className="space-y-2 rounded-lg border border-gray-200 p-4">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-xs font-semibold text-gray-950">{calibrated ? 'Calibrated probabilities' : 'Model scores'}</h3>
                {!calibrated && <span className="text-[11px] text-gray-500">Not presented as calibrated probability</span>}
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {Object.entries(scores).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between rounded-md bg-gray-50 px-3 py-2 text-xs">
                    <span className="text-gray-700">{classLabel(key)}</span>
                    <span className="font-mono font-semibold text-gray-950">{percent(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {branchEvidence && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-gray-950">Branch evidence</h3>
              <div className="grid gap-2 md:grid-cols-3">
                {Object.entries(branchEvidence).map(([key, rawEvidence]) => {
                  const evidence = asRecord(rawEvidence);
                  return (
                    <div key={key} className="rounded-lg border border-gray-200 p-3 text-xs">
                      <div className="flex items-center justify-between gap-2 font-medium text-gray-900">
                        <span>{humanize(key)}</span>
                        <span className={evidence?.available === false ? 'text-gray-500' : 'text-emerald-700'}>{evidence?.available === false ? 'Unavailable' : 'Available'}</span>
                      </div>
                      <div className="mt-1 text-gray-600">
                        {evidence?.confidence !== undefined ? `Confidence ${percent(evidence.confidence)}` : evidence?.sync_score !== undefined ? `Sync score ${percent(evidence.sync_score)}` : 'No score returned.'}
                      </div>
                      {typeof evidence?.details === 'string' && <div className="mt-1 text-gray-500">{evidence.details}</div>}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {findings.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-gray-950">Visual findings</h3>
              <ul className="list-disc space-y-1 pl-5 text-xs text-gray-600">
                {findings.map((finding) => <li key={finding}>{finding}</li>)}
              </ul>
            </div>
          )}

          {warningItems.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
              <div className="mb-1 flex items-center gap-2 font-semibold"><AlertTriangle className="h-4 w-4" />Warnings and limitations</div>
              <ul className="list-disc space-y-1 pl-5">{warningItems.map((warning) => <li key={warning}>{warning}</li>)}</ul>
            </div>
          )}
          <TechnicalDetails value={result} />
        </>
      )}
    </section>
  );
};

const DevicePanel: React.FC<{ investigation: InvestigationRecord }> = ({ investigation }) => {
  const module = investigation.modules.device_attribution as ModuleEnvelope<unknown>;
  const envelope = asRecord(module.result);
  const result = asRecord(envelope?.result) || envelope;
  const deviceResult: UnknownRecord = asRecord(result?.device_attribution) || result || {};
  const exif = asRecord(result?.metadata_exif) || asRecord(result?.metadata);
  const evidence = Array.isArray(deviceResult?.evidence) ? deviceResult.evidence.filter((item): item is string => typeof item === 'string') : [];
  const warnings = Array.isArray(envelope?.warnings) ? envelope.warnings.filter((item): item is string => typeof item === 'string') : [];
  return (
    <section className="space-y-4 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <ModuleHeader title="Device attribution" module={module} />
      {!result ? <EmptyModuleResult module={module} /> : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ['Attributed device', textValue(deviceResult.display_name || deviceResult.attributed_device || deviceResult.camera_model)],
              ['Method', humanize(deviceResult.primary_method || deviceResult.attribution_method)],
              ['Confidence', percent(deviceResult.confidence)],
              ['PCE / p-value', `${scalarValue(deviceResult.pce)} / ${scalarValue(deviceResult.p_value)}`],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg border border-gray-200 bg-gray-50 p-3">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">{label}</div>
                <div className="mt-1 break-words text-sm font-medium text-gray-950">{value}</div>
              </div>
            ))}
          </div>
          {(deviceResult.matched_model || deviceResult.known_device_match) && <div className="text-xs text-gray-600">Matched model: <span className="font-medium text-gray-950">{textValue(deviceResult.matched_model || deviceResult.known_device_match)}</span></div>}
          {evidence.length > 0 && <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600"><div className="font-semibold text-gray-950">Device evidence</div><ul className="mt-1 list-disc space-y-1 pl-5">{evidence.map((item) => <li key={item}>{item}</li>)}</ul></div>}
          {warnings.length > 0 && <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900"><div className="font-semibold">Device warnings</div><ul className="mt-1 list-disc space-y-1 pl-5">{warnings.map((item) => <li key={item}>{item}</li>)}</ul></div>}
          {exif && <TechnicalDetails value={exif} />}
        </>
      )}
    </section>
  );
};

const SourcePanel: React.FC<{ investigation: InvestigationRecord; dossierSource: 'live' | 'sample' | null }> = ({ investigation, dossierSource }) => {
  const module = investigation.modules.source_attribution as ModuleEnvelope<unknown>;
  const result = asRecord(module.result);
  const patientZero = asRecord(result?.patient_zero);
  const candidates = Array.isArray(result?.candidate_sources) ? result.candidate_sources : [];
  return (
    <section className="space-y-4 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <ModuleHeader title="Source attribution" module={module} />
      {!result ? <EmptyModuleResult module={module} /> : (
        <>
          {patientZero ? (
            <div className="rounded-lg border border-red-100 bg-red-50/50 p-4">
              <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-start">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-red-700">Patient-zero candidate</div>
                  <div className="mt-1 break-words text-sm font-semibold text-gray-950">{textValue(patientZero.domain || patientZero.publisher)}</div>
                </div>
                {(typeof patientZero.url === 'string' || typeof patientZero.media_url === 'string' || typeof patientZero.source_page_url === 'string') && (
                  <a
                    href={textValue(patientZero.url || patientZero.media_url || patientZero.source_page_url, '#')}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-xs font-medium text-blue-700 hover:underline"
                  >
                    Open returned source <ExternalLink className="h-3.5 w-3.5" />
                  </a>
                )}
              </div>
              <div className="mt-3 grid gap-2 text-xs text-gray-700 sm:grid-cols-3">
                <div><span className="text-gray-500">Match type</span><div className="font-medium text-gray-950">{humanize(patientZero.match_type || patientZero.crop_type)}</div></div>
                <div><span className="text-gray-500">Confidence / probability</span><div className="font-medium text-gray-950">{percent(patientZero.confidence ?? patientZero.probability)}</div></div>
                <div><span className="text-gray-500">Inlier ratio</span><div className="font-medium text-gray-950">{percent(patientZero.inlier_ratio)}</div></div>
              </div>
              {(typeof patientZero.evidence === 'string' || typeof patientZero.details === 'string') && <p className="mt-3 text-xs leading-5 text-gray-600">{textValue(patientZero.evidence || patientZero.details)}</p>}
            </div>
          ) : <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600">No patient-zero candidate was returned.</div>}

          {candidates.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-gray-950">Returned candidate sources</h3>
              <div className="divide-y divide-gray-200 rounded-lg border border-gray-200">
                {candidates.map((candidate, index) => {
                  const item = asRecord(candidate);
                  const url = item?.url || item?.media_url || item?.source_page_url;
                  return (
                    <div key={`${textValue(item?.url, 'candidate')}-${index}`} className="flex flex-col justify-between gap-2 p-3 sm:flex-row sm:items-center">
                      <div className="min-w-0"><div className="truncate text-xs font-medium text-gray-950">{textValue(item?.domain || item?.publisher || url)}</div><div className="text-[11px] text-gray-500">{humanize(item?.match_type || item?.crop_type)} · score {percent(item?.probability ?? item?.confidence)}</div></div>
                      {typeof url === 'string' && /^https?:\/\//i.test(url) && <a href={url} target="_blank" rel="noreferrer" className="inline-flex shrink-0 items-center gap-1 text-xs text-blue-700 hover:underline">Open <ExternalLink className="h-3.5 w-3.5" /></a>}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <TechnicalDetails value={result} />
        </>
      )}
      {investigation.artifacts && Object.keys(investigation.artifacts).length > 0 && (
        <div className="space-y-2 border-t border-gray-200 pt-4">
          <h3 className="text-xs font-semibold text-gray-950">Returned artifacts</h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(investigation.artifacts).map(([key, artifact]) => (
              <a
                key={key}
                href={dossierSource === 'live' ? getArtifactUrl(investigation.investigation_id, key) : artifact.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-md border border-gray-200 px-3 py-2 text-xs font-medium text-blue-700 hover:border-blue-300 hover:bg-blue-50"
              >
                <FileDown className="h-3.5 w-3.5" />
                {key}
              </a>
            ))}
          </div>
        </div>
      )}
    </section>
  );
};

export const InvestigationOverview: React.FC<InvestigationOverviewProps> = ({ investigation, dossierSource }) => (
  <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
    <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">{dossierSource === 'live' ? 'Live gateway investigation' : 'Explicit sample investigation'}</p>
          <StatusBadge status={investigation.status} />
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-gray-950">Investigation results</h1>
        <p className="text-sm text-gray-600">Detection, device attribution, source attribution, and artifact links below are from this response.</p>
      </div>
    </div>

    <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="grid gap-4 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <div><span className="text-gray-500">Investigation ID</span><div className="mt-1 break-all font-mono font-medium text-gray-950">{investigation.investigation_id}</div></div>
        <div><span className="text-gray-500">Case ID</span><div className="mt-1 break-all font-mono font-medium text-gray-950">{investigation.case_id}</div></div>
        <div><span className="text-gray-500">Evidence</span><div className="mt-1 break-all font-medium text-gray-950">{investigation.media.filename}</div><div className="text-gray-500">{investigation.media.media_type} · {(investigation.media.size_bytes / 1024 / 1024).toFixed(2)} MB</div></div>
        <div><span className="text-gray-500">Execution time</span><div className="mt-1 font-mono font-medium text-gray-950">{investigation.execution_time_sec !== undefined ? `${investigation.execution_time_sec.toFixed(2)} seconds` : 'Unavailable'}</div></div>
      </div>
      <div className="mt-4 border-t border-gray-200 pt-4"><span className="text-xs text-gray-500">Evidence SHA-256</span><div className="mt-1 break-all font-mono text-xs text-gray-800">{investigation.media.sha256}</div></div>
    </section>

    {investigation.warnings && investigation.warnings.length > 0 && (
      <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /><div><div className="font-semibold">Investigation warnings</div><ul className="mt-1 list-disc space-y-1 pl-4">{investigation.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div></div>
    )}

    <div className="grid gap-6 xl:grid-cols-2">
      <DetectionPanel investigation={investigation} />
      <DevicePanel investigation={investigation} />
    </div>
    <SourcePanel investigation={investigation} dossierSource={dossierSource} />
  </div>
);
