import React from 'react';
import { Download, Info } from 'lucide-react';
import { useCaseStore } from '../../store/caseStore';
import { InvestigationEmptyState } from '../InvestigationEmptyState';

export const Tab3Homography: React.FC = () => {
  const { activeDossier, showToast } = useCaseStore();

  if (!activeDossier) return <InvestigationEmptyState title="No homography investigation loaded" />;

  const sourceModule = activeDossier.modules.source_attribution;
  const homography = sourceModule.result?.homography;
  const formatNumber = (value: number | undefined, digits = 2) => value === undefined ? 'Unavailable' : value.toFixed(digits);
  const matrix = homography?.matrix_3x3;

  const exportHomographyJson = () => {
    const payload = {
      investigation_id: activeDossier.investigation_id,
      case_id: activeDossier.case_id,
      media_sha256: activeDossier.media.sha256,
      source_attribution_status: sourceModule.status,
      homography: homography || null,
      exported_at_utc: new Date().toISOString(),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `homography_${activeDossier.investigation_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
    showToast('Homography data exported.');
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">Crop alignment</p>
          <h1 className="text-2xl font-semibold tracking-tight text-gray-950">Optical homography</h1>
          <p className="text-sm text-gray-600">Only the homography returned in the current investigation record is shown.</p>
        </div>
        <button type="button" onClick={exportHomographyJson} className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-300 bg-white px-3.5 py-2 text-xs font-medium text-gray-700 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700">
          <Download className="h-4 w-4" /> Export JSON
        </button>
      </div>

      <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col justify-between gap-2 border-b border-gray-200 pb-3 sm:flex-row sm:items-center">
          <div><h2 className="text-sm font-semibold text-gray-950">Source attribution homography</h2><p className="mt-0.5 text-xs text-gray-500">Investigation {activeDossier.investigation_id}</p></div>
          <span className="rounded-full border border-gray-200 bg-gray-50 px-2 py-1 text-[11px] font-semibold text-gray-700">{sourceModule.status}</span>
        </div>

        {!homography ? (
          <div className="mt-5 flex items-start gap-2 rounded-lg border border-gray-200 bg-gray-50 p-4 text-xs text-gray-600">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-gray-500" />
            <span>Homography is unavailable because the current gateway response did not return a homography object.</span>
          </div>
        ) : (
          <div className="mt-5 space-y-5">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {[
                ['RANSAC inlier ratio', homography.ransac_inlier_ratio === undefined ? 'Unavailable' : `${(homography.ransac_inlier_ratio * 100).toFixed(2)}%`],
                ['RANSAC inlier count', homography.ransac_inlier_count?.toString() || 'Unavailable'],
                ['Perspective skew', `${formatNumber(homography.perspective_skew_deg)}°`],
                ['Area retention', homography.area_retention_pct === undefined ? 'Unavailable' : `${formatNumber(homography.area_retention_pct)}%`],
              ].map(([label, value]) => <div key={label} className="rounded-lg border border-gray-200 bg-gray-50 p-3"><div className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">{label}</div><div className="mt-1 font-mono text-sm font-medium text-gray-950">{value}</div></div>)}
            </div>

            <div className="grid gap-5 lg:grid-cols-[1fr_auto]">
              <div>
                <h3 className="mb-2 text-xs font-semibold text-gray-950">3 × 3 transformation matrix</h3>
                {matrix ? <div className="overflow-x-auto rounded-lg border border-gray-200 bg-gray-50 p-4"><table className="w-full min-w-80 border-collapse text-right font-mono text-sm"><tbody>{matrix.map((row, rowIndex) => <tr key={rowIndex}>{row.map((value, columnIndex) => <td key={columnIndex} className="border border-gray-200 px-4 py-2 text-gray-900">{typeof value === 'number' ? value.toFixed(4) : 'Unavailable'}</td>)}</tr>)}</tbody></table></div> : <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-xs text-gray-600">Matrix unavailable.</div>}
              </div>
              <div className="space-y-3 text-xs">
                <h3 className="text-xs font-semibold text-gray-950">Coordinate transform</h3>
                <div className="rounded-lg border border-gray-200 p-3 text-gray-600">Offset: <span className="font-mono text-gray-950">{homography.coordinate_offset ? `dx ${formatNumber(homography.coordinate_offset.dx)}, dy ${formatNumber(homography.coordinate_offset.dy)}` : 'Unavailable'}</span></div>
              </div>
            </div>
          </div>
        )}
      </section>

      <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-4 text-xs text-gray-700">
        <div className="flex items-center gap-2 font-semibold text-gray-950"><Info className="h-4 w-4 text-blue-700" />Current evidence</div>
        <div className="mt-1 break-all font-mono text-gray-600">{activeDossier.media.filename} · {activeDossier.media.sha256}</div>
      </div>
    </div>
  );
};
