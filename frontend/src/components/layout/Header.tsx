import React, { useState } from 'react';
import {
  Check,
  CircleDot,
  FileText,
  Film,
  GitBranch,
  Image as ImageIcon,
  ListTree,
  LogOut,
  Scan,
  UploadCloud,
  UserCheck,
} from 'lucide-react';
import { OFFICER_PRESETS, useCaseStore, type TabKey } from '../../store/caseStore';

export const Header: React.FC = () => {
  const {
    activeTab,
    setActiveTab,
    caseSession,
    activeDossier,
    dossierSource,
    capabilities,
    loadSampleDossier,
    login,
    logout,
  } = useCaseStore();
  const [showSwitchModal, setShowSwitchModal] = useState(false);

  const tabs: Array<{ key: TabKey; label: string; icon: React.ReactNode }> = [
    { key: 'ingestion', label: 'Ingest', icon: <UploadCloud className="h-4 w-4" /> },
    { key: 'provenance', label: 'Results', icon: <GitBranch className="h-4 w-4" /> },
    { key: 'homography', label: 'Homography', icon: <Scan className="h-4 w-4" /> },
    { key: 'sec65b', label: 'Report', icon: <FileText className="h-4 w-4" /> },
    { key: 'audit', label: 'Custody', icon: <ListTree className="h-4 w-4" /> },
  ];
  const gatewayOnline = capabilities.gateway?.status === 'ONLINE';

  return (
    <>
      <header className="sticky top-0 z-30 border-b border-gray-200 bg-white/95 shadow-sm backdrop-blur no-print">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            <div className="shrink-0">
              <div className="text-base font-bold tracking-tight text-gray-950">Pramaan-X</div>
              <div className="text-[11px] text-gray-500">Multimedia provenance suite</div>
            </div>
            <span className="hidden h-8 w-px bg-gray-200 sm:block" />
            <div className="hidden items-center gap-1.5 text-xs text-gray-600 sm:flex">
              <CircleDot className={`h-3.5 w-3.5 ${gatewayOnline ? 'text-emerald-700' : 'text-amber-600'}`} />
              <span>Gateway :8000</span>
              <span className={gatewayOnline ? 'font-medium text-emerald-700' : 'font-medium text-amber-700'}>
                {gatewayOnline ? 'connected' : 'offline'}
              </span>
            </div>
          </div>

          <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs">
            <div className="rounded-md border border-gray-200 bg-gray-50 px-3 py-1.5 text-gray-600">
              <span className="mr-1.5 text-gray-500">Case</span>
              <span className="font-mono font-medium text-gray-950">{caseSession.firNumber}</span>
            </div>
            <div className="flex items-center gap-2 text-gray-600">
              <span className="hidden xl:inline">{caseSession.officerInCharge} ({caseSession.officerId})</span>
              <button
                type="button"
                onClick={() => setShowSwitchModal(true)}
                className="rounded-md border border-gray-200 px-2 py-1.5 text-gray-600 transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
              >
                Switch officer
              </button>
              <button
                type="button"
                onClick={logout}
                className="rounded-md p-1.5 text-gray-500 transition-colors hover:bg-red-50 hover:text-red-700"
                title="Sign out"
                aria-label="Sign out"
              >
                <LogOut className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 pb-3 lg:flex-row lg:items-center lg:justify-between">
          <nav className="flex min-w-0 gap-1 overflow-x-auto rounded-lg border border-gray-200 bg-gray-50 p-1" aria-label="Investigation sections">
            {tabs.map((tab) => {
              const selected = activeTab === tab.key;
              return (
                <button
                  type="button"
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  aria-current={selected ? 'page' : undefined}
                  className={`flex shrink-0 items-center gap-2 rounded-md px-3 py-2 text-xs font-medium transition-colors ${
                    selected ? 'bg-white text-blue-700 shadow-sm ring-1 ring-gray-200' : 'text-gray-600 hover:bg-white hover:text-gray-950'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              );
            })}
          </nav>

          <div className="flex shrink-0 items-center gap-1.5 rounded-lg border border-gray-200 bg-white p-1 text-xs">
            <span className="px-2 text-[10px] font-semibold uppercase tracking-wide text-gray-400">Samples</span>
            <button
              type="button"
              onClick={() => loadSampleDossier('image')}
              className={`flex items-center gap-1 rounded-md px-2 py-1.5 transition-colors ${
                dossierSource === 'sample' && activeDossier?.media.media_type === 'image'
                  ? 'bg-blue-50 font-semibold text-blue-700'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-950'
              }`}
              title="Load the bundled image sample"
            >
              <ImageIcon className="h-3.5 w-3.5" />
              Image
            </button>
            <button
              type="button"
              onClick={() => loadSampleDossier('video')}
              className={`flex items-center gap-1 rounded-md px-2 py-1.5 transition-colors ${
                dossierSource === 'sample' && activeDossier?.media.media_type === 'video'
                  ? 'bg-blue-50 font-semibold text-blue-700'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-950'
              }`}
              title="Load the bundled video sample"
            >
              <Film className="h-3.5 w-3.5" />
              Video
            </button>
          </div>
        </div>
      </header>

      {showSwitchModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-950/30 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md space-y-4 rounded-2xl border border-gray-200 bg-white p-6 shadow-xl">
            <div className="flex items-center gap-3 border-b border-gray-200 pb-3">
              <UserCheck className="h-5 w-5 text-blue-700" />
              <div>
                <h2 className="text-sm font-semibold text-gray-950">Switch officer in-charge</h2>
                <p className="text-xs text-gray-500">The selected officer is used in the case context and report.</p>
              </div>
            </div>
            <div className="space-y-2">
              {OFFICER_PRESETS.map((preset) => {
                const selected = preset.badgeId === caseSession.officerId;
                return (
                  <button
                    type="button"
                    key={preset.badgeId}
                    onClick={() => {
                      login(preset);
                      setShowSwitchModal(false);
                    }}
                    className={`flex w-full items-center justify-between rounded-lg border p-3 text-left transition-colors ${
                      selected ? 'border-blue-300 bg-blue-50' : 'border-gray-200 hover:border-blue-200 hover:bg-gray-50'
                    }`}
                  >
                    <span>
                      <span className="block text-xs font-semibold text-gray-950">{preset.name} ({preset.badgeId})</span>
                      <span className="mt-0.5 block text-[11px] text-gray-500">{preset.role} · {preset.station}</span>
                    </span>
                    {selected && <Check className="h-4 w-4 text-blue-700" />}
                  </button>
                );
              })}
            </div>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setShowSwitchModal(false)}
                className="rounded-md border border-gray-200 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50 hover:text-gray-950"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
