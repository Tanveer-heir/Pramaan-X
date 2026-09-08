import React, { useState } from 'react';
import { 
  UploadCloud, 
  GitBranch, 
  Scan, 
  FileText, 
  ListTree,
  Image as ImageIcon,
  Film,
  CircleDot,
  LogOut,
  UserCheck,
  Check
} from 'lucide-react';
import { useCaseStore, OFFICER_PRESETS, type TabKey } from '../../store/caseStore';

export const Header: React.FC = () => {
  const { 
    activeTab, 
    setActiveTab, 
    caseSession, 
    activeDossier, 
    loadSampleDossier, 
    capabilities,
    login,
    logout 
  } = useCaseStore();

  const [showSwitchModal, setShowSwitchModal] = useState<boolean>(false);

  const tabs: { key: TabKey; label: string; icon: React.ReactNode }[] = [
    { key: 'ingestion', label: 'Evidence Ingestion', icon: <UploadCloud className="w-4 h-4" /> },
    { key: 'provenance', label: 'Origin & Lineage', icon: <GitBranch className="w-4 h-4" /> },
    { key: 'homography', label: 'Crop Homography', icon: <Scan className="w-4 h-4" /> },
    { key: 'sec65b', label: 'Court Affidavit (65B)', icon: <FileText className="w-4 h-4" /> },
    { key: 'audit', label: 'Chain of Custody', icon: <ListTree className="w-4 h-4" /> },
  ];

  const isGatewayOnline = capabilities.gateway?.status === 'ONLINE';

  return (
    <>
      <header className="w-full bg-[#121215] border-b border-[#27272a] sticky top-0 z-30 no-print select-none">
        {/* Upper Subtle Status Bar */}
        <div className="max-w-7xl mx-auto px-6 py-2.5 flex items-center justify-between text-xs border-b border-[#1f1f23]">
          <div className="flex items-center gap-3 text-[#a1a1aa]">
            <span className="text-[#f4f4f5] font-semibold tracking-tight">Pramaan-X</span>
            <span className="text-[#3f3f46]">/</span>
            <span>Multimedia Provenance Suite</span>
            <span className="text-[#3f3f46]">/</span>
            <span className="text-[#71717a]">Chandigarh Police Cyber Crime Division</span>
          </div>

          <div className="flex items-center gap-4 text-[#a1a1aa]">
            <div className="flex items-center gap-1.5 font-mono text-[11px]">
              <CircleDot className={`w-3 h-3 ${isGatewayOnline ? 'text-emerald-500' : 'text-amber-500'}`} />
              <span className="text-[#71717a]">Gateway :8000</span>
              <span className={isGatewayOnline ? 'text-emerald-400' : 'text-amber-400'}>
                {isGatewayOnline ? 'Connected' : 'Standalone'}
              </span>
            </div>

            <span className="text-[#3f3f46]">|</span>

            <div className="flex items-center gap-2">
              <span className="text-[#71717a]">Officer:</span>
              <span className="text-[#f4f4f5] font-medium">{caseSession.officerInCharge} ({caseSession.officerId})</span>
              <button
                onClick={() => setShowSwitchModal(true)}
                className="text-[11px] text-[#a1a1aa] hover:text-[#f4f4f5] px-1.5 py-0.5 rounded hover:bg-[#18181b] transition-colors cursor-pointer"
                title="Switch officer identity"
              >
                [Switch]
              </button>
              <button
                onClick={logout}
                className="text-[#71717a] hover:text-rose-400 p-1 rounded hover:bg-[#18181b] transition-colors cursor-pointer ml-1"
                title="Sign out of workstation"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Main Navigation & Case Context */}
        <div className="max-w-7xl mx-auto px-6 py-3.5 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* Modern Segmented Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-[#18181b] p-1 rounded-lg border border-[#27272a] overflow-x-auto scrollbar-none">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer whitespace-nowrap ${
                    isActive
                      ? 'bg-[#27272a] text-[#f4f4f5] shadow-sm font-semibold'
                      : 'text-[#a1a1aa] hover:text-[#f4f4f5] hover:bg-[#222226]'
                  }`}
                >
                  <span className={isActive ? 'text-[#f4f4f5]' : 'text-[#71717a]'}>
                    {tab.icon}
                  </span>
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Case Reference & Sample Switcher */}
          <div className="flex items-center gap-3 text-xs">
            <div className="bg-[#18181b] border border-[#27272a] px-3 py-1.5 rounded-md flex items-center gap-2">
              <span className="text-[#71717a]">Active Case:</span>
              <span className="text-[#f4f4f5] font-mono font-medium">{caseSession.firNumber}</span>
            </div>

            <div className="bg-[#18181b] border border-[#27272a] p-0.5 rounded-md flex items-center gap-0.5">
              <button
                onClick={() => loadSampleDossier('image')}
                title="Load Image Sub-Crop Forensic Dossier"
                className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
                  activeDossier.media.media_type === 'image'
                    ? 'bg-[#27272a] text-[#f4f4f5] font-medium'
                    : 'text-[#a1a1aa] hover:text-[#f4f4f5]'
                }`}
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Image Exhibit</span>
              </button>
              <button
                onClick={() => loadSampleDossier('video')}
                title="Load Video Voice Deepfake Forensic Dossier"
                className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
                  activeDossier.media.media_type === 'video'
                    ? 'bg-[#27272a] text-[#f4f4f5] font-medium'
                    : 'text-[#a1a1aa] hover:text-[#f4f4f5]'
                }`}
              >
                <Film className="w-3.5 h-3.5" />
                <span>Video Exhibit</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Officer Switcher Modal */}
      {showSwitchModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-[#121215] border border-[#27272a] rounded-xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex items-center gap-3 border-b border-[#27272a] pb-3">
              <UserCheck className="w-5 h-5 text-emerald-400" />
              <div>
                <h3 className="text-sm font-semibold text-[#f4f4f5]">
                  Switch Officer In-Charge
                </h3>
                <p className="text-xs text-[#71717a]">
                  Synchronize credentials with active case and court affidavit
                </p>
              </div>
            </div>

            <div className="space-y-2">
              {OFFICER_PRESETS.map((preset) => {
                const isSelected = caseSession.officerId === preset.badgeId;
                return (
                  <button
                    key={preset.badgeId}
                    onClick={() => {
                      login(preset);
                      setShowSwitchModal(false);
                    }}
                    className={`w-full text-left p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between ${
                      isSelected
                        ? 'bg-[#18181b] border-[#52525b] ring-1 ring-[#52525b]'
                        : 'bg-[#121215] border-[#27272a] hover:border-[#3f3f46]'
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="text-xs font-semibold text-[#f4f4f5]">
                        {preset.name} ({preset.badgeId})
                      </div>
                      <div className="text-[11px] text-[#71717a]">
                        {preset.role} • {preset.station}
                      </div>
                    </div>
                    {isSelected && <Check className="w-4 h-4 text-emerald-400" />}
                  </button>
                );
              })}
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowSwitchModal(false)}
                className="px-3.5 py-1.5 rounded bg-[#18181b] hover:bg-[#222226] text-xs text-[#a1a1aa] hover:text-[#f4f4f5] cursor-pointer"
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
