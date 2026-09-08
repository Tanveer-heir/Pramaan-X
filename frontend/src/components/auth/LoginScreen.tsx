import React, { useState } from 'react';
import { 
  ShieldCheck, 
  ArrowRight, 
  Lock, 
  BadgeCheck, 
  KeyRound
} from 'lucide-react';
import { useCaseStore, OFFICER_PRESETS, type OfficerProfile } from '../../store/caseStore';

export const LoginScreen: React.FC = () => {
  const { login } = useCaseStore();
  const [selectedPreset, setSelectedPreset] = useState<OfficerProfile>(OFFICER_PRESETS[0]);
  const [pin, setPin] = useState<string>('7402');
  const [customMode, setCustomMode] = useState<boolean>(false);
  const [customName, setCustomName] = useState<string>('');
  const [customBadge, setCustomBadge] = useState<string>('');
  const [customStation, setCustomStation] = useState<string>('PS Cyber Crime, UT Chandigarh');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (customMode) {
      if (!customName.trim() || !customBadge.trim()) return;
      login({
        name: customName.trim(),
        badgeId: customBadge.trim(),
        station: customStation.trim(),
        role: 'Authorized Forensic Examiner',
      });
    } else {
      login(selectedPreset);
    }
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-[#f4f4f5] flex flex-col justify-center items-center px-4 py-12 selection:bg-[#27272a] selection:text-[#ffffff]">
      <div className="w-full max-w-lg space-y-8">
        {/* Department & Suite Branding */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-[#18181b] border border-[#27272a] text-[#f4f4f5] mb-2 shadow-sm">
            <ShieldCheck className="w-6 h-6 text-emerald-400" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-[#f4f4f5]">
            Pramaan-X Workstation
          </h1>
          <p className="text-xs text-[#a1a1aa] font-medium">
            Chandigarh Police • Cyber Crime Investigation Division
          </p>
          <div className="inline-block px-3 py-1 rounded-full bg-[#18181b] border border-[#27272a] text-[11px] font-mono text-[#71717a] mt-1">
            Restricted Law Enforcement & Judicial Terminal
          </div>
        </div>

        {/* Authentication Card */}
        <div className="bg-[#121215] border border-[#27272a] rounded-2xl p-8 shadow-2xl space-y-6">
          <div className="flex items-center justify-between border-b border-[#27272a] pb-4">
            <div className="flex items-center gap-2">
              <KeyRound className="w-4 h-4 text-[#a1a1aa]" />
              <span className="text-sm font-medium text-[#f4f4f5]">
                {customMode ? 'Custom Officer Login' : 'Officer Identity Verification'}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setCustomMode(!customMode)}
              className="text-xs text-[#71717a] hover:text-[#f4f4f5] transition-colors cursor-pointer"
            >
              {customMode ? 'Use Presets' : 'Custom Input'}
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {!customMode ? (
              <div className="space-y-3">
                <label className="text-xs text-[#a1a1aa] font-medium block">
                  Select Officer Identity (1-Click Presets for Demo)
                </label>
                <div className="space-y-2">
                  {OFFICER_PRESETS.map((preset) => {
                    const isSelected = selectedPreset.badgeId === preset.badgeId;
                    return (
                      <div
                        key={preset.badgeId}
                        onClick={() => {
                          setSelectedPreset(preset);
                          setPin(preset.badgeId.replace('CY-', ''));
                        }}
                        className={`p-3.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                          isSelected
                            ? 'bg-[#18181b] border-[#52525b] shadow-sm ring-1 ring-[#52525b]'
                            : 'bg-[#121215] border-[#27272a] hover:border-[#3f3f46]'
                        }`}
                      >
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold text-[#f4f4f5]">
                              {preset.name}
                            </span>
                            <span className="text-[10px] font-mono bg-[#27272a] text-[#a1a1aa] px-2 py-0.5 rounded">
                              {preset.badgeId}
                            </span>
                          </div>
                          <div className="text-[11px] text-[#71717a]">
                            {preset.role} • {preset.station}
                          </div>
                        </div>

                        {isSelected && (
                          <BadgeCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="text-xs text-[#a1a1aa] font-medium block">Officer Name & Rank</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. SI Vikramaditya Singh"
                    value={customName}
                    onChange={(e) => setCustomName(e.target.value)}
                    className="w-full bg-[#18181b] border border-[#27272a] rounded-lg px-3.5 py-2 text-xs text-[#f4f4f5] focus:border-[#71717a] focus:outline-none"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs text-[#a1a1aa] font-medium block">Badge / Officer ID</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. CY-7402"
                    value={customBadge}
                    onChange={(e) => setCustomBadge(e.target.value)}
                    className="w-full bg-[#18181b] border border-[#27272a] rounded-lg px-3.5 py-2 text-xs text-[#f4f4f5] font-mono focus:border-[#71717a] focus:outline-none"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs text-[#a1a1aa] font-medium block">Station / Unit</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. PS Cyber Crime, UT Chandigarh"
                    value={customStation}
                    onChange={(e) => setCustomStation(e.target.value)}
                    className="w-full bg-[#18181b] border border-[#27272a] rounded-lg px-3.5 py-2 text-xs text-[#f4f4f5] focus:border-[#71717a] focus:outline-none"
                  />
                </div>
              </div>
            )}

            {/* Access Code / PIN Input */}
            <div className="space-y-1.5">
              <label className="text-xs text-[#a1a1aa] font-medium block">
                Forensic Terminal Access PIN
              </label>
              <div className="relative">
                <Lock className="w-3.5 h-3.5 text-[#71717a] absolute left-3.5 top-3" />
                <input
                  type="password"
                  value={pin}
                  onChange={(e) => setPin(e.target.value)}
                  placeholder="Enter 4-digit PIN"
                  className="w-full bg-[#18181b] border border-[#27272a] rounded-lg pl-9 pr-3.5 py-2 text-xs text-[#f4f4f5] font-mono tracking-widest focus:border-[#71717a] focus:outline-none"
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full py-3 px-4 rounded-xl bg-[#f4f4f5] hover:bg-[#ffffff] text-[#09090b] font-semibold text-xs flex items-center justify-center gap-2 cursor-pointer transition-colors shadow-md mt-2"
            >
              <span>Authenticate & Open Workstation</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </form>

          <div className="pt-2 border-t border-[#27272a] text-[11px] text-[#71717a] flex items-center justify-between font-mono">
            <span>Security: ICJS/CCTNS Mock</span>
            <span className="text-emerald-400">Direct Ingress</span>
          </div>
        </div>

        {/* Disclaimer Footer */}
        <p className="text-center text-xs text-[#52525b] max-w-sm mx-auto leading-relaxed">
          Authorized for Chandigarh Police Cyber Crime Division personnel and court-appointed forensic examiners.
        </p>
      </div>
    </div>
  );
};
