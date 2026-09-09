import React, { useState } from 'react';
import { ArrowRight, BadgeCheck, KeyRound, Lock, ShieldCheck } from 'lucide-react';
import { OFFICER_PRESETS, useCaseStore, type OfficerProfile } from '../../store/caseStore';

export const LoginScreen: React.FC = () => {
  const { login } = useCaseStore();
  const [selectedPreset, setSelectedPreset] = useState<OfficerProfile>(OFFICER_PRESETS[0]);
  const [pin, setPin] = useState('7402');
  const [customMode, setCustomMode] = useState(false);
  const [customName, setCustomName] = useState('');
  const [customBadge, setCustomBadge] = useState('');
  const [customStation, setCustomStation] = useState('PS Cyber Crime, UT Chandigarh');

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (customMode) {
      if (!customName.trim() || !customBadge.trim()) return;
      login({ name: customName.trim(), badgeId: customBadge.trim(), station: customStation.trim(), role: 'Authorized Forensic Examiner' });
    } else {
      login(selectedPreset);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 text-gray-950">
      <div className="w-full max-w-lg space-y-7">
        <div className="space-y-2 text-center"><div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-blue-700 text-white shadow-sm"><ShieldCheck className="h-6 w-6" /></div><h1 className="text-2xl font-semibold tracking-tight">Pramaan-X Workstation</h1><p className="text-xs font-medium text-gray-600">Chandigarh Police · Cyber Crime Investigation Division</p><div className="inline-block rounded-full border border-gray-200 bg-white px-3 py-1 text-[11px] text-gray-500">Restricted forensic terminal</div></div>

        <div className="space-y-6 rounded-2xl border border-gray-200 bg-white p-7 shadow-sm">
          <div className="flex items-center justify-between border-b border-gray-200 pb-4"><div className="flex items-center gap-2 text-sm font-semibold"><KeyRound className="h-4 w-4 text-blue-700" />{customMode ? 'Custom officer login' : 'Officer identity verification'}</div><button type="button" onClick={() => setCustomMode(!customMode)} className="text-xs text-blue-700 hover:underline">{customMode ? 'Use presets' : 'Custom input'}</button></div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {!customMode ? <div className="space-y-3"><label className="block text-xs font-medium text-gray-700">Select officer identity</label><div className="space-y-2">{OFFICER_PRESETS.map((preset) => { const selected = selectedPreset.badgeId === preset.badgeId; return <button type="button" key={preset.badgeId} onClick={() => { setSelectedPreset(preset); setPin(preset.badgeId.replace('CY-', '')); }} className={`flex w-full items-center justify-between rounded-xl border p-3.5 text-left transition-colors ${selected ? 'border-blue-300 bg-blue-50' : 'border-gray-200 hover:border-blue-200 hover:bg-gray-50'}`}><span><span className="flex items-center gap-2 text-xs font-semibold text-gray-950">{preset.name}<span className="rounded bg-white px-2 py-0.5 font-mono text-[10px] text-gray-600 ring-1 ring-gray-200">{preset.badgeId}</span></span><span className="mt-0.5 block text-[11px] text-gray-500">{preset.role} · {preset.station}</span></span>{selected && <BadgeCheck className="h-4 w-4 shrink-0 text-blue-700" />}</button>; })}</div></div> : <div className="space-y-4">{[['Officer name & rank', customName, setCustomName, 'e.g. SI Vikramaditya Singh'], ['Badge / officer ID', customBadge, setCustomBadge, 'e.g. CY-7402'], ['Station / unit', customStation, setCustomStation, 'e.g. PS Cyber Crime, UT Chandigarh']].map(([label, value, setter, placeholder]) => <label key={label as string} className="block space-y-1.5 text-xs font-medium text-gray-700">{label as string}<input type="text" required value={value as string} onChange={(event) => (setter as React.Dispatch<React.SetStateAction<string>>)(event.target.value)} placeholder={placeholder as string} className="w-full rounded-lg border border-gray-300 px-3.5 py-2 text-xs text-gray-950 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-100" /></label>)}</div>}

            <label className="block space-y-1.5 text-xs font-medium text-gray-700">Forensic terminal access PIN<div className="relative"><Lock className="absolute left-3.5 top-2.5 h-3.5 w-3.5 text-gray-400" /><input type="password" value={pin} onChange={(event) => setPin(event.target.value)} placeholder="Enter PIN" className="w-full rounded-lg border border-gray-300 py-2 pl-9 pr-3.5 text-xs font-mono tracking-widest text-gray-950 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-100" /></div></label>
            <button type="submit" className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-700 px-4 py-3 text-xs font-semibold text-white transition-colors hover:bg-blue-800"><span>Authenticate & open workstation</span><ArrowRight className="h-3.5 w-3.5" /></button>
          </form>
        </div>
      </div>
    </div>
  );
};
