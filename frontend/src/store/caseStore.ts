import { create } from 'zustand';
import type { CaseSession, GatewayCapabilities, InvestigationRecord } from '../types/contract';
import { SAMPLE_IMAGE_DOSSIER, SAMPLE_VIDEO_DOSSIER } from '../data/sampleDossiers';

export type TabKey = 'ingestion' | 'provenance' | 'homography' | 'sec65b' | 'audit';

export interface OfficerProfile {
  name: string;
  badgeId: string;
  station: string;
  role: string;
}

export const OFFICER_PRESETS: OfficerProfile[] = [
  {
    name: 'SI Vikramaditya Singh',
    badgeId: 'CY-7402',
    station: 'PS Cyber Crime, UT Chandigarh',
    role: 'Certified Forensic Examiner',
  },
  {
    name: 'Insp. Gurpreet Kaur',
    badgeId: 'CY-5190',
    station: 'State Cyber Crime Cell, Sector 17',
    role: 'Lead Investigating Officer',
  },
  {
    name: 'DSP Harinder Sandhu',
    badgeId: 'CY-1004',
    station: 'HQ Cyber Forensics & Intelligence',
    role: 'Supervisory Forensic Division Chief',
  },
];

interface CaseState {
  isAuthenticated: boolean;
  activeTab: TabKey;
  caseSession: CaseSession;
  capabilities: GatewayCapabilities;
  activeDossier: InvestigationRecord;
  isAnalyzing: boolean;
  analysisStep: string;
  toastMessage: string | null;

  login: (officer: OfficerProfile) => void;
  logout: () => void;
  setActiveTab: (tab: TabKey) => void;
  updateCaseSession: (session: Partial<CaseSession>) => void;
  setCapabilities: (caps: GatewayCapabilities) => void;
  setActiveDossier: (dossier: InvestigationRecord) => void;
  setIsAnalyzing: (analyzing: boolean, step?: string) => void;
  loadSampleDossier: (type: 'image' | 'video') => void;
  showToast: (msg: string) => void;
}

export const useCaseStore = create<CaseState>((set) => ({
  isAuthenticated: true, // starts authenticated for immediate workflow access, with seamless Sign Out / Switch anytime
  activeTab: 'ingestion',
  caseSession: {
    firNumber: 'FIR No. 142/2025',
    policeStation: OFFICER_PRESETS[0].station,
    officerInCharge: OFFICER_PRESETS[0].name,
    officerId: OFFICER_PRESETS[0].badgeId,
    judicialCourt: 'Court of Chief Judicial Magistrate, Chandigarh',
  },
  capabilities: {
    gateway: { status: 'ONLINE', port: 8000, service_health_timeout_sec: 3 },
    detection: { status: 'ONLINE', url: 'http://localhost:8001', device: 'cuda:0 (NVIDIA RTX)', image_ready: true, video_ready: true },
    prnu: { status: 'ONLINE', url: 'http://localhost:8002', reference_count: 8, synthetic_allowed: false },
    source_attribution: { status: 'READY', astar_ready: true, api_keys_configured: true },
  },
  activeDossier: SAMPLE_IMAGE_DOSSIER,
  isAnalyzing: false,
  analysisStep: '',
  toastMessage: null,

  login: (officer) =>
    set((state) => ({
      isAuthenticated: true,
      caseSession: {
        ...state.caseSession,
        officerInCharge: officer.name,
        officerId: officer.badgeId,
        policeStation: officer.station,
      },
      toastMessage: `Authenticated as ${officer.name} (${officer.badgeId})`,
    })),
  logout: () =>
    set({
      isAuthenticated: false,
      toastMessage: 'Signed out of forensic workstation',
    }),
  setActiveTab: (activeTab) => set({ activeTab }),
  updateCaseSession: (partial) =>
    set((state) => ({ caseSession: { ...state.caseSession, ...partial } })),
  setCapabilities: (capabilities) => set({ capabilities }),
  setActiveDossier: (activeDossier) => set({ activeDossier }),
  setIsAnalyzing: (isAnalyzing, analysisStep = '') => set({ isAnalyzing, analysisStep }),
  loadSampleDossier: (type) =>
    set({
      activeDossier: type === 'image' ? SAMPLE_IMAGE_DOSSIER : SAMPLE_VIDEO_DOSSIER,
      toastMessage: `Loaded ${type === 'image' ? 'Image Sub-Crop Exhibit (FIR 142/2025)' : 'Video Speech Deepfake Exhibit (FIR 089/2025)'}`,
    }),
  showToast: (toastMessage) => {
    set({ toastMessage });
    setTimeout(() => {
      set((state) => (state.toastMessage === toastMessage ? { toastMessage: null } : {}));
    }, 4000);
  },
}));
