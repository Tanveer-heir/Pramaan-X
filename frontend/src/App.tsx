import React, { useEffect } from 'react';
import { useCaseStore } from './store/caseStore';
import { Header } from './components/layout/Header';
import { LoginScreen } from './components/auth/LoginScreen';
import { Tab1Ingestion } from './components/tabs/Tab1Ingestion';
import { Tab2Provenance } from './components/tabs/Tab2Provenance';
import { Tab3Homography } from './components/tabs/Tab3Homography';
import { Tab4Section65B } from './components/tabs/Tab4Section65B';
import { Tab5AuditLedger } from './components/tabs/Tab5AuditLedger';
import { CheckCircle2 } from 'lucide-react';

export const App: React.FC = () => {
  const { isAuthenticated, activeTab, toastMessage, previewUrl } = useCaseStore();

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  if (!isAuthenticated) {
    return <LoginScreen />;
  }

  return (
    <div className="min-h-screen bg-white text-gray-950 flex flex-col font-sans selection:bg-blue-100 selection:text-gray-950">
      <Header />

      {/* Main Spacious Content Canvas */}
      <main className="flex-1 pb-20">
        {activeTab === 'ingestion' && <Tab1Ingestion />}
        {activeTab === 'provenance' && <Tab2Provenance />}
        {activeTab === 'homography' && <Tab3Homography />}
        {activeTab === 'sec65b' && <Tab4Section65B />}
        {activeTab === 'audit' && <Tab5AuditLedger />}
      </main>

      {/* Minimal, Professional Footer */}
      <footer className="w-full bg-white border-t border-gray-200 py-6 text-xs text-gray-500 no-print">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-gray-950">Pramaan-X</span>
            <span>•</span>
            <span>Chandigarh Police Cyber Crime Investigation Division</span>
          </div>

          <div className="font-mono text-[11px] text-gray-500 flex items-center gap-3">
            <span>ISO/IEC 27037:2012</span>
            <span>•</span>
            <span>Section 65B(4) IEA & Section 63 BSA 2023 Compliant</span>
          </div>
        </div>
      </footer>

      {/* Toast Notification HUD */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 bg-white border border-blue-200 text-gray-950 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 text-xs font-medium animate-fade-in no-print">
          <CheckCircle2 className="w-4 h-4 text-blue-700 shrink-0" />
          <span>{toastMessage}</span>
        </div>
      )}
    </div>
  );
};

export default App;
