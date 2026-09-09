import React, { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  Check,
  CheckCircle2,
  Clock,
  Copy,
  FileCheck,
  FileCode,
  Globe,
  Shield,
  UploadCloud,
  X,
} from 'lucide-react';
import { useCaseStore } from '../../store/caseStore';
import { computeFileSha256, fetchCapabilities, uploadEvidence } from '../../services/gateway';

export const Tab1Ingestion: React.FC = () => {
  const {
    caseSession,
    updateCaseSession,
    setCapabilities,
    selectedFile,
    previewUrl,
    setSelectedEvidence,
    clearSelectedEvidence,
    clearInvestigationResult,
    setInvestigationResult,
    setActiveTab,
    showToast,
    isAnalyzing,
    setIsAnalyzing,
    analysisStep,
  } = useCaseStore();

  const [fileHash, setFileHash] = useState<string>('');
  const [remoteUrl, setRemoteUrl] = useState<string>('');
  const [isHashing, setIsHashing] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [copiedHash, setCopiedHash] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isImageFile = selectedFile
    ? selectedFile.type.startsWith('image/') || /\.(jpe?g|png|webp)$/i.test(selectedFile.name)
    : false;
  const isVideoFile = selectedFile
    ? selectedFile.type.startsWith('video/') || /\.(mp4|mov|m4v|avi|mkv|webm)$/i.test(selectedFile.name)
    : false;

  useEffect(() => {
    fetchCapabilities().then(setCapabilities);
  }, [setCapabilities]);

  const handleFileSelect = async (file: File) => {
    const preview = URL.createObjectURL(file);
    clearInvestigationResult();
    setSelectedEvidence(file, preview);
    setUploadError(null);
    setFileHash('');
    setIsHashing(true);

    try {
      setFileHash(await computeFileSha256(file));
    } catch (error) {
      console.error('[Pramaan-X] Hash calculation failed:', error);
      setFileHash('CALCULATION_ERROR');
    } finally {
      setIsHashing(false);
    }
  };

  const handleRemoveFile = () => {
    clearSelectedEvidence();
    setFileHash('');
    setUploadError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const copyHashToClipboard = () => {
    if (!fileHash) return;
    void navigator.clipboard.writeText(fileHash);
    setCopiedHash(true);
    setTimeout(() => setCopiedHash(false), 2000);
    showToast('SHA-256 hash copied to clipboard');
  };

  const executeForensicAnalysis = async () => {
    if (!selectedFile) {
      showToast('Attach an evidence file before starting the gateway analysis.');
      return;
    }

    clearInvestigationResult();
    setIsAnalyzing(true, 'Preparing the evidence for the CPH Gateway (:8000)…');
    setUploadError(null);

    try {
      setIsAnalyzing(true, 'Dispatching the evidence to the CPH Gateway (:8000)…');
      const dossier = await uploadEvidence(selectedFile, caseSession.firNumber);
      setInvestigationResult(dossier);
      showToast(
        dossier.status === 'PARTIAL'
          ? `Investigation ${dossier.investigation_id} completed partially.`
          : `Investigation ${dossier.investigation_id} completed.`,
      );
      setActiveTab('provenance');
    } catch (error) {
      console.error('[Pramaan-X] Evidence upload failed:', error);
      setUploadError(error instanceof Error ? error.message : 'Gateway connection error. Verify port 8000 is listening.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-6 py-8">
      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">Evidence ingestion</p>
        <h1 className="text-2xl font-semibold tracking-tight text-gray-950">Submit current evidence for analysis</h1>
        <p className="max-w-3xl text-sm leading-6 text-gray-600">
          Hash the selected file in the browser, submit it to the CPH Gateway on port 8000, and review the returned investigation record across the tabs.
        </p>
      </div>

      <section className="space-y-6 rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col justify-between gap-3 border-b border-gray-200 pb-4 sm:flex-row sm:items-center">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-950">
            <FileCode className="h-4 w-4 text-blue-700" />
            Select evidence file
          </div>
          <span className="text-xs text-gray-500">Client-side SHA-256 · ISO/IEC 27037:2012</span>
        </div>

        <div
          onDrop={(event) => {
            event.preventDefault();
            const file = event.dataTransfer.files[0];
            if (file) void handleFileSelect(file);
          }}
          onDragOver={(event) => event.preventDefault()}
          onClick={() => !selectedFile && fileInputRef.current?.click()}
          className={`rounded-xl border border-dashed p-8 transition-colors ${
            selectedFile
              ? 'border-blue-200 bg-blue-50/30'
              : 'cursor-pointer border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50/40'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void handleFileSelect(file);
            }}
          />

          {!selectedFile ? (
            <div className="mx-auto max-w-lg space-y-4 py-8 text-center">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-blue-100 text-blue-700">
                <UploadCloud className="h-7 w-7" />
              </div>
              <div className="space-y-1">
                <div className="text-sm font-semibold text-gray-950">Drop an image or video here, or browse</div>
                <div className="text-xs text-gray-500">JPG, PNG, WEBP, MP4, MOV, MKV, or WEBM</div>
              </div>
              <div className="inline-flex items-center gap-1.5 rounded-full border border-gray-200 bg-white px-3 py-1 text-[11px] text-gray-600">
                <Shield className="h-3 w-3 text-emerald-700" />
                The original file is sent only to the configured gateway endpoint.
              </div>
            </div>
          ) : (
            <div className="space-y-5" onClick={(event) => event.stopPropagation()}>
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
                    <FileCheck className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="break-all font-mono text-sm font-semibold text-gray-950">{selectedFile.name}</div>
                    <div className="mt-0.5 text-xs text-gray-500">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB · {selectedFile.type || 'Binary media'}
                    </div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleRemoveFile}
                  className="rounded-md p-1.5 text-gray-500 transition-colors hover:bg-red-50 hover:text-red-700"
                  title="Remove evidence"
                  aria-label="Remove evidence"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              {previewUrl && (
                <div className="flex min-h-48 items-center justify-center overflow-hidden rounded-lg border border-gray-200 bg-gray-50 p-2">
                  {isVideoFile ? (
                    <video src={previewUrl} controls className="max-h-80 w-auto rounded object-contain" />
                  ) : isImageFile ? (
                    <img src={previewUrl} alt={`Preview of ${selectedFile.name}`} className="max-h-80 w-auto rounded object-contain" />
                  ) : (
                    <div className="px-4 py-10 text-center text-xs text-amber-700">Browser preview is unavailable for this file type.</div>
                  )}
                </div>
              )}

              <div className="flex items-center justify-between gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3">
                <div className="min-w-0 space-y-0.5">
                  <span className="block text-[11px] font-medium uppercase tracking-wider text-gray-500">Ingress SHA-256</span>
                  <div className="truncate font-mono text-xs text-emerald-700 select-all">
                    {isHashing ? 'Computing cryptographic hash…' : fileHash || 'Unavailable'}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={copyHashToClipboard}
                  disabled={!fileHash || isHashing}
                  className="shrink-0 rounded p-2 text-gray-500 transition-colors hover:bg-white hover:text-gray-950 disabled:cursor-not-allowed disabled:opacity-40"
                  title="Copy SHA-256"
                  aria-label="Copy SHA-256"
                >
                  {copiedHash ? <Check className="h-4 w-4 text-emerald-700" /> : <Copy className="h-4 w-4" />}
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <label htmlFor="remote-url" className="flex items-center gap-1.5 text-xs font-medium text-gray-700">
            <Globe className="h-3.5 w-3.5" />
            Optional source reference
          </label>
          <input
            id="remote-url"
            type="url"
            placeholder="https://example.org/source-post"
            value={remoteUrl}
            onChange={(event) => setRemoteUrl(event.target.value)}
            className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-xs text-gray-950 placeholder:text-gray-400 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-100"
          />
          <p className="text-xs text-gray-500">Stored as a reference for now; the current gateway contract accepts file uploads only.</p>
        </div>
      </section>

      <section className="space-y-5 rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="border-b border-gray-200 pb-4">
          <h2 className="text-sm font-semibold text-gray-950">Case metadata & custody assignment</h2>
          <p className="mt-1 text-xs text-gray-500">These values travel with the case context shown in the report and ledger.</p>
        </div>
        <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
          <label className="space-y-2 text-xs font-medium text-gray-700">
            FIR / case reference
            <input
              type="text"
              value={caseSession.firNumber}
              onChange={(event) => updateCaseSession({ firNumber: event.target.value })}
              className="w-full rounded-lg border border-gray-300 px-3.5 py-2 text-xs font-mono text-gray-950 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-100"
            />
          </label>
          <label className="space-y-2 text-xs font-medium text-gray-700">
            Police station / unit
            <input
              type="text"
              value={caseSession.policeStation}
              onChange={(event) => updateCaseSession({ policeStation: event.target.value })}
              className="w-full rounded-lg border border-gray-300 px-3.5 py-2 text-xs text-gray-950 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-100"
            />
          </label>
          <label className="space-y-2 text-xs font-medium text-gray-700">
            Forensic examiner in-charge
            <input
              type="text"
              value={caseSession.officerInCharge}
              onChange={(event) => updateCaseSession({ officerInCharge: event.target.value })}
              className="w-full rounded-lg border border-gray-300 px-3.5 py-2 text-xs text-gray-950 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-100"
            />
          </label>
        </div>
        <div className="flex flex-col justify-between gap-3 rounded-lg border border-blue-100 bg-blue-50/50 p-4 text-xs text-gray-700 sm:flex-row sm:items-center">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-700" />
            <span>Standard: ISO/IEC 27037:2012</span>
          </div>
          <span>Admissibility: Section 65B(4) IEA & Section 63 BSA 2023</span>
        </div>
      </section>

      {remoteUrl && !selectedFile && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>Select a local evidence file to run the current `/api/v1/investigate/upload` gateway contract.</span>
        </div>
      )}

      {uploadError && (
        <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-xs text-red-900">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
          <div className="space-y-1">
            <div className="font-semibold">Gateway communication notice</div>
            <div>{uploadError}</div>
            <div className="text-red-800/80">Start the CPH Gateway on port 8000, then submit this same file again.</div>
          </div>
        </div>
      )}

      <button
        type="button"
        onClick={() => void executeForensicAnalysis()}
        disabled={isAnalyzing || !selectedFile}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-700 px-6 py-4 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-500"
      >
        {isAnalyzing ? <Clock className="h-4 w-4 animate-spin" /> : <ArrowRight className="h-4 w-4" />}
        <span>{isAnalyzing ? 'Analyzing current evidence…' : 'Run multi-modal forensic analysis'}</span>
      </button>

      {isAnalyzing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-950/30 p-6 backdrop-blur-sm">
          <div className="w-full max-w-md space-y-5 rounded-2xl border border-gray-200 bg-white p-7 shadow-xl">
            <div className="flex items-center gap-3">
              <Clock className="h-5 w-5 animate-spin text-blue-700" />
              <div>
                <h3 className="text-base font-semibold text-gray-950">Forensic pipeline in progress</h3>
                <p className="text-xs text-gray-500">The previous result is cleared until this gateway response arrives.</p>
              </div>
            </div>
            <div className="space-y-2 rounded-lg border border-gray-200 bg-gray-50 p-4 text-xs text-gray-700">
              <div className="flex items-center gap-2 font-medium text-gray-950">
                <span className="h-2 w-2 animate-pulse rounded-full bg-blue-700" />
                <span>{analysisStep}</span>
              </div>
              <div className="text-[11px] leading-relaxed text-gray-500">Gateway fan-out: Detection · PRNU · Source Attribution · Custody</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
