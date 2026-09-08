import type { GatewayCapabilities, InvestigationRecord } from '../types/contract';

const GATEWAY_BASE_URL = (import.meta as any).env?.VITE_GATEWAY_URL || 'http://localhost:8000';

/**
 * Computes cryptographically verified SHA-256 digest directly from client volatile buffer
 * Conforms to ISO/IEC 27037:2012 client-side ingestion verification
 */
export async function computeFileSha256(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Probes the CPH Gateway live capabilities across Node 1 (8001), Node 2 (8002), and Node 3 (Internal)
 */
export async function fetchCapabilities(): Promise<GatewayCapabilities> {
  try {
    const res = await fetch(`${GATEWAY_BASE_URL}/api/v1/capabilities`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!res.ok) {
      throw new Error(`Gateway returned HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.warn('[AMOT-FAS] Gateway offline or unreachable:', err);
    return {
      gateway: { status: 'OFFLINE', port: 8000, service_health_timeout_sec: 3 },
      detection: { status: 'OFFLINE', url: 'http://localhost:8001', error: 'Connection refused' },
      prnu: { status: 'OFFLINE', url: 'http://localhost:8002', error: 'Connection refused' },
      source_attribution: { status: 'STANDALONE_LOCAL', astar_ready: true, api_keys_configured: false }
    };
  }
}

/**
 * Submits evidence file to POST /api/v1/investigate/upload via multipart/form-data
 */
export async function uploadEvidence(file: File, caseId?: string): Promise<InvestigationRecord> {
  const formData = new FormData();
  formData.append('file', file);
  if (caseId && caseId.trim()) {
    formData.append('case_id', caseId.trim());
  }

  const res = await fetch(`${GATEWAY_BASE_URL}/api/v1/investigate/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errorDetail = `Upload failed with HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {}
    throw new Error(errorDetail);
  }

  return await res.json();
}

/**
 * Constructs official URL for backend investigation artifacts
 */
export function getArtifactUrl(investigationId: string, artifactKey: string): string {
  return `${GATEWAY_BASE_URL}/api/v1/investigations/${investigationId}/artifacts/${artifactKey}`;
}
