import type { GatewayCapabilities, InvestigationRecord } from '../types/contract';

// In Vite development, use the same-origin proxy so localhost/127.0.0.1 and
// gateway CORS settings cannot disagree. Production deployments should provide
// VITE_GATEWAY_URL when the gateway is hosted on a separate origin.
const GATEWAY_BASE_URL = (import.meta as any).env?.VITE_GATEWAY_URL || ((import.meta as any).env?.DEV ? '' : 'http://127.0.0.1:8000');

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
 * Probes the CPH Gateway's aggregated child-service capabilities.
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
    const payload = await res.json() as Record<string, unknown>;
    const readCapability = (key: string) => {
      const value = payload[key];
      return value && typeof value === 'object' ? value as Record<string, unknown> : {};
    };
    const imageDetection = readCapability('image_detection');
    const videoDetection = readCapability('video_detection');
    const prnu = readCapability('prnu_image_attribution');
    const sourceImage = readCapability('source_attribution_image');
    const sourceVideo = readCapability('source_attribution_video');
    const gatewayHealthy = [imageDetection, videoDetection, prnu, sourceImage, sourceVideo]
      .every((capability) => capability.healthy === true);

    return {
      gateway: { status: gatewayHealthy ? 'ONLINE' : 'DEGRADED', port: 8000, service_health_timeout_sec: 3 },
      detection: {
        status: imageDetection.healthy === true || videoDetection.healthy === true ? 'ONLINE' : 'OFFLINE',
        url: 'gateway-managed',
        image_ready: imageDetection.available === true,
        video_ready: videoDetection.available === true,
        error: typeof imageDetection.detail === 'string' ? imageDetection.detail : null,
      },
      prnu: {
        status: prnu.healthy === true ? 'ONLINE' : 'OFFLINE',
        url: 'gateway-managed',
        error: typeof prnu.detail === 'string' ? prnu.detail : null,
      },
      source_attribution: {
        status: sourceImage.healthy === true || sourceVideo.healthy === true ? 'READY' : 'DEGRADED',
        astar_ready: sourceImage.available === true || sourceVideo.available === true,
        api_keys_configured: undefined,
      },
    };
  } catch (err) {
    console.warn('[Pramaan-X] Gateway offline or unreachable:', err);
    return {
      gateway: { status: 'OFFLINE', port: 8000, service_health_timeout_sec: 3 },
      detection: { status: 'OFFLINE', url: 'gateway-managed', error: 'Connection refused' },
      prnu: { status: 'OFFLINE', url: 'gateway-managed', error: 'Connection refused' },
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

  let res: Response;
  try {
    res = await fetch(`${GATEWAY_BASE_URL}/api/v1/investigate/upload`, {
      method: 'POST',
      body: formData,
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'Network request was blocked or refused.';
    throw new Error(`Cannot reach the CPH Gateway at port 8000. Start the gateway and try again. (${detail})`);
  }

  if (!res.ok) {
    let errorDetail = `Upload failed with HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        const detail = typeof errJson.detail === 'string' ? errJson.detail : errJson.detail.message || errJson.detail.code;
        if (detail) errorDetail = `${errorDetail}: ${detail}`;
      }
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
