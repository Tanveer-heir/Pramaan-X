// pramaan_x_investigation_v1 Type Definitions
// Corresponds strictly to docs/API_CONTRACT.md

export type ModuleStatus = 'COMPLETED' | 'NOT_APPLICABLE' | 'FAILED' | 'TIMEOUT';

export type InvestigationStatus = 'COMPLETED' | 'PARTIAL' | 'FAILED';

export interface MediaMetadata {
  filename: string;
  media_type: 'image' | 'video';
  sha256: string;
  size_bytes: number;
}

export interface ModuleEnvelope<T = Record<string, unknown>> {
  status: ModuleStatus;
  reason?: string | null;
  message?: string | null;
  duration_sec?: number;
  result: T | null;
}

export interface ImageAssessment {
  label?: string;
  confidence_level?: string;
  summary?: string;
}

export interface ImageDetectionResult {
  schema_version?: string;
  assessment?: ImageAssessment | 'LIKELY_AUTHENTIC' | 'SUSPICIOUS' | 'LIKELY_MANIPULATED' | string;
  confidence?: 'LOW' | 'MEDIUM' | 'HIGH' | string;
  confidence_level?: 'LOW' | 'MEDIUM' | 'HIGH' | string;
  visual_findings?: string[];
  supporting_signals?: Record<string, unknown>;
  limitations?: string[];
  input?: Record<string, unknown>;
}

export interface VideoDetectionResult {
  selected_class?: 'REAL' | 'VISUAL_MANIPULATION' | 'AUDIO_MANIPULATION' | 'AUDIO_VISUAL_MANIPULATION' | string;
  classification?: string;
  class_logits?: Record<string, number>;
  softmax_probabilities?: {
    REAL?: number;
    VISUAL_MANIPULATION?: number;
    AUDIO_MANIPULATION?: number;
    AUDIO_VISUAL_MANIPULATION?: number;
    [key: string]: number | undefined;
  };
  branch_evidence?: {
    visual?: { available: boolean; confidence?: number; details?: string };
    audio?: { available: boolean; confidence?: number; details?: string };
    av_sync?: { available: boolean; sync_score?: number; details?: string };
  };
  counterfactual_modality_analysis?: Record<string, unknown>;
  temporal_evidence?: Array<{
    timestamp_sec: number;
    frame_idx: number;
    anomaly_score: number;
    label: string;
  }>;
}

export interface DeviceAttributionResult {
  attributed_device?: string | null;
  attribution_method?: 'PRNU' | 'METADATA' | 'INCONCLUSIVE' | string;
  confidence?: number;
  pce?: number;
  p_value?: number;
  matched_model?: string | null;
  metadata_exif?: {
    make?: string;
    model?: string;
    software?: string;
    datetime_original?: string;
    focal_length?: string;
    exposure_time?: string;
    iso?: string | number;
    [key: string]: unknown;
  };
}

export interface SourceAttributionResult {
  patient_zero?: {
    url?: string;
    media_url?: string;
    source_page_url?: string;
    probability?: number;
    confidence?: number;
    evidence?: string;
    details?: string;
    match_type?: string;
    domain?: string;
    timestamp?: string;
    is_authoritative_wire?: boolean;
    resolution?: [number, number];
    pdq_distance?: number;
    crop_type?: 'exact' | 'target_is_crop_of_candidate' | 'candidate_is_crop_of_target' | 'none';
    inlier_ratio?: number;
    inlier_count?: number;
    bounding_box?: [number, number, number, number]; // [x, y, w, h]
    scale_factor?: number;
    [key: string]: unknown;
  };
  lineage_summary?: {
    total_hops?: number;
    propagation_span?: string;
    root_match_score?: number;
    platforms_involved?: string[];
  };
  homography?: {
    matrix_3x3?: number[][];
    ransac_inlier_ratio?: number;
    ransac_inlier_count?: number;
    perspective_skew_deg?: number;
    coordinate_offset?: { dx: number; dy: number };
    area_retention_pct?: number;
  };
  candidate_sources?: Array<Record<string, unknown>>;
  lineage_path?: Array<Record<string, unknown>>;
  target?: Record<string, unknown>;
  target_media?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ArtifactRef {
  url: string;
  media_type: string;
  sha256: string;
}

export interface InvestigationSummary {
  detection_label: string | null;
  detection_score_semantics: string;
  attributed_device: string | null;
  device_method: string | null;
  patient_zero_candidate_domain: string | null;
}

export interface CustodyEntry {
  sequence: number;
  event: string;
  timestamp: string;
  input_sha256: string;
  module_statuses: Record<string, ModuleStatus>;
  artifact_ids: string[];
  previous_entry_hash: string | null;
  entry_hash: string;
}

export interface InvestigationRecord {
  schema_version: 'pramaan_x_investigation_v1';
  investigation_id: string;
  case_id: string;
  status: InvestigationStatus;
  media: MediaMetadata;
  modules: {
    detection: ModuleEnvelope<ImageDetectionResult | VideoDetectionResult>;
    device_attribution: ModuleEnvelope<DeviceAttributionResult>;
    source_attribution: ModuleEnvelope<SourceAttributionResult>;
  };
  artifacts?: Record<string, ArtifactRef>;
  summary?: InvestigationSummary | null;
  custody?: {
    hash_algorithm: 'sha256';
    entries: CustodyEntry[];
  };
  warnings?: string[];
  execution_time_sec?: number;
}

export interface GatewayCapabilities {
  gateway?: {
    status: string;
    port: number;
    service_health_timeout_sec: number;
  };
  detection?: {
    status: 'ONLINE' | 'OFFLINE' | string;
    url: string;
    device?: string;
    image_ready?: boolean;
    video_ready?: boolean;
    error?: string | null;
  };
  prnu?: {
    status: 'ONLINE' | 'OFFLINE' | string;
    url: string;
    reference_count?: number;
    synthetic_allowed?: boolean;
    error?: string | null;
  };
  source_attribution?: {
    status: 'READY' | 'DEGRADED' | string;
    astar_ready?: boolean;
    api_keys_configured?: boolean;
  };
}

export interface CaseSession {
  firNumber: string;
  policeStation: string;
  officerInCharge: string;
  officerId: string;
  judicialCourt: string;
}
