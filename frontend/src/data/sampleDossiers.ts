import type { InvestigationRecord } from '../types/contract';

export const SAMPLE_IMAGE_DOSSIER: InvestigationRecord = {
  schema_version: 'pramaan_x_investigation_v1',
  investigation_id: 'INV-2025-CH01-8912',
  case_id: 'FIR No. 142/2025',
  status: 'COMPLETED',
  media: {
    filename: 'farmers_tractor_rally_viral_crop.jpg',
    media_type: 'image',
    sha256: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2',
    size_bytes: 482914
  },
  modules: {
    detection: {
      status: 'COMPLETED',
      reason: null,
      message: null,
      duration_sec: 4.8,
      result: {
        schema_version: 'pramaan_x_image_analysis_v1',
        assessment: 'LIKELY_MANIPULATED',
        confidence: 'HIGH',
        visual_findings: [
          'Sub-crop perimeter truncation detected with 15.8% canvas reduction',
          'Discrete Cosine Transform (DCT) block boundary misalignment in lower quadrant',
          'Inconsistent illumination gradient relative to ambient sky source'
        ],
        supporting_signals: {
          vlm_triage_score: 0.89,
          edge_inconsistency_detected: true,
          jpeg_ghost_delta: 0.034
        },
        limitations: [
          'Color saturation may be influenced by multiple social re-compressions',
          'Analyzed on static pixel array'
        ]
      }
    },
    device_attribution: {
      status: 'COMPLETED',
      reason: null,
      message: null,
      duration_sec: 3.2,
      result: {
        attributed_device: 'OnePlus 12R (CPH2585)',
        attribution_method: 'PRNU',
        confidence: 0.942,
        pce: 48.72,
        p_value: 0.000018,
        matched_model: 'OnePlus 12R Sony IMX890 Sensor',
        metadata_exif: {
          make: 'OnePlus',
          model: 'CPH2585',
          software: 'OxygenOS 14.0',
          datetime_original: '2025-01-14T03:42:10Z',
          focal_length: '24mm (equiv)',
          exposure_time: '1/250s',
          iso: 100
        }
      }
    },
    source_attribution: {
      status: 'COMPLETED',
      reason: null,
      message: null,
      duration_sec: 14.6,
      result: {
        patient_zero: {
          url: 'https://images.firstpost.com/wp-content/uploads/2021/01/Farmers-tractor-rally-AP-640.jpg',
          domain: 'apimages.com / Associated Press Wire',
          timestamp: '2021-01-26T04:18:22Z',
          is_authoritative_wire: true,
          resolution: [1920, 1080],
          pdq_distance: 12,
          crop_type: 'target_is_crop_of_candidate',
          inlier_ratio: 0.9746,
          inlier_count: 1384,
          bounding_box: [142, 88, 1280, 720],
          scale_factor: 1.0
        },
        lineage_summary: {
          total_hops: 4,
          propagation_span: '29h 18m Delta',
          root_match_score: 0.994,
          platforms_involved: ['Telegram (@LeakOps_India)', 'Twitter / X', 'WhatsApp Forwarded Cluster', 'Court Exhibit']
        },
        homography: {
          matrix_3x3: [
            [0.9984, -0.0012, 142.30],
            [0.0011, 0.9979, -88.10],
            [0.0000, 0.0000, 1.0000]
          ],
          ransac_inlier_ratio: 0.9746,
          ransac_inlier_count: 1384,
          perspective_skew_deg: 0.14,
          coordinate_offset: { dx: 142.3, dy: -88.1 },
          area_retention_pct: 84.2
        }
      }
    }
  },
  artifacts: {
    lineage: {
      url: '/astar_lineage_tree.html',
      media_type: 'text/html',
      sha256: '7b2a9d4f1e8c3b5a6d9e0f2c4b8a1e7d3f5b9c0a2e4d6f8a1b3c5e7d9f1a3b5c'
    }
  },
  summary: {
    detection_label: 'LIKELY_MANIPULATED',
    detection_score_semantics: 'categorical_vlm_triage',
    attributed_device: 'OnePlus 12R (CPH2585)',
    device_method: 'PRNU',
    patient_zero_candidate_domain: 'apimages.com'
  },
  custody: {
    hash_algorithm: 'sha256',
    entries: [
      {
        sequence: 1,
        event: 'INGEST_BITSTREAM_VOLATILE',
        timestamp: '2025-01-15T09:36:12Z',
        input_sha256: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2',
        module_statuses: { gateway: 'COMPLETED' },
        artifact_ids: ['exhibit_a1_master.jpg'],
        previous_entry_hash: null,
        entry_hash: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2'
      },
      {
        sequence: 2,
        event: 'PERCEPTUAL_DCT_HASH',
        timestamp: '2025-01-15T09:36:15Z',
        input_sha256: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2',
        module_statuses: { detection: 'COMPLETED' },
        artifact_ids: ['triage_assessment.json'],
        previous_entry_hash: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2',
        entry_hash: '3f7a8b9c1d2e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a'
      },
      {
        sequence: 3,
        event: 'HOMOGRAPHY_RANSAC_CONVERGENCE',
        timestamp: '2025-01-15T09:36:24Z',
        input_sha256: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2',
        module_statuses: { source_attribution: 'COMPLETED' },
        artifact_ids: ['homography_matrix_H3x3.json'],
        previous_entry_hash: '3f7a8b9c1d2e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a',
        entry_hash: '8a2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c'
      },
      {
        sequence: 4,
        event: 'DAG_PATIENT_ZERO_ISOLATED',
        timestamp: '2025-01-15T09:36:28Z',
        input_sha256: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2',
        module_statuses: { source_attribution: 'COMPLETED' },
        artifact_ids: ['astar_lineage_tree.html'],
        previous_entry_hash: '8a2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c',
        entry_hash: 'b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6'
      },
      {
        sequence: 5,
        event: 'SEC_65B_CERT_COMPILATION',
        timestamp: '2025-01-15T09:36:31Z',
        input_sha256: '9e107d9d372bb6826bd81d3542a419d6a117b3a614e9f714856035b68d3ca3e2',
        module_statuses: { gateway: 'COMPLETED' },
        artifact_ids: ['sec65b_court_certificate.pdf'],
        previous_entry_hash: 'b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6',
        entry_hash: '0x8F92A1B4C7D8E2F305619A4B8C1D7E3F5A9B2C4D6E8F0A1B3C5D7E9F1A2B4C6D'
      }
    ]
  },
  warnings: [],
  execution_time_sec: 22.6
};

export const SAMPLE_VIDEO_DOSSIER: InvestigationRecord = {
  schema_version: 'pramaan_x_investigation_v1',
  investigation_id: 'INV-2025-CH02-4410',
  case_id: 'FIR No. 089/2025',
  status: 'COMPLETED',
  media: {
    filename: 'minister_statement_cloned_voice.mp4',
    media_type: 'video',
    sha256: 'e8d1a45b7f9c2d3e8a1b4c6e9f0a2b5c7d8e1f3a6b9c0d2e4f7a8b1c3d5e7f9a',
    size_bytes: 8492040
  },
  modules: {
    detection: {
      status: 'COMPLETED',
      reason: null,
      message: null,
      duration_sec: 16.4,
      result: {
        selected_class: 'AUDIO_MANIPULATION',
        classification: 'Synthetic Audio Replacement (Deepfake Speech)',
        class_logits: {
          REAL: -1.82,
          VISUAL_MANIPULATION: -0.42,
          AUDIO_MANIPULATION: 2.94,
          AUDIO_VISUAL_MANIPULATION: 0.18
        },
        softmax_probabilities: {
          REAL: 0.007,
          VISUAL_MANIPULATION: 0.031,
          AUDIO_MANIPULATION: 0.902,
          AUDIO_VISUAL_MANIPULATION: 0.060
        },
        branch_evidence: {
          visual: { available: true, confidence: 0.88, details: 'Facial pixel landmarks unaltered' },
          audio: { available: true, confidence: 0.96, details: 'Neural vocoder spectral phase artifacts in 2-4kHz band' },
          av_sync: { available: true, sync_score: 0.32, details: 'Speech phoneme to bilabial mouth closure divergence > 140ms' }
        },
        temporal_evidence: [
          { timestamp_sec: 2.4, frame_idx: 72, anomaly_score: 0.88, label: 'Voice timbre divergence onset' },
          { timestamp_sec: 4.1, frame_idx: 123, anomaly_score: 0.94, label: 'Labiomental desync peak' }
        ]
      }
    },
    device_attribution: {
      status: 'NOT_APPLICABLE',
      reason: 'device attribution currently supports still images only',
      message: null,
      duration_sec: 0.0,
      result: null
    },
    source_attribution: {
      status: 'COMPLETED',
      reason: null,
      message: null,
      duration_sec: 21.3,
      result: {
        patient_zero: {
          url: 'https://cdn.sansadtv.nic.in/archive/parliament_speech_session262.mp4',
          domain: 'sansadtv.nic.in / Official State Broadcast',
          timestamp: '2024-12-10T06:30:00Z',
          is_authoritative_wire: true,
          resolution: [1920, 1080],
          crop_type: 'exact',
          inlier_ratio: 0.991,
          inlier_count: 2410
        },
        lineage_summary: {
          total_hops: 3,
          propagation_span: '14h 05m Delta',
          root_match_score: 0.988,
          platforms_involved: ['Sansad TV Official', 'Dubbed Re-upload YouTube', 'Court Seized WhatsApp Video']
        }
      }
    }
  },
  artifacts: {
    lineage: {
      url: '/astar_lineage_tree.html',
      media_type: 'text/html',
      sha256: '9f8e7d6c5b4a3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b'
    }
  },
  summary: {
    detection_label: 'AUDIO_MANIPULATION',
    detection_score_semantics: 'uncalibrated_softmax',
    attributed_device: null,
    device_method: null,
    patient_zero_candidate_domain: 'sansadtv.nic.in'
  },
  custody: {
    hash_algorithm: 'sha256',
    entries: [
      {
        sequence: 1,
        event: 'INGEST_BITSTREAM_VOLATILE',
        timestamp: '2025-01-12T14:10:00Z',
        input_sha256: 'e8d1a45b7f9c2d3e8a1b4c6e9f0a2b5c7d8e1f3a6b9c0d2e4f7a8b1c3d5e7f9a',
        module_statuses: { gateway: 'COMPLETED' },
        artifact_ids: ['exhibit_v1_video.mp4'],
        previous_entry_hash: null,
        entry_hash: 'e8d1a45b7f9c2d3e8a1b4c6e9f0a2b5c7d8e1f3a6b9c0d2e4f7a8b1c3d5e7f9a'
      }
    ]
  },
  warnings: [],
  execution_time_sec: 37.7
};
