"""
Section 2 Master Pipeline Orchestrator (§2.2, §2.5).
Assembles Provenance Check, Origin Tracing & Source Attribution, and Chain of Custody.
"""

from typing import Dict, Any, Optional
import uuid
import hashlib
from datetime import datetime, timezone

from src.common.schemas import (
    CombinedEvidenceReport,
    ProvenanceEvidence,
    OriginTracingEvidence,
    AuditLogEntry
)
from src.common.logger import logger
from src.source_attribution.pipeline import SourceAttributionPipeline
from src.fingerprinting.pipeline import FingerprintingPipeline
from src.metadata_provenance.pipeline import MetadataProvenancePipeline


class Section2Orchestrator:
    """Coordinates execution across Provenance Check, Origin Tracing, and Evidence Merging."""

    def __init__(self):
        self.source_attribution = SourceAttributionPipeline()
        self.fingerprinting = FingerprintingPipeline()
        self.metadata_provenance = MetadataProvenancePipeline()

    async def analyze(
        self,
        media_path: str,
        case_id: Optional[str] = None,
        section1_evidence: Optional[Dict[str, Any]] = None
    ) -> CombinedEvidenceReport:
        case_id = case_id or f"case_{uuid.uuid4().hex[:8]}"
        logger.info("pipeline.orchestrator.start", case_id=case_id, media_path=media_path)

        # 1. Initialize Chain of Custody
        media_hash = hashlib.sha256(media_path.encode()).hexdigest()
        custody = self.metadata_provenance.initialize_chain_of_custody(media_hash)

        # 2. Stage: Provenance Check (§2.3)
        c2pa_res = self.metadata_provenance.check_c2pa(media_path)
        exif_res = self.metadata_provenance.check_metadata_consistency(media_path)
        prnu_res = self.fingerprinting.extract_prnu(media_path)
        plat_res = self.fingerprinting.analyze_compression(media_path)

        provenance = ProvenanceEvidence(
            c2pa=c2pa_res,
            metadata=exif_res,
            prnu=prnu_res,
            platform_fingerprint=plat_res
        )

        custody.audit_log.append(
            AuditLogEntry(
                stage="provenance_complete",
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=custody.audit_log[-1].output_hash,
                output_hash=hashlib.sha256(provenance.model_dump_json().encode()).hexdigest(),
                prev_entry_hash=hashlib.sha256(custody.audit_log[-1].model_dump_json().encode()).hexdigest()
            )
        )

        # 3. Stage: Origin Tracing & Source Attribution (§2.4)
        pdq_hash = self.fingerprinting.compute_pdq_hash(media_path)
        internal_matches = self.fingerprinting.match_internal_dataset(pdq_hash)

        origin_tracing = await self.source_attribution.execute(
            media_path=media_path,
            internal_matches=internal_matches
        )

        custody.audit_log.append(
            AuditLogEntry(
                stage="origin_tracing_complete",
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=custody.audit_log[-1].output_hash,
                output_hash=hashlib.sha256(origin_tracing.model_dump_json().encode()).hexdigest(),
                prev_entry_hash=hashlib.sha256(custody.audit_log[-1].model_dump_json().encode()).hexdigest()
            )
        )

        # 4. Final Aggregation (§2.5)
        report = CombinedEvidenceReport(
            case_id=case_id,
            detection=section1_evidence or {"status": "Section 1 input placeholder", "manipulated": False},
            provenance=provenance,
            origin_tracing=origin_tracing,
            chain_of_custody=custody
        )

        custody.audit_log.append(
            AuditLogEntry(
                stage="report_generated",
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=custody.audit_log[-1].output_hash,
                output_hash=hashlib.sha256(report.model_dump_json().encode()).hexdigest(),
                prev_entry_hash=hashlib.sha256(custody.audit_log[-1].model_dump_json().encode()).hexdigest()
            )
        )

        logger.info("pipeline.orchestrator.complete", case_id=case_id)
        return report
