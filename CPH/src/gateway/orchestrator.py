"""Concurrent orchestration for one immutable evidence object."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import uuid

import httpx

from src.common.logger import logger

from .artifacts import ArtifactRegistry
from .config import GatewayConfig
from .media import StoredMedia
from .models import (
    CapabilityStatus,
    CapabilitiesResponse,
    CustodyEntry,
    CustodyRecord,
    InvestigationModules,
    InvestigationResponse,
    InvestigationStatus,
    InvestigationSummary,
    MediaType,
    ModuleResult,
    ModuleStatus,
)


@dataclass
class ExecutionOutput:
    result: dict[str, Any]
    artifacts: dict[str, Path] = field(default_factory=dict)


ModuleRunner = Callable[[StoredMedia, Path], Awaitable[ExecutionOutput]]


class ModuleExecutionError(RuntimeError):
    def __init__(self, reason: str, public_message: str) -> None:
        super().__init__(public_message)
        self.reason = reason
        self.public_message = public_message


def new_investigation_id() -> str:
    return f"INV-{uuid.uuid4().hex[:16].upper()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _custody_entry(
    sequence: int,
    event: str,
    input_sha256: str,
    previous_hash: str | None,
    module_statuses: dict[str, ModuleStatus] | None = None,
    artifact_ids: list[str] | None = None,
) -> CustodyEntry:
    timestamp = datetime.now(timezone.utc).isoformat()
    body = {
        "sequence": sequence,
        "event": event,
        "timestamp": timestamp,
        "input_sha256": input_sha256,
        "module_statuses": {
            key: value.value for key, value in (module_statuses or {}).items()
        },
        "artifact_ids": artifact_ids or [],
        "previous_entry_hash": previous_hash,
    }
    entry_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CustodyEntry(**body, entry_hash=entry_hash)


def _public_payload(
    value: Any,
    evidence_sha256: str,
    artifact_paths: dict[str, tuple[Path, str]],
    key: str = "",
) -> Any:
    """Redact local paths while retaining the native response structure."""
    if isinstance(value, dict):
        return {
            str(child_key): _public_payload(
                child_value,
                evidence_sha256,
                artifact_paths,
                str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_public_payload(item, evidence_sha256, artifact_paths, key) for item in value]
    if not isinstance(value, str):
        return value

    for path, url in artifact_paths.values():
        if value in {str(path), str(path.resolve()), f"file://{path.resolve()}"}:
            return url
    lower_key = key.lower()
    if value.startswith("file://"):
        return "REDACTED_INTERNAL_PATH"
    if any(token in lower_key for token in ("path", "directory", "folder")):
        candidate = Path(value)
        if candidate.is_absolute() or value.startswith(("data/", "runs/", "outputs/")):
            if "image" in lower_key or lower_key in {"path", "media_path", "file_path"}:
                return f"evidence://sha256/{evidence_sha256}"
            return "REDACTED_INTERNAL_PATH"
    return value


async def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
    if response.status_code != 200:
        raise ModuleExecutionError(
            "UPSTREAM_HTTP_ERROR",
            f"Forensic service returned HTTP {response.status_code}.",
        )
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise ModuleExecutionError(
            "MALFORMED_UPSTREAM_JSON",
            "Forensic service returned invalid JSON.",
        ) from exc
    if not isinstance(data, dict):
        raise ModuleExecutionError(
            "MALFORMED_UPSTREAM_JSON",
            "Forensic service result must be a JSON object.",
        )
    return data


def build_default_detection_runner(config: GatewayConfig) -> ModuleRunner:
    async def run(media: StoredMedia, _: Path) -> ExecutionOutput:
        result = await _post_json(
            f"{config.detection_service_url}/api/v1/detect",
            {"media_path": str(media.local_path)},
        )
        return ExecutionOutput(result=result)

    return run


def build_default_prnu_runner(config: GatewayConfig) -> ModuleRunner:
    async def run(media: StoredMedia, _: Path) -> ExecutionOutput:
        result = await _post_json(
            f"{config.prnu_service_url}/api/analyse",
            {"media_path": str(media.local_path)},
        )
        if result.get("ok") is not True or not isinstance(result.get("result"), dict):
            raise ModuleExecutionError(
                "PRNU_ANALYSIS_FAILED",
                "Device-attribution service did not produce a usable result.",
            )
        return ExecutionOutput(result=result)

    return run


def build_default_source_runner() -> ModuleRunner:
    async def run(media: StoredMedia, artifact_dir: Path) -> ExecutionOutput:
        if media.public.media_type is MediaType.IMAGE:
            from src.research.astar_attribution.astar_engine import AStarVisualAttributionEngine

            graph = artifact_dir / "lineage.html"
            engine = AStarVisualAttributionEngine(output_graph_path=str(graph))
            result = await engine.trace_origin_async(str(media.local_path))
            artifacts = {"lineage": graph} if graph.is_file() else {}
            return ExecutionOutput(result=result, artifacts=artifacts)

        from src.source_attribution.pipeline import SourceAttributionPipeline

        pipeline = SourceAttributionPipeline(output_dir=str(artifact_dir))
        result = await pipeline.execute(media_path=str(media.local_path))
        artifacts = {
            key: path
            for key, path in {
                "social_graph": artifact_dir / "social_graph.html",
                "dissemination_graph": artifact_dir / "dissemination_graph.html",
            }.items()
            if path.is_file()
        }
        return ExecutionOutput(result=result.model_dump(mode="json"), artifacts=artifacts)

    return run


class UnifiedInvestigationOrchestrator:
    def __init__(
        self,
        config: GatewayConfig,
        artifact_registry: ArtifactRegistry | None = None,
        detection_runner: ModuleRunner | None = None,
        prnu_runner: ModuleRunner | None = None,
        source_runner: ModuleRunner | None = None,
    ) -> None:
        self.config = config
        self.artifacts = artifact_registry or ArtifactRegistry(config.investigation_output_dir)
        self.detection_runner = detection_runner or build_default_detection_runner(config)
        self.prnu_runner = prnu_runner or build_default_prnu_runner(config)
        self.source_runner = source_runner or build_default_source_runner()

    async def _execute(
        self,
        name: str,
        runner: ModuleRunner,
        media: StoredMedia,
        artifact_dir: Path,
        timeout: float,
    ) -> tuple[ModuleResult, dict[str, Path]]:
        started = perf_counter()
        try:
            output = await asyncio.wait_for(runner(media, artifact_dir), timeout=timeout)
            if not isinstance(output, ExecutionOutput) or not isinstance(output.result, dict):
                raise ModuleExecutionError(
                    "MALFORMED_MODULE_RESULT",
                    "Forensic module did not return a JSON object.",
                )
            return (
                ModuleResult(
                    status=ModuleStatus.COMPLETED,
                    duration_sec=perf_counter() - started,
                    result=output.result,
                ),
                output.artifacts,
            )
        except TimeoutError:
            logger.warning("gateway.module_timeout", module=name, timeout_sec=timeout)
            return (
                ModuleResult(
                    status=ModuleStatus.TIMEOUT,
                    reason=f"{name.upper()}_TIMEOUT",
                    message="Forensic module exceeded its configured timeout.",
                    duration_sec=perf_counter() - started,
                ),
                {},
            )
        except ModuleExecutionError as exc:
            logger.warning("gateway.module_failed", module=name, reason=exc.reason)
            return (
                ModuleResult(
                    status=ModuleStatus.FAILED,
                    reason=exc.reason,
                    message=exc.public_message,
                    duration_sec=perf_counter() - started,
                ),
                {},
            )
        except Exception as exc:
            logger.exception("gateway.module_failed", module=name, error_type=type(exc).__name__)
            return (
                ModuleResult(
                    status=ModuleStatus.FAILED,
                    reason=f"{name.upper()}_EXECUTION_FAILED",
                    message="Forensic module failed. See server logs for details.",
                    duration_sec=perf_counter() - started,
                ),
                {},
            )

    async def investigate(self, media: StoredMedia, case_id: str | None = None) -> InvestigationResponse:
        started = perf_counter()
        effective_case_id = case_id.strip() if case_id and case_id.strip() else f"CASE-{media.investigation_id[4:]}"
        artifact_dir = self.artifacts.investigation_dir(media.investigation_id)
        ingested = _custody_entry(1, "EVIDENCE_INGESTED", media.public.sha256, None)

        detection_task = self._execute(
            "detection",
            self.detection_runner,
            media,
            artifact_dir,
            self.config.detection_timeout_sec,
        )
        source_task = self._execute(
            "source_attribution",
            self.source_runner,
            media,
            artifact_dir,
            self.config.source_attribution_timeout_sec,
        )
        if media.public.media_type is MediaType.IMAGE:
            prnu_task = self._execute(
                "device_attribution",
                self.prnu_runner,
                media,
                artifact_dir,
                self.config.prnu_timeout_sec,
            )
        else:
            async def not_applicable() -> tuple[ModuleResult, dict[str, Path]]:
                return (
                    ModuleResult(
                        status=ModuleStatus.NOT_APPLICABLE,
                        reason="device attribution currently supports still images only",
                        duration_sec=0.0,
                    ),
                    {},
                )

            prnu_task = not_applicable()

        detection_pair, prnu_pair, source_pair = await asyncio.gather(
            detection_task,
            prnu_task,
            source_task,
        )
        detection, detection_artifacts = detection_pair
        device, device_artifacts = prnu_pair
        source, source_artifacts = source_pair

        registered = {}
        path_urls: dict[str, tuple[Path, str]] = {}
        for key, path in {
            **detection_artifacts,
            **device_artifacts,
            **source_artifacts,
        }.items():
            try:
                reference = self.artifacts.register(media.investigation_id, key, path)
                registered[key] = reference
                path_urls[key] = (path, reference.url)
            except ValueError:
                logger.warning("gateway.artifact_rejected", artifact=key)

        for module in (detection, device, source):
            if module.result is not None:
                module.result = _public_payload(
                    module.result,
                    media.public.sha256,
                    path_urls,
                )

        statuses = {
            "detection": detection.status,
            "device_attribution": device.status,
            "source_attribution": source.status,
        }
        completed = sum(value is ModuleStatus.COMPLETED for value in statuses.values())
        failed = sum(value in {ModuleStatus.FAILED, ModuleStatus.TIMEOUT} for value in statuses.values())
        if completed == 0:
            overall = InvestigationStatus.FAILED
        elif failed:
            overall = InvestigationStatus.PARTIAL
        else:
            overall = InvestigationStatus.COMPLETED

        warnings = []
        final_hash = _sha256_file(media.local_path)
        if final_hash != media.public.sha256:
            overall = InvestigationStatus.FAILED
            warnings.append("Evidence integrity check failed after module execution.")

        executed = _custody_entry(
            2,
            "MODULE_EXECUTION_RECORDED",
            media.public.sha256,
            ingested.entry_hash,
            statuses,
            sorted(registered),
        )
        return InvestigationResponse(
            investigation_id=media.investigation_id,
            case_id=effective_case_id,
            status=overall,
            media=media.public,
            modules=InvestigationModules(
                detection=detection,
                device_attribution=device,
                source_attribution=source,
            ),
            artifacts=registered,
            summary=_build_summary(detection, device, source),
            custody=CustodyRecord(entries=[ingested, executed]),
            warnings=warnings,
            execution_time_sec=perf_counter() - started,
        )

    async def capabilities(self) -> CapabilitiesResponse:
        async def health(url: str) -> dict[str, Any] | None:
            try:
                return await asyncio.wait_for(_get_json(url), timeout=self.config.health_timeout_sec)
            except Exception:
                return None

        detection_health, prnu_health = await asyncio.gather(
            health(f"{self.config.detection_service_url}/health"),
            health(f"{self.config.prnu_service_url}/health"),
        )
        detection_caps = (detection_health or {}).get("capabilities", {})
        prnu_caps = (prnu_health or {}).get("capabilities", {})
        image_source_available = all(
            importlib.util.find_spec(name) is not None for name in ("PIL", "cv2", "numpy")
        )
        video_source_available = image_source_available and importlib.util.find_spec("cv2") is not None
        return CapabilitiesResponse(
            image_detection=CapabilityStatus(
                available=bool(detection_caps.get("image", {}).get("available")),
                healthy=detection_health is not None,
                detail=None if detection_health else "Detection service is unreachable",
            ),
            video_detection=CapabilityStatus(
                available=bool(detection_caps.get("video", {}).get("available")),
                healthy=detection_health is not None,
                detail=None if detection_health else "Detection service is unreachable",
            ),
            prnu_image_attribution=CapabilityStatus(
                available=bool(prnu_caps.get("image_attribution", {}).get("available")),
                healthy=prnu_health is not None,
                detail=None if prnu_health else "PRNU service is unreachable",
            ),
            source_attribution_image=CapabilityStatus(
                available=image_source_available,
                healthy=image_source_available,
                detail=None if image_source_available else "Required local image-attribution runtime is unavailable",
            ),
            source_attribution_video=CapabilityStatus(
                available=video_source_available,
                healthy=video_source_available,
                detail=None if video_source_available else "Required video keyframe runtime is unavailable",
            ),
        )


async def _get_json(url: str) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Health response must be an object")
    return data


def _build_summary(
    detection: ModuleResult,
    device: ModuleResult,
    source: ModuleResult,
) -> InvestigationSummary | None:
    detection_payload = detection.result or {}
    label = detection_payload.get("selected_class")
    score_semantics = "uncalibrated_softmax" if label else None
    if not label:
        assessment = detection_payload.get("assessment")
        if isinstance(assessment, dict):
            label = assessment.get("label")
            score_semantics = "categorical_vlm_triage"

    device_payload = device.result or {}
    native_device = device_payload.get("result", device_payload)
    attribution = native_device.get("device_attribution", {}) if isinstance(native_device, dict) else {}
    legacy_device = native_device.get("device", {}) if isinstance(native_device, dict) else {}
    attributed_device = attribution.get("display_name") or attribution.get("prediction") or legacy_device.get("label")
    device_method = attribution.get("primary_method") or legacy_device.get("method")

    source_payload = source.result or {}
    patient = source_payload.get("patient_zero") or source_payload.get("earliest_candidate") or {}
    domain = patient.get("domain") if isinstance(patient, dict) else None
    if not any((label, attributed_device, domain)):
        return None
    return InvestigationSummary(
        detection_label=label,
        detection_score_semantics=score_semantics,
        attributed_device=attributed_device,
        device_method=device_method,
        patient_zero_candidate_domain=domain,
    )
