"""LLM-assisted still-image authenticity assessment for Pramaan-X.

This module is deliberately separate from the trained video detector. It provides
conservative visual triage through an OpenAI-compatible multimodal chat endpoint.
It never presents an LLM score as a calibrated forensic probability.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_KIMI_MODEL = "kimi-k3"
SCHEMA_VERSION = "pramaan_x_image_analysis_v1"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LABELS = {"LIKELY_AUTHENTIC", "SUSPICIOUS", "LIKELY_MANIPULATED"}
CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}
SEVERITIES = {"LOW", "MEDIUM", "HIGH", "INCONCLUSIVE"}
SIGNAL_KEYS = (
    "facial_anatomy",
    "hands_body",
    "lighting",
    "reflections",
    "textures",
    "background_geometry",
    "text_symbols",
    "compositing_edges",
)
FINDING_CATEGORIES = {
    "facial_anatomy",
    "hands_body",
    "lighting",
    "reflections",
    "textures",
    "background_geometry",
    "text_symbols",
    "compositing_edges",
    "semantic_consistency",
    "general_uncertainty",
}
SYSTEM_PROMPT = """You assist with visual authenticity triage in a digital-forensics workflow.
Inspect the supplied image for visible evidence of AI generation, synthetic alteration,
face manipulation, compositing, or editing. Consider facial anatomy, hands and body
geometry, lighting and shadows, reflections, textures, background structure and
perspective, repeated patterns, text and symbols, edge/compositing artifacts, and
physically or semantically inconsistent details.

Do not assume an image is fake because it is unusual, compressed, stylized, low
resolution, or professionally edited. Do not assume it is authentic because no
obvious artifact is visible. Describe only anomalies you can actually observe.
If evidence is weak or ambiguous, prefer SUSPICIOUS or LOW confidence.

Return only a JSON object with this shape:
{
  "assessment": {
    "label": "LIKELY_AUTHENTIC | SUSPICIOUS | LIKELY_MANIPULATED",
    "confidence_level": "LOW | MEDIUM | HIGH",
    "summary": "conservative observed-evidence summary"
  },
  "visual_findings": [
    {
      "category": "facial_anatomy | hands_body | lighting | reflections | textures | background_geometry | text_symbols | compositing_edges | semantic_consistency | general_uncertainty",
      "severity": "LOW | MEDIUM | HIGH | INCONCLUSIVE",
      "region": "image region or whole_image",
      "finding": "only an observed finding"
    }
  ],
  "supporting_signals": {
    "facial_anatomy": "short assessment",
    "hands_body": "short assessment",
    "lighting": "short assessment",
    "reflections": "short assessment",
    "textures": "short assessment",
    "background_geometry": "short assessment",
    "text_symbols": "short assessment",
    "compositing_edges": "short assessment"
  }
}"""


CompletionFn = Callable[[dict[str, Any], str, str, float], dict[str, Any]]



def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Unable to read environment file: {path}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise RuntimeError(f"Invalid .env entry at {path}:{line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key[0].isdigit() or not key.replace("_", "").isalnum():
            raise RuntimeError(f"Invalid environment variable name at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_mime(image_format: str | None, path: Path) -> str:
    if image_format:
        return Image.MIME.get(image_format.upper(), "application/octet-stream")
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def inspect_image(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            exif = image.getexif()
            metadata = {
                "filename": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "exif": {
                    "present": bool(exif),
                    "software": str(exif.get(305)) if exif.get(305) is not None else None,
                    "camera_make": str(exif.get(271)) if exif.get(271) is not None else None,
                    "camera_model": str(exif.get(272)) if exif.get(272) is not None else None,
                },
            }
            if not metadata["format"] or image.width < 1 or image.height < 1:
                raise ValueError("Image has invalid format or dimensions")
            return metadata
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to decode image {path}: {exc}") from exc


def _request_payload(image_path: Path, model: str, image_format: str | None, retry: bool) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime = _image_mime(image_format, image_path)
    retry_instruction = (
        "This is a bounded retry. Return only one valid JSON object. Do not add markdown fences."
        if retry
        else "Return only one valid JSON object. Do not add markdown fences."
    )
    return {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": retry_instruction},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            },
        ],
    }


def _default_completion(
    payload: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout: float,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib_request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read(512).decode("utf-8", errors="replace")
        raise RuntimeError(f"Image model request failed with HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Image model request failed: {exc.reason}") from exc
    try:
        response_payload = json.loads(raw)
        content = response_payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Image model returned an invalid API response shape") from exc
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Image model returned empty content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Image model content was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Image model content must be a JSON object")
    return parsed


def validate_model_result(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Image assessment must be a JSON object")
    assessment = value.get("assessment")
    if not isinstance(assessment, dict):
        raise ValueError("Image assessment is missing the assessment object")
    label, confidence, summary = (
        assessment.get("label"),
        assessment.get("confidence_level"),
        assessment.get("summary"),
    )
    if label not in LABELS:
        raise ValueError(f"Invalid image assessment label: {label!r}")
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"Invalid image confidence level: {confidence!r}")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Image assessment summary must be a non-empty string")

    findings = value.get("visual_findings")
    if not isinstance(findings, list):
        raise ValueError("visual_findings must be a list")
    clean_findings: list[dict[str, str]] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"Finding {index} must be an object")
        category, severity, region, text = (
            finding.get("category"),
            finding.get("severity"),
            finding.get("region"),
            finding.get("finding"),
        )
        if category not in FINDING_CATEGORIES:
            raise ValueError(f"Invalid finding category at index {index}: {category!r}")
        if severity not in SEVERITIES:
            raise ValueError(f"Invalid finding severity at index {index}: {severity!r}")
        if not all(isinstance(item, str) and item.strip() for item in (region, text)):
            raise ValueError(f"Finding {index} requires non-empty region and finding text")
        clean_findings.append({
            "category": category,
            "severity": severity,
            "region": region,
            "finding": text,
        })

    signals = value.get("supporting_signals")
    if not isinstance(signals, dict) or any(
        key not in signals or not isinstance(signals[key], str) or not signals[key].strip()
        for key in SIGNAL_KEYS
    ):
        raise ValueError("supporting_signals must contain a non-empty string for every signal category")
    return {
        "assessment": {
            "label": label,
            "confidence_level": confidence,
            "summary": summary,
        },
        "visual_findings": clean_findings,
        "supporting_signals": {key: signals[key] for key in SIGNAL_KEYS},
    }


def analyse_image_with_vlm(
    image_path: Path,
    model: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_KIMI_BASE_URL,
    timeout: float = 90.0,
    request_fn: CompletionFn | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("Image analysis API credential is missing")
    metadata = inspect_image(image_path)
    completion = request_fn or _default_completion
    parsed: dict[str, Any] | None = None
    last_error: Exception | None = None
    for retry in (False, True):
        try:
            payload = _request_payload(image_path, model, metadata["format"], retry)
            parsed = completion(payload, api_key, base_url, timeout)
            return validate_model_result(parsed)
        except (ValueError, RuntimeError) as exc:
            last_error = exc
            if retry:
                break
    raise RuntimeError(f"Image analysis response validation failed after one retry: {last_error}") from last_error


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_result(image_path: Path, model_result: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "media_type": "image",
        "input": inspect_image(image_path),
        "assessment": model_result["assessment"],
        "visual_findings": model_result["visual_findings"],
        "supporting_signals": model_result["supporting_signals"],
        "analysis": {
            "method": "llm_assisted_visual_authenticity_assessment",
            "model": model,
            "probability_note": "This is categorical model triage, not a calibrated forensic probability.",
        },
        "limitations": [
            "Assessment is produced by a multimodal language model and is not a calibrated forensic probability.",
            "A sophisticated manipulation may contain no obvious visible artifact.",
            "A genuine unusual, stylized, compressed, or low-resolution image may appear suspicious.",
            "Missing EXIF or metadata does not imply manipulation.",
            "The result should be reviewed with independent forensic and provenance evidence.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    load_env_file()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("image", type=Path, help="Path to a JPG, JPEG, PNG, or WEBP image")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", default=os.environ.get("PRAMAAN_X_IMAGE_MODEL", DEFAULT_KIMI_MODEL))
    result.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_KIMI_BASE_URL))
    result.add_argument("--api-key-env", default="KIMI_API_KEY")
    result.add_argument("--timeout", type=float, default=90.0)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing image-analysis credential in environment variable {args.api_key_env}")
    model_result = analyse_image_with_vlm(
        args.image.expanduser().resolve(),
        args.model,
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    result = build_result(args.image.expanduser().resolve(), model_result, args.model)
    atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
