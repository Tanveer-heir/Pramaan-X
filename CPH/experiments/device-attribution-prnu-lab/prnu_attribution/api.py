from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
from email import policy
from email.parser import BytesParser
import json
import mimetypes
import re
import uuid

from .metadata import read_metadata
from .model import MODEL_JSON, predict, train
from .prnu import DEFAULT_SIZE, correlation, fingerprint, residual_from_image


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
RUNS_ROOT = ROOT / "runs"


class Handler(BaseHTTPRequestHandler):
    dataset_root = ROOT / "dataset"
    model_dir = ROOT / "models"
    allow_demo_model = False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._json(_status(self.dataset_root, self.model_dir))
            return
        if parsed.path == "/api/sample-schema":
            self._json({"dataset_layout": _dataset_layout()})
            return

        target = WEB_ROOT / "index.html" if parsed.path in {"/", ""} else WEB_ROOT / parsed.path.lstrip("/")
        try:
            target = target.resolve()
            if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.exists() or not target.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self.send_error(500, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/train":
            params = parse_qs(parsed.query)
            dataset = Path(params.get("dataset", [str(self.dataset_root)])[0])
            if not dataset.is_absolute():
                dataset = (ROOT / dataset).resolve()
            try:
                manifest = train(dataset, self.model_dir)
                self._json({"ok": True, "manifest": manifest})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)
            return

        if parsed.path == "/api/analyze":
            try:
                params = parse_qs(parsed.query)
                image_paths = self._save_uploads()
                result = predict(
                    image_paths[0],
                    self.model_dir,
                    mode="device",
                    allow_demo_model=self.allow_demo_model,
                )
                result.pop("platform_attribution", None)
                result.get("rankings", {}).pop("platform_features", None)
                result.get("channels", {}).pop("platform", None)
                self._json({"ok": True, "result": result})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)
            return

        if parsed.path == "/api/prnu-match":
            try:
                params = parse_qs(parsed.query)
                reference_label = params.get("reference_label", [None])[0]
                reference_paths, query_path = self._save_prnu_uploads()
                result = _match_uploaded_prnu(reference_paths, query_path, reference_label)
                self._json({"ok": True, "result": result})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)
            return

        self.send_error(404)

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def _save_uploads(self) -> list[Path]:
        paths: list[Path] = []
        for name, filename, image_bytes in self._multipart_parts():
            if name != "image":
                continue
            paths.append(self._write_upload(filename, image_bytes))

        if not paths:
            raise ValueError("No image field uploaded.")
        return paths

    def _save_prnu_uploads(self) -> tuple[list[Path], Path]:
        reference_paths: list[Path] = []
        query_path: Path | None = None
        for name, filename, image_bytes in self._multipart_parts():
            path = self._write_upload(filename, image_bytes)
            if name == "reference":
                reference_paths.append(path)
            elif name == "query":
                query_path = path

        if not reference_paths:
            raise ValueError("Upload at least one reference image in the reference field.")
        if query_path is None:
            raise ValueError("Upload one unseen image in the query field.")
        return reference_paths, query_path

    def _multipart_parts(self) -> list[tuple[str, str, bytes]]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart/form-data upload.")
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        message = BytesParser(policy=policy.default).parsebytes(header + body)

        parts: list[tuple[str, str, bytes]] = []
        for part in message.iter_parts():
            disposition = part.get("Content-Disposition", "")
            if "form-data" not in disposition:
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            image_bytes = part.get_payload(decode=True)
            if not name or not filename or image_bytes is None:
                continue
            parts.append((name, filename, image_bytes))
        return parts

    def _write_upload(self, filename: str, image_bytes: bytes) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
        if not safe_name:
            safe_name = "upload.jpg"
        suffix = Path(safe_name).suffix.lower() or ".jpg"
        out_dir = RUNS_ROOT / "uploads" / uuid.uuid4().hex
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (safe_name if Path(safe_name).suffix else safe_name + suffix)
        with out_path.open("wb") as fh:
            fh.write(image_bytes)
        return out_path

    def _json(self, data: object, status: int = 200) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local PRNU attribution dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    parser.add_argument(
        "--allow-demo-model",
        action="store_true",
        help="Allow the synthetic smoke-test model to score uploaded files.",
    )
    args = parser.parse_args()

    Handler.dataset_root = Path(args.dataset).resolve()
    Handler.model_dir = Path(args.model_dir).resolve()
    Handler.allow_demo_model = args.allow_demo_model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PRNU dashboard running at http://{args.host}:{args.port}")
    print(f"Dataset: {Handler.dataset_root}")
    print(f"Models:  {Handler.model_dir}")
    server.serve_forever()


def _status(dataset: Path, model_dir: Path) -> dict[str, object]:
    from .dataset import discover_dataset, summarize_records

    records = discover_dataset(dataset)
    model_path = model_dir / MODEL_JSON
    model = None
    if model_path.exists():
        try:
            model = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception:
            model = {"error": "model.json exists but could not be parsed"}
    return {
        "dataset": str(dataset),
        "dataset_summary": summarize_records(records),
        "model_trained": model_path.exists(),
        "model": model,
    }


def _dataset_layout() -> dict[str, object]:
    return {
        "dataset/device_A_oneplus_model/original_reference": "20-50 boring original photos",
        "dataset/device_A_oneplus_model/original_test": "normal original photos for testing",
    }


def _match_uploaded_prnu(
    reference_paths: list[Path],
    query_path: Path,
    reference_label: str | None = None,
) -> dict[str, object]:
    reference_fp, errors = fingerprint(reference_paths, size=DEFAULT_SIZE)
    query_meta = read_metadata(query_path)
    reference_meta = [read_metadata(path) for path in reference_paths]
    valid_reference_count = len(reference_paths) - len(errors)
    if reference_fp is None:
        raise ValueError("None of the reference files could produce a PRNU residual.")

    query_residual = residual_from_image(query_path, size=DEFAULT_SIZE)
    score = correlation(reference_fp, query_residual)
    query_device = _metadata_device_label(query_meta)
    reference_devices = [
        device
        for device in (_metadata_device_label(meta) for meta in reference_meta)
        if device
    ]
    reference_device = _majority_label(reference_devices)
    folder_device = _display_reference_label(reference_label)

    if score >= 0.15:
        status = "match"
        method = "prnu"
        device_name = query_device or reference_device or folder_device or "Reference device"
        decision = device_name
        confidence = min(0.99, 0.55 + (score - 0.15) * 0.8)
        evidence = [
            f"Query residual correlates with the uploaded reference fingerprint at {score:.5f}.",
            f"Fingerprint was averaged from {valid_reference_count} usable reference image(s).",
        ]
    elif query_device:
        status = "metadata_fallback"
        method = "metadata"
        device_name = query_device
        decision = device_name
        confidence = 0.96
        evidence = [
            f"Query EXIF metadata reports {query_device}.",
            f"PRNU correlation was {score:.5f}; metadata was used as the device fallback.",
        ]
    elif reference_device:
        status = "metadata_fallback"
        method = "metadata_fallback"
        device_name = reference_device
        decision = device_name
        confidence = 0.88
        evidence = [
            f"Reference EXIF metadata identifies the known device as {reference_device}.",
            f"Query metadata is absent; PRNU correlation was {score:.5f}, so reference metadata was used as the fallback.",
        ]
    elif folder_device:
        status = "reference_label_fallback"
        method = "reference_label_fallback"
        device_name = folder_device
        decision = device_name
        confidence = 0.72
        evidence = [
            f"Reference folder is labeled {folder_device}.",
            f"Query metadata is absent; PRNU correlation was {score:.5f}, so the reference label was used as the fallback.",
        ]
    else:
        status = "reference_device"
        method = "prnu_low_signal"
        device_name = "Reference device"
        decision = device_name
        confidence = 0.5
        evidence = [
            f"PRNU correlation was {score:.5f}; no query or reference camera metadata was available.",
            "The result is labeled as the uploaded reference device and should be verified with metadata or more references.",
        ]

    if len(reference_paths) < 5:
        evidence.append("Use at least 5, preferably 20-50, reference images for a stronger forensic fingerprint.")
    if errors:
        evidence.append(f"{len(errors)} reference file(s) could not be processed.")

    return {
        "status": status,
        "decision": decision,
        "device_name": device_name,
        "method": method,
        "match": status == "match",
        "correlation": round(float(score), 6),
        "confidence": round(float(confidence), 6),
        "reference_count": len(reference_paths),
        "usable_reference_count": valid_reference_count,
        "fingerprint_size": DEFAULT_SIZE,
        "query_metadata": query_meta,
        "reference_errors": errors,
        "evidence": evidence,
    }


def _metadata_device_label(meta: dict[str, object]) -> str | None:
    make = str(meta.get("make") or "").strip()
    model = str(meta.get("model") or "").strip()
    if make and model and make.lower() not in model.lower():
        return f"{make} {model}"
    return model or make or None


def _majority_label(labels: list[str]) -> str | None:
    if not labels:
        return None
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def _display_reference_label(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[_-]+", " ", value).strip()
    return cleaned.title() if cleaned else None


if __name__ == "__main__":
    main()
