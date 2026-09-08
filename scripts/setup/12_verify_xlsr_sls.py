"""Validate the official XLSR-SLS ONNX checkpoint and local CUDA runtime.

This is a compatibility gate, not dataset extraction. It downloads only when
``--download`` is supplied, verifies the pinned SHA-256, locates the internal
1024-dimensional SLS embedding tensor, and runs one exact 64,600-sample window.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:
    resource = None
import subprocess
import threading
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "xlsr_sls"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "xlsr-sls.onnx"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "interim" / "setup_logs" / "xlsr_sls_compatibility.json"

MODEL_REPO = "SpeechAntiSpoofingBenchmarks/XLSR-SLS"
MODEL_FILENAME = "xlsr-sls.onnx"
MODEL_REVISION = "1ce22562df9eda0c39209140003206e9c134a61e"
MODEL_SHA256 = "0aa36298a1a893bfb58a8d721dd30d046157f559da1165f95d2a0b69988c1bbd"
MINIMUM_MODEL_SIZE_BYTES = 1_000_000_000
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 64_600
EMBEDDING_DIM = 1_024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--download", action="store_true", help="Download the pinned public checkpoint if absent.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-forward", action="store_true", help="Validate file and graph only.")
    return parser


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path.exists():
        return args.model_path
    if not args.download:
        raise FileNotFoundError(
            f"XLSR-SLS checkpoint not found at {args.model_path}. "
            "Rerun with --download to fetch the pinned public model."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install requirements-audio.txt before using --download.") from exc
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=args.model_path.parent,
        )
    )
    if downloaded.resolve() != args.model_path.resolve():
        raise RuntimeError(f"Downloaded checkpoint path differs from requested path: {downloaded}")
    return downloaded


def locate_sls_embedding_tensor(model: Any) -> tuple[str, str]:
    """Find the 1024-d tensor consumed by the final 1024-to-2 classifier."""
    initializer_shapes = {
        initializer.name: tuple(int(dimension) for dimension in initializer.dims)
        for initializer in model.graph.initializer
    }
    classifier_weights = {
        name
        for name, shape in initializer_shapes.items()
        if len(shape) == 2 and sorted(shape) == [2, EMBEDDING_DIM]
    }
    candidates: list[tuple[str, str]] = []
    for node in model.graph.node:
        if not classifier_weights.intersection(node.input):
            continue
        data_inputs = [name for name in node.input if name and name not in initializer_shapes]
        if len(data_inputs) == 1:
            candidates.append((data_inputs[0], node.name or node.op_type))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one final 1024-to-2 classifier, found {len(candidates)} candidates.")
    return candidates[0]


def concrete_input_shape(metadata_shape: list[Any]) -> tuple[int, ...]:
    rank = len(metadata_shape)
    if rank == 2:
        return (1, WINDOW_SAMPLES)
    if rank == 3:
        return (1, WINDOW_SAMPLES, 1)
    raise RuntimeError(f"Unexpected XLSR-SLS input rank: {rank}, metadata={metadata_shape}")


def resolve_providers(ort: Any, device: str, memory_limit_gb: float) -> tuple[str, list[Any]]:
    available = ort.get_available_providers()
    resolved = "cuda" if device == "auto" and "CUDAExecutionProvider" in available else device
    if resolved == "auto":
        resolved = "cpu"
    if resolved == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable. Available providers: {available}")
        limit_bytes = int(memory_limit_gb * 1024**3)
        return resolved, [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "gpu_mem_limit": limit_bytes,
                "arena_extend_strategy": "kSameAsRequested",
            }),
            "CPUExecutionProvider",
        ]
    return "cpu", ["CPUExecutionProvider"]


def peak_host_rss_bytes() -> int:
    if resource is None:
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024) if platform.system() == "Linux" else int(value)


def parse_process_gpu_memory(output: str, process_id: int) -> int:
    total_mib = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid, memory_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if pid == process_id:
            total_mib += memory_mib
    return total_mib * 1024**2


class GPUMemoryMonitor:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.peak_bytes = 0
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.wait(0.1):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                self.available = True
                self.peak_bytes = max(self.peak_bytes, parse_process_gpu_memory(result.stdout, os.getpid()))

    def start(self) -> None:
        if self.enabled:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install the isolated audio dependencies from requirements-audio.txt.") from exc

    model_path = resolve_model_path(args)
    size = model_path.stat().st_size
    if size < MINIMUM_MODEL_SIZE_BYTES:
        raise RuntimeError(f"Checkpoint is unexpectedly small: {size} bytes")
    digest = sha256_file(model_path)
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: expected {MODEL_SHA256}, got {digest}")

    graph_started = perf_counter()
    model = onnx.load_model(str(model_path), load_external_data=False)
    onnx.checker.check_model(model)
    embedding_tensor, classifier_node = locate_sls_embedding_tensor(model)
    graph_seconds = perf_counter() - graph_started
    del model
    gc.collect()

    resolved_device, providers = resolve_providers(ort, args.device, args.gpu_memory_limit_gb)
    report: dict[str, Any] = {
        "status": "PASS_GRAPH_ONLY" if args.skip_forward else "PASS",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "model_size_bytes": size,
        "model_sha256": digest,
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE,
        "sls_embedding_dim": EMBEDDING_DIM,
        "sls_embedding_tensor": embedding_tensor,
        "classifier_node": classifier_node,
        "resolved_device": resolved_device,
        "available_providers": ort.get_available_providers(),
        "onnx_graph_validation_seconds": graph_seconds,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }
    if args.skip_forward:
        report["peak_host_rss_bytes"] = peak_host_rss_bytes()
        return report

    if resolved_device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA validation requires the project's CUDA-enabled PyTorch installation.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        report.update({
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_cudnn_version": torch.backends.cudnn.version(),
        })
        # Importing PyTorch loads its bundled CUDA and cuDNN libraries. ORT
        # 1.21+ can then preload any remaining compatible runtime libraries.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    gpu_monitor = GPUMemoryMonitor(resolved_device == "cuda")
    gpu_monitor.start()
    try:
        session_started = perf_counter()
        session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
        report["session_creation_seconds"] = perf_counter() - session_started
        report["active_providers"] = session.get_providers()
        if resolved_device == "cuda" and (not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider"):
            raise RuntimeError(f"CUDA was requested but the active provider order is {session.get_providers()}")
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Expected one waveform input, found {len(inputs)}")
        input_metadata = inputs[0]
        shape = concrete_input_shape(list(input_metadata.shape))
        rng = np.random.default_rng(args.seed)
        waveform = (rng.standard_normal(shape) * 0.01).astype(np.float32)
        forward_started = perf_counter()
        outputs = session.run(None, {input_metadata.name: waveform})
        forward_seconds = perf_counter() - forward_started
    finally:
        gpu_monitor.stop()
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one classifier output, found {len(outputs)}")
    output = np.asarray(outputs[0])
    if output.shape != (1, 2) or not np.isfinite(output).all():
        raise RuntimeError(f"Unexpected or non-finite output: shape={output.shape}")
    probability_sum = float(np.exp(output.astype(np.float64)).sum(axis=1)[0])
    if abs(probability_sum - 1.0) > 1e-3:
        raise RuntimeError(f"Output does not behave like two-class log probabilities: exp-sum={probability_sum}")
    report.update({
        "input_name": input_metadata.name,
        "input_metadata_shape": [str(value) for value in input_metadata.shape],
        "tested_input_shape": list(shape),
        "output_shape": list(output.shape),
        "output_log_probabilities": output[0].astype(float).tolist(),
        "bona_fide_log_probability": float(output[0, 1]),
        "forward_seconds": forward_seconds,
        "peak_process_gpu_vram_bytes": gpu_monitor.peak_bytes if gpu_monitor.available else None,
        "peak_host_rss_bytes": peak_host_rss_bytes(),
    })
    return report


def main() -> None:
    args = build_parser().parse_args()
    if args.gpu_memory_limit_gb <= 0:
        raise ValueError("--gpu-memory-limit-gb must be positive")
    try:
        report = run_gate(args)
    except Exception as exc:
        report = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "python_version": platform.python_version(),
            "onnx_version": package_version("onnx"),
            "onnxruntime_gpu_version": package_version("onnxruntime-gpu"),
            "huggingface_hub_version": package_version("huggingface-hub"),
        }
        write_report(args.report_path, report)
        print(json.dumps(report, sort_keys=True))
        raise
    write_report(args.report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
