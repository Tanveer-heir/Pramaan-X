---
name: ai-ml-engineering
description: Machine Learning and AI engineering discipline. Covers model inference pipelines (PyTorch, ONNX), perceptual hashing (PDQ), vector search (FAISS), batching, device management (CUDA/CPU fallback), and evaluation metrics.
---

# AI & Machine Learning Engineering

This skill defines standards and patterns for model inference, computer vision pipelines, perceptual hashing, vector search, and model evaluation—specifically tailored for media forensics, provenance verification, and origin tracing.

---

## 1. Inference Pipeline Architecture

### A. Device Management & Graceful Hardware Fallback
Never hardcode `cuda:0`. Always write hardware-resilient initialization with automatic fallback:

```python
import torch

def get_compute_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_compute_device()
```

### B. Memory Hygiene & OOM Prevention
* **Inference Context**: Always disable gradient calculation and set evaluation mode:
  ```python
  model.eval()
  with torch.inference_mode(): # Faster than torch.no_grad()
      outputs = model(inputs.to(DEVICE))
  ```
* **CUDA Cache Cleanup**: In batch loops or after processing heavy inputs, clear allocations to prevent fragmentation:
  ```python
  import gc
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  gc.collect()
  ```
* **Half-Precision (FP16 / BF16)**: Use `torch.autocast(device_type="cuda", dtype=torch.float16)` to cut GPU VRAM consumption by ~50% and double throughput on supported GPUs.

---

## 2. Perceptual Hashing & Origin Tracing Pipelines

For tracking image dissemination and near-duplicate variants:

### A. PDQ Perceptual Hashing & Hamming Distance
* PDQ produces a 256-bit hash resilient to cropping, compression, and scaling.
* Near-duplicate clustering:
  - Hamming Distance $\le 30$: High-confidence match (identical or light compression/re-encoding).
  - Hamming Distance $31 - 60$: Moderate similarity (significant crop, watermark added, filter applied).
  - Hamming Distance $> 60$: Unrelated or heavily transformed media.

```python
def compute_hamming_distance(hash1_bits: int, hash2_bits: int) -> int:
    """Compute bitwise Hamming distance between two binary hashes."""
    return bin(hash1_bits ^ hash2_bits).count("1")
```

### B. Dense Vector Search (FAISS / Vector Store)
* Use cosine similarity or inner product with normalized embeddings (e.g. CLIP / ViT / OpenCLIP):
  ```python
  import faiss
  import numpy as np

  # Normalize embeddings to unit norm for cosine similarity via IndexFlatIP
  faiss.normalize_L2(embeddings)
  index = faiss.IndexFlatIP(dimension)
  index.add(embeddings)

  # Query top-K
  distances, indices = index.search(query_embedding, k=10)
  ```

---

## 3. Acceleration with ONNX Runtime

When deploying deep learning models (e.g., face detection, deepfake classifiers, feature extractors) to production:
* Export PyTorch models to **ONNX** format for portable, high-speed execution:
  ```python
  import onnxruntime as ort

  providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
  session = ort.InferenceSession("models/classifier.onnx", providers=providers)
  outputs = session.run(None, {"input": input_tensor.numpy()})
  ```
* Benefit: Eliminates PyTorch runtime overhead in lightweight microservices, reduces Docker image size by gigabytes.

---

## 4. Evaluation & Verification Metrics

Never declare a model or detection threshold "ready" without quantitative verification:

| Metric | Formula | Primary Use in Forensics |
| :--- | :--- | :--- |
| **Precision** | $\frac{TP}{TP + FP}$ | Minimizing False Positives (crucial: avoid falsely accusing authentic media of being fake). |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Catching all manipulated instances in high-stakes monitoring. |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Balanced measure on imbalanced test distributions. |
| **AUC-ROC / EER** | Receiver Operating Characteristic / Equal Error Rate | Evaluating classifier threshold discrimination across all operating points. |

---

## 5. Defensive Edge-Case Handling

* **Corrupt Images**: Catch `PIL.UnidentifiedImageError` and truncated image exceptions gracefully.
* **Aspect Ratio & Sizing**: Check image dimensions before resizing; reject 0x0 or single-pixel dummy payloads before passing to model tensors.
* **Color Space Consistency**: Verify conversion (e.g. `BGR -> RGB` when using OpenCV with PyTorch / PIL).
