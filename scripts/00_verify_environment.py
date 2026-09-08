import cv2
import torch
import numpy as np

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)
    print("VRAM:", round(props.total_memory / 1024**3, 2), "GB")

    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x

    print("GPU matrix multiplication: SUCCESS")
    print(
        "Allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        "GB",
    )

print("=" * 60)