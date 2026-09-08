"""
Chandigarh Police Hackathon - Section 2 Package.
Configures runtime environment to prioritize PyTorch and prevent TensorFlow/Protobuf conflicts.
"""
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
