"""
Video Keyframe Extraction & Temporal Localization Engine (§2.4b).
Extracts representative keyframes from video inputs and links them to timestamps.
"""

from typing import List, Dict, Any, Optional
import os
from src.common.logger import logger

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class VideoKeyframeExtractor:
    """
    Extracts representative keyframes from video files at 1 fps
    for cross-platform reverse search and multimodal matching.
    """

    def __init__(self, output_dir: str = "data/cache/keyframes"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def is_video(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"]

    def extract_keyframes(
        self,
        video_path: str,
        sample_rate_sec: float = 1.0,
        max_frames: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Extracts 1 frame per second from the video file.
        Returns metadata for each frame including timestamp_sec and local path.
        """
        if not os.path.exists(video_path):
            logger.warn("keyframe.video_not_found", path=video_path)
            return []

        logger.info("keyframe.extracting", video=video_path, rate=sample_rate_sec)
        frames = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        if HAS_OPENCV:
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_interval = int(fps * sample_rate_sec) or 1
                frame_count = 0
                saved_count = 0

                while cap.isOpened() and saved_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        timestamp_sec = round(frame_count / fps, 2)
                        out_path = os.path.join(
                            self.output_dir,
                            f"{base_name}_frame_{saved_count:03d}_{int(timestamp_sec)}s.jpg"
                        )
                        cv2.imwrite(out_path, frame)
                        frames.append({
                            "frame_idx": saved_count,
                            "timestamp_sec": timestamp_sec,
                            "timestamp_formatted": f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}",
                            "frame_path": out_path
                        })
                        saved_count += 1
                    frame_count += 1

                cap.release()
                if frames:
                    logger.info("keyframe.extracted_success", count=len(frames))
                    return frames
            except Exception as e:
                logger.warn("keyframe.opencv_error", error=str(e))

        logger.warning("keyframe.extraction_unavailable", video=video_path)
        return []
