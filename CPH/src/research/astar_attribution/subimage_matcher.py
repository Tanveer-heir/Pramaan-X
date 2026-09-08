"""
Sub-Image Homography & Crop Detection Engine
Chandigarh Police Hackathon - A* Attribution Research Track

Detects exact image clones and partial crops/sub-images using:
1. Meta PDQ Hash (256-bit perceptual hamming distance).
2. SIFT / ORB feature extraction.
3. FLANN / BFMatcher with Lowe's ratio test (0.75).
4. RANSAC Planar Homography estimation (5.0 px reprojection threshold).
5. Geometric containment & bounding box localization for sub-region identification.
"""

import os
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np

try:
    import pdqhash
    HAS_PDQ = True
except ImportError:
    HAS_PDQ = False

try:
    from src.common.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Dataclass encapsulating image matching and crop detection results."""
    is_match: bool
    is_partial_crop: bool
    crop_type: str  # 'exact', 'target_is_crop_of_candidate', 'candidate_is_crop_of_target', 'none'
    pdq_distance: int
    inlier_ratio: float
    inlier_count: int
    bounding_box: Optional[List[int]]  # [x, y, w, h]
    scale_factor: float
    similarity_score: float
    details: str


class SubImageMatcher:
    """
    Forensic engine to identify exact copies, resized variations,
    and partial crops/sub-images between query targets and candidate media.
    """

    def __init__(
        self,
        min_inliers: int = 12,
        inlier_ratio_threshold: float = 0.15,
        pdq_threshold: int = 30,
        ratio_test_threshold: float = 0.75,
        ransac_reproj_threshold: float = 5.0,
        prefer_sift: bool = True,
    ):
        self.min_inliers = min_inliers
        self.inlier_ratio_threshold = inlier_ratio_threshold
        self.pdq_threshold = pdq_threshold
        self.ratio_test_threshold = ratio_test_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold

        # Initialize detector: SIFT preferred with ORB fallback
        self.detector_name = "NONE"
        self.detector = None
        self.norm_type = cv2.NORM_L2

        if prefer_sift:
            try:
                self.detector = cv2.SIFT_create()
                self.detector_name = "SIFT"
                self.norm_type = cv2.NORM_L2
            except Exception as e:
                logger.warning(f"Failed to initialize SIFT ({e}), falling back to ORB.")

        if self.detector is None:
            try:
                self.detector = cv2.ORB_create(nfeatures=3000)
                self.detector_name = "ORB"
                self.norm_type = cv2.NORM_HAMMING
            except Exception as e:
                logger.error(f"Failed to initialize ORB detector: {e}")

        # Initialize matcher
        self.matcher = cv2.BFMatcher(self.norm_type)

        # Feature and PDQ in-memory caches to prevent redundant re-computation
        self.feature_cache: Dict[str, Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]] = {}
        self.pdq_cache: Dict[str, Tuple[Optional[np.ndarray], int]] = {}

    def compute_pdq(self, img_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """Computes Meta's 256-bit PDQ hash and quality score."""
        if not HAS_PDQ or img_rgb is None:
            return None, 0
        try:
            hash_vec, quality = pdqhash.compute(img_rgb)
            return hash_vec, int(quality)
        except Exception as e:
            logger.warning(f"PDQ hash computation failed: {e}")
            return None, 0

    def compute_pdq_distance(
        self, hash1: Optional[np.ndarray], hash2: Optional[np.ndarray]
    ) -> int:
        """Calculates bitwise Hamming distance between two 256-bit PDQ vectors."""
        if hash1 is None or hash2 is None:
            return 999
        try:
            return int(np.count_nonzero(hash1 != hash2))
        except Exception:
            return 999

    def extract_features(
        self, img_bgr: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extracts keypoints and descriptors using the configured detector (SIFT/ORB)."""
        if self.detector is None or img_bgr is None:
            return [], None
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            max_dim = max(h, w)
            if max_dim > 1920:
                scale = 1920.0 / float(max_dim)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                kp, des = self.detector.detectAndCompute(gray_small, None)
                inv_s = 1.0 / scale
                for p in kp:
                    p.pt = (p.pt[0] * inv_s, p.pt[1] * inv_s)
                return kp, des
            else:
                kp, des = self.detector.detectAndCompute(gray, None)
                return kp, des
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return [], None

    def match_features(
        self, des1: Optional[np.ndarray], des2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        """Matches feature descriptors using k-NN with Lowe's ratio test (0.75)."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []

        try:
            raw_matches = self.matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.ratio_test_threshold * n.distance:
                        good_matches.append(m)
            return good_matches
        except Exception as e:
            logger.warning(f"Feature matching failed: {e}")
            return []

    def _compute_scale(self, H: np.ndarray) -> float:
        """Computes scale factor from 3x3 homography matrix."""
        if H is None or H.shape != (3, 3):
            return 1.0
        try:
            h22 = H[2, 2]
            if abs(h22) < 1e-9:
                return 1.0
            H_norm = H / h22
            det = np.linalg.det(H_norm[:2, :2])
            scale = float(np.sqrt(abs(det)))
            return scale if np.isfinite(scale) and scale > 1e-6 else 1.0
        except Exception:
            return 1.0

    def compare_images(self, target_path: str, candidate_path: str) -> MatchResult:
        """
        Compares a target image against a candidate image.

        Evaluates:
        - Perceptual duplicate status (PDQ Hamming distance <= 30)
        - Planar homography alignment (RANSAC with Lowe's ratio test)
        - Partial crop detection (Target is crop of Candidate, or vice versa)
        - Extracted bounding box and scale factor

        Returns:
            MatchResult
        """
        # 1. Validate file existence
        if not os.path.isfile(target_path):
            return MatchResult(
                is_match=False,
                is_partial_crop=False,
                crop_type="none",
                pdq_distance=999,
                inlier_ratio=0.0,
                inlier_count=0,
                bounding_box=None,
                scale_factor=1.0,
                similarity_score=0.0,
                details=f"Target file does not exist: {target_path}",
            )

        if not os.path.isfile(candidate_path):
            return MatchResult(
                is_match=False,
                is_partial_crop=False,
                crop_type="none",
                pdq_distance=999,
                inlier_ratio=0.0,
                inlier_count=0,
                bounding_box=None,
                scale_factor=1.0,
                similarity_score=0.0,
                details=f"Candidate file does not exist: {candidate_path}",
            )

        # 2. Read images
        img_target_bgr = cv2.imread(target_path)
        img_candidate_bgr = cv2.imread(candidate_path)

        if img_target_bgr is None:
            return MatchResult(
                is_match=False,
                is_partial_crop=False,
                crop_type="none",
                pdq_distance=999,
                inlier_ratio=0.0,
                inlier_count=0,
                bounding_box=None,
                scale_factor=1.0,
                similarity_score=0.0,
                details=f"Failed to decode target image: {target_path}",
            )

        if img_candidate_bgr is None:
            return MatchResult(
                is_match=False,
                is_partial_crop=False,
                crop_type="none",
                pdq_distance=999,
                inlier_ratio=0.0,
                inlier_count=0,
                bounding_box=None,
                scale_factor=1.0,
                similarity_score=0.0,
                details=f"Failed to decode candidate image: {candidate_path}",
            )

        ht, wt = img_target_bgr.shape[:2]
        hc, wc = img_candidate_bgr.shape[:2]

        # 3. Compute PDQ perceptual hashes & Hamming distance with caching
        if target_path in self.pdq_cache:
            hash_t, _ = self.pdq_cache[target_path]
        else:
            img_target_rgb = cv2.cvtColor(img_target_bgr, cv2.COLOR_BGR2RGB)
            hash_t, q_t = self.compute_pdq(img_target_rgb)
            self.pdq_cache[target_path] = (hash_t, q_t)

        if candidate_path in self.pdq_cache:
            hash_c, _ = self.pdq_cache[candidate_path]
        else:
            img_candidate_rgb = cv2.cvtColor(img_candidate_bgr, cv2.COLOR_BGR2RGB)
            hash_c, q_c = self.compute_pdq(img_candidate_rgb)
            self.pdq_cache[candidate_path] = (hash_c, q_c)

        pdq_dist = self.compute_pdq_distance(hash_t, hash_c)

        # 4. Extract features & match with caching
        if target_path in self.feature_cache:
            kp_t, des_t = self.feature_cache[target_path]
        else:
            kp_t, des_t = self.extract_features(img_target_bgr)
            self.feature_cache[target_path] = (kp_t, des_t)

        if candidate_path in self.feature_cache:
            kp_c, des_c = self.feature_cache[candidate_path]
        else:
            kp_c, des_c = self.extract_features(img_candidate_bgr)
            self.feature_cache[candidate_path] = (kp_c, des_c)

        good_matches = self.match_features(des_t, des_c)

        inlier_count = 0
        inlier_ratio = 0.0
        H = None
        mask = None
        src_span_x = 0.0
        src_span_y = 0.0
        dst_span_x = 0.0
        dst_span_y = 0.0

        if len(good_matches) >= 4:
            src_pts = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_c[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold)
            if mask is not None:
                inlier_count = int(np.sum(mask))
                inlier_ratio = float(inlier_count / len(good_matches)) if len(good_matches) > 0 else 0.0
                if inlier_count > 0:
                    inlier_src = src_pts[mask.ravel() == 1].reshape(-1, 2)
                    inlier_dst = dst_pts[mask.ravel() == 1].reshape(-1, 2)
                    src_span_x = float((inlier_src[:, 0].max() - inlier_src[:, 0].min()) / wt)
                    src_span_y = float((inlier_src[:, 1].max() - inlier_src[:, 1].min()) / ht)
                    dst_span_x = float((inlier_dst[:, 0].max() - inlier_dst[:, 0].min()) / wc)
                    dst_span_y = float((inlier_dst[:, 1].max() - inlier_dst[:, 1].min()) / hc)

        # Check planar alignment condition
        min_required_inliers = 18 if pdq_dist > 40 else self.min_inliers
        has_planar_alignment = (
            H is not None
            and inlier_count >= min_required_inliers
            and inlier_ratio >= self.inlier_ratio_threshold
        )

        # 5. Evaluate exact / near clone via PDQ
        if pdq_dist <= self.pdq_threshold:
            scale = float(round(self._compute_scale(H), 4)) if H is not None else 1.0
            sim = float(round(max(0.0, 1.0 - pdq_dist / 256.0), 4))
            return MatchResult(
                is_match=True,
                is_partial_crop=False,
                crop_type="exact",
                pdq_distance=pdq_dist,
                inlier_ratio=inlier_ratio,
                inlier_count=inlier_count,
                bounding_box=[0, 0, wc, hc],
                scale_factor=scale,
                similarity_score=sim,
                details=(
                    f"Exact/near clone detected via PDQ Hash (Hamming dist={pdq_dist} <= {self.pdq_threshold}, "
                    f"inliers={inlier_count}, ratio={inlier_ratio:.2f})."
                ),
            )

        # 6. Evaluate partial crop / sub-image match via Homography
        if has_planar_alignment:
            try:
                # Geometric corner transformation
                t_corners = np.float32([[0, 0], [wt, 0], [wt, ht], [0, ht]]).reshape(-1, 1, 2)
                c_corners = np.float32([[0, 0], [wc, 0], [wc, hc], [0, hc]]).reshape(-1, 1, 2)

                target_in_cand = cv2.perspectiveTransform(t_corners, H)

                # Invert H to project candidate onto target
                cand_in_target = None
                det_H = np.linalg.det(H)
                if abs(det_H) > 1e-12:
                    inv_H = np.linalg.inv(H)
                    cand_in_target = cv2.perspectiveTransform(c_corners, inv_H)

                # Boundary containment checks (with 5% margin tolerance)
                tol_c = 0.05 * max(wc, hc)
                tx_min, ty_min = target_in_cand[:, 0, 0].min(), target_in_cand[:, 0, 1].min()
                tx_max, ty_max = target_in_cand[:, 0, 0].max(), target_in_cand[:, 0, 1].max()
                target_in_cand_inside = (
                    tx_min >= -tol_c
                    and ty_min >= -tol_c
                    and tx_max <= wc + tol_c
                    and ty_max <= hc + tol_c
                )

                cand_in_target_inside = False
                area_c_in_t = 0.0
                area_t = float(wt * ht)
                if cand_in_target is not None:
                    tol_t = 0.05 * max(wt, ht)
                    cx_min, cy_min = cand_in_target[:, 0, 0].min(), cand_in_target[:, 0, 1].min()
                    cx_max, cy_max = cand_in_target[:, 0, 0].max(), cand_in_target[:, 0, 1].max()
                    cand_in_target_inside = (
                        cx_min >= -tol_t
                        and cy_min >= -tol_t
                        and cx_max <= wt + tol_t
                        and cy_max <= ht + tol_t
                    )
                    area_c_in_t = abs(cv2.contourArea(cand_in_target))

                # Area coverage check
                area_t_in_c = abs(cv2.contourArea(target_in_cand))
                area_c = float(wc * hc)
                coverage_t_in_c = (area_t_in_c / area_c) if area_c > 0 else 1.0
                coverage_c_in_t = (area_c_in_t / area_t) if area_t > 0 else 1.0

                scale = float(round(self._compute_scale(H), 4))
                sim_score = float(round(min(1.0, 0.5 * inlier_ratio + 0.5 * min(1.0, inlier_count / 50.0)), 4))

                # Geometric sanity checks to prevent degenerate homography collapse onto tiny text/logos:
                # 1. Scale factor must be within realistic visual bounds [0.15x, 6.5x]
                # 2. Inliers must not be confined to a tiny localized patch/logo (< 18% width or height)
                is_scale_valid = (0.15 <= scale <= 6.5)

                # Case 6a: Target is a sub-region / crop of Candidate
                # Target is the crop, so inliers must span across Target's field of view
                target_has_dispersion = (
                    (src_span_x * src_span_y >= 0.025)
                    and max(src_span_x, src_span_y) >= 0.18
                    and min(src_span_x, src_span_y) >= 0.08
                )
                if target_in_cand_inside and is_scale_valid and target_has_dispersion and (not cand_in_target_inside or coverage_t_in_c < 0.90):
                    bx = max(0, int(round(tx_min)))
                    by = max(0, int(round(ty_min)))
                    bw = max(1, min(wc - bx, int(round(tx_max - tx_min))))
                    bh = max(1, min(hc - by, int(round(ty_max - ty_min))))

                    if bw >= 80 and bh >= 80 and (bw * bh) >= (0.04 * area_c):
                        return MatchResult(
                            is_match=True,
                            is_partial_crop=True,
                            crop_type="target_is_crop_of_candidate",
                            pdq_distance=pdq_dist,
                            inlier_ratio=inlier_ratio,
                            inlier_count=inlier_count,
                            bounding_box=[bx, by, bw, bh],
                            scale_factor=scale,
                            similarity_score=sim_score,
                            details=(
                                f"Target is a cropped sub-region of Candidate (bbox=[{bx}, {by}, {bw}, {bh}] in candidate, "
                                f"scale={scale:.3f}, inliers={inlier_count}, ratio={inlier_ratio:.2f})."
                            ),
                        )

                # Case 6b: Candidate is a sub-region / crop of Target
                # Candidate is the crop, so inliers must span across Candidate's field of view
                cand_has_dispersion = (
                    (dst_span_x * dst_span_y >= 0.025)
                    and max(dst_span_x, dst_span_y) >= 0.18
                    and min(dst_span_x, dst_span_y) >= 0.08
                )
                if cand_in_target_inside and is_scale_valid and cand_has_dispersion and (not target_in_cand_inside or coverage_c_in_t < 0.90):
                    bx = max(0, int(round(cx_min)))
                    by = max(0, int(round(cy_min)))
                    bw = max(1, min(wt - bx, int(round(cx_max - cx_min))))
                    bh = max(1, min(ht - by, int(round(cy_max - cy_min))))
                    inv_scale = float(round(1.0 / scale, 4)) if scale > 1e-6 else 1.0

                    if bw >= 80 and bh >= 80 and (bw * bh) >= (0.04 * area_t):
                        return MatchResult(
                            is_match=True,
                            is_partial_crop=True,
                            crop_type="candidate_is_crop_of_target",
                            pdq_distance=pdq_dist,
                            inlier_ratio=inlier_ratio,
                            inlier_count=inlier_count,
                            bounding_box=[bx, by, bw, bh],
                            scale_factor=inv_scale,
                            similarity_score=sim_score,
                            details=(
                                f"Candidate is a cropped sub-region of Target (bbox=[{bx}, {by}, {bw}, {bh}] in target, "
                                f"scale={inv_scale:.3f}, inliers={inlier_count}, ratio={inlier_ratio:.2f})."
                            ),
                        )

                # Case 6c: Geometric alignment matches the full frame (~100% overlap)
                broad_dispersion = (
                    (src_span_x * src_span_y >= 0.04 and dst_span_x * dst_span_y >= 0.04)
                    and max(src_span_x, src_span_y) >= 0.22
                    and max(dst_span_x, dst_span_y) >= 0.22
                )
                if broad_dispersion and (pdq_dist <= 42 or (inlier_count >= 35 and inlier_ratio >= 0.40)):
                    return MatchResult(
                        is_match=True,
                        is_partial_crop=False,
                        crop_type="exact",
                        pdq_distance=pdq_dist,
                        inlier_ratio=inlier_ratio,
                        inlier_count=inlier_count,
                        bounding_box=[0, 0, wc, hc],
                        scale_factor=scale,
                        similarity_score=sim_score,
                        details=(
                            f"Full-frame planar geometric match (coverage={coverage_t_in_c:.2f}, "
                            f"inliers={inlier_count}, ratio={inlier_ratio:.2f}, PDQ dist={pdq_dist})."
                        ),
                    )

            except Exception as e:
                logger.warning(f"Geometric verification failed ({e}), evaluating fallback.")

        # 7. No match found
        pdq_sim = float(round(max(0.0, 1.0 - pdq_dist / 256.0), 4)) if pdq_dist < 999 else 0.0
        return MatchResult(
            is_match=False,
            is_partial_crop=False,
            crop_type="none",
            pdq_distance=pdq_dist,
            inlier_ratio=inlier_ratio,
            inlier_count=inlier_count,
            bounding_box=None,
            scale_factor=1.0,
            similarity_score=pdq_sim,
            details=(
                f"No clone or partial crop match found (PDQ distance={pdq_dist}, "
                f"inlier_count={inlier_count}/{len(good_matches)}, ratio={inlier_ratio:.2f})."
            ),
        )

    def visualize_match(
        self,
        target_path: str,
        candidate_path: str,
        output_path: str,
        result: Optional[MatchResult] = None,
    ) -> bool:
        """
        Visualizes the match by drawing the bounding box on the host image
        and side-by-side comparison, saving to output_path.
        """
        if result is None:
            result = self.compare_images(target_path, candidate_path)

        if not result.is_match or result.bounding_box is None:
            return False

        try:
            img_t = cv2.imread(target_path)
            img_c = cv2.imread(candidate_path)
            if img_t is None or img_c is None:
                return False

            bx, by, bw, bh = result.bounding_box

            if result.crop_type == "target_is_crop_of_candidate":
                # Draw bounding box on candidate
                annotated = img_c.copy()
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
                label = f"Sub-Image Match (Scale: {result.scale_factor:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (bx, max(20, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, annotated)
                return True

            elif result.crop_type == "candidate_is_crop_of_target":
                # Draw bounding box on target
                annotated = img_t.copy()
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
                label = f"Sub-Image Match (Scale: {result.scale_factor:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (bx, max(20, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, annotated)
                return True

            return False
        except Exception as e:
            logger.error(f"Failed to visualize match: {e}")
            return False
