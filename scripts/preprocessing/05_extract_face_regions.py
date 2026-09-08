from pathlib import Path
import argparse
import math

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAME_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "frames"
)

FACE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "faces"
)

MOUTH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "mouths"
)

EYE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "eyes"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "face_features"
)

QA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "face_qa"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "face_smoke_summary.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


# ============================================================
# MEDIAPIPE LANDMARK INDICES
# ============================================================

# Standard 6-point EAR layout
RIGHT_EYE_EAR = [
    33,
    160,
    158,
    133,
    153,
    144,
]

LEFT_EYE_EAR = [
    362,
    385,
    387,
    263,
    373,
    380,
]

# Additional eye landmarks for a cleaner eye crop
RIGHT_EYE_REGION = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]

LEFT_EYE_REGION = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
]

# Outer + inner lip landmarks
MOUTH_REGION = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
]


# ============================================================
# GEOMETRY
# ============================================================

def euclidean(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
    )


def normalized_to_pixels(
    landmark,
    width,
    height,
):
    x = int(
        np.clip(
            landmark.x * width,
            0,
            width - 1,
        )
    )

    y = int(
        np.clip(
            landmark.y * height,
            0,
            height - 1,
        )
    )

    return x, y


def landmark_points(
    landmarks,
    indices,
    width,
    height,
):
    return [
        normalized_to_pixels(
            landmarks[i],
            width,
            height,
        )
        for i in indices
    ]


def calculate_ear(points):
    """
    points = [p1, p2, p3, p4, p5, p6]

    EAR =
        (|p2-p6| + |p3-p5|)
        --------------------
             2 |p1-p4|
    """

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = euclidean(
        p2,
        p6,
    )

    vertical_2 = euclidean(
        p3,
        p5,
    )

    horizontal = euclidean(
        p1,
        p4,
    )

    if horizontal < 1e-6:
        return np.nan

    return (
        vertical_1
        + vertical_2
    ) / (
        2.0
        * horizontal
    )


def bbox_from_points(
    points,
    width,
    height,
    margin=0.15,
):
    xs = [
        p[0]
        for p in points
    ]

    ys = [
        p[1]
        for p in points
    ]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    box_width = (
        x_max
        - x_min
    )

    box_height = (
        y_max
        - y_min
    )

    x_margin = int(
        box_width
        * margin
    )

    y_margin = int(
        box_height
        * margin
    )

    x1 = max(
        0,
        x_min - x_margin,
    )

    y1 = max(
        0,
        y_min - y_margin,
    )

    x2 = min(
        width,
        x_max + x_margin,
    )

    y2 = min(
        height,
        y_max + y_margin,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def crop_from_bbox(
    image,
    bbox,
):
    x1, y1, x2, y2 = bbox

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return image[
        y1:y2,
        x1:x2,
    ].copy()


# ============================================================
# QA DRAWING
# ============================================================

def draw_bbox(
    image,
    bbox,
    color,
    label,
):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        image,
        label,
        (
            x1,
            max(20, y1 - 5),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


# ============================================================
# SAMPLE PROCESSING
# ============================================================

def process_sample(
    sample_dir,
    landmarker,
    fps,
):
    sample_id = (
        sample_dir.name
    )

    frame_paths = sorted(
        sample_dir.glob(
            "frame_*.jpg"
        )
    )

    face_dir = (
        FACE_ROOT
        / sample_id
    )

    mouth_dir = (
        MOUTH_ROOT
        / sample_id
    )

    eye_dir = (
        EYE_ROOT
        / sample_id
    )

    face_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mouth_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eye_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    QA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    detected_frames = 0
    qa_saved = False

    for frame_index, frame_path in enumerate(
        frame_paths
    ):

        image_bgr = cv2.imread(
            str(frame_path)
        )

        if image_bgr is None:

            print(
                f"  WARNING: failed to read "
                f"{frame_path.name}"
            )

            continue

        height, width = (
            image_bgr.shape[:2]
        )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=image_rgb,
        )

        result = landmarker.detect(
            mp_image
        )

        timestamp_sec = (
            frame_index
            / fps
        )

        base_row = {
            "frame_name": (
                frame_path.name
            ),
            "frame_index": (
                frame_index
            ),
            "timestamp_sec": (
                timestamp_sec
            ),
            "face_detected": 0,
            "left_ear": np.nan,
            "right_ear": np.nan,
            "mean_ear": np.nan,
            "ear_difference": np.nan,
            "mouth_open_ratio": np.nan,
        }

        if not result.face_landmarks:
            rows.append(
                base_row
            )
            continue

        detected_frames += 1

        landmarks = (
            result
            .face_landmarks[0]
        )

        # ----------------------------------------------------
        # FACE BOUNDING BOX
        # ----------------------------------------------------

        all_face_points = [
            normalized_to_pixels(
                landmark,
                width,
                height,
            )
            for landmark
            in landmarks
        ]

        face_bbox = (
            bbox_from_points(
                all_face_points,
                width,
                height,
                margin=0.12,
            )
        )

        # ----------------------------------------------------
        # EYES
        # ----------------------------------------------------

        left_ear_points = (
            landmark_points(
                landmarks,
                LEFT_EYE_EAR,
                width,
                height,
            )
        )

        right_ear_points = (
            landmark_points(
                landmarks,
                RIGHT_EYE_EAR,
                width,
                height,
            )
        )

        left_ear = calculate_ear(
            left_ear_points
        )

        right_ear = calculate_ear(
            right_ear_points
        )

        mean_ear = np.nanmean(
            [
                left_ear,
                right_ear,
            ]
        )

        ear_difference = abs(
            left_ear
            - right_ear
        )

        eye_points = (
            landmark_points(
                landmarks,
                LEFT_EYE_REGION,
                width,
                height,
            )
            + landmark_points(
                landmarks,
                RIGHT_EYE_REGION,
                width,
                height,
            )
        )

        eye_bbox = (
            bbox_from_points(
                eye_points,
                width,
                height,
                margin=0.30,
            )
        )

        # ----------------------------------------------------
        # MOUTH
        # ----------------------------------------------------

        mouth_points = (
            landmark_points(
                landmarks,
                MOUTH_REGION,
                width,
                height,
            )
        )

        mouth_bbox = (
            bbox_from_points(
                mouth_points,
                width,
                height,
                margin=0.30,
            )
        )

        upper_lip = (
            normalized_to_pixels(
                landmarks[13],
                width,
                height,
            )
        )

        lower_lip = (
            normalized_to_pixels(
                landmarks[14],
                width,
                height,
            )
        )

        left_mouth = (
            normalized_to_pixels(
                landmarks[61],
                width,
                height,
            )
        )

        right_mouth = (
            normalized_to_pixels(
                landmarks[291],
                width,
                height,
            )
        )

        mouth_width = euclidean(
            left_mouth,
            right_mouth,
        )

        if mouth_width > 1e-6:

            mouth_open_ratio = (
                euclidean(
                    upper_lip,
                    lower_lip,
                )
                / mouth_width
            )

        else:
            mouth_open_ratio = (
                np.nan
            )

        # ----------------------------------------------------
        # CROPS
        # ----------------------------------------------------

        face_crop = crop_from_bbox(
            image_bgr,
            face_bbox,
        )

        mouth_crop = crop_from_bbox(
            image_bgr,
            mouth_bbox,
        )

        eye_crop = crop_from_bbox(
            image_bgr,
            eye_bbox,
        )

        if face_crop is not None:

            cv2.imwrite(
                str(
                    face_dir
                    / frame_path.name
                ),
                face_crop,
            )

        if mouth_crop is not None:

            cv2.imwrite(
                str(
                    mouth_dir
                    / frame_path.name
                ),
                mouth_crop,
            )

        if eye_crop is not None:

            cv2.imwrite(
                str(
                    eye_dir
                    / frame_path.name
                ),
                eye_crop,
            )

        # ----------------------------------------------------
        # QA IMAGE
        # ----------------------------------------------------

        if not qa_saved:

            qa = (
                image_bgr.copy()
            )

            draw_bbox(
                qa,
                face_bbox,
                (0, 255, 0),
                "FACE",
            )

            draw_bbox(
                qa,
                eye_bbox,
                (255, 0, 0),
                "EYES",
            )

            draw_bbox(
                qa,
                mouth_bbox,
                (0, 0, 255),
                "MOUTH",
            )

            cv2.putText(
                qa,
                (
                    f"EAR={mean_ear:.3f}"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(
                    QA_ROOT
                    / f"{sample_id}_qa.jpg"
                ),
                qa,
            )

            qa_saved = True

        # ----------------------------------------------------
        # FEATURE ROW
        # ----------------------------------------------------

        rows.append(
            {
                **base_row,

                "face_detected": 1,

                "left_ear": (
                    left_ear
                ),

                "right_ear": (
                    right_ear
                ),

                "mean_ear": (
                    mean_ear
                ),

                "ear_difference": (
                    ear_difference
                ),

                "mouth_open_ratio": (
                    mouth_open_ratio
                ),

                "face_x1": (
                    face_bbox[0]
                ),

                "face_y1": (
                    face_bbox[1]
                ),

                "face_x2": (
                    face_bbox[2]
                ),

                "face_y2": (
                    face_bbox[3]
                ),

                "mouth_x1": (
                    mouth_bbox[0]
                ),

                "mouth_y1": (
                    mouth_bbox[1]
                ),

                "mouth_x2": (
                    mouth_bbox[2]
                ),

                "mouth_y2": (
                    mouth_bbox[3]
                ),

                "eyes_x1": (
                    eye_bbox[0]
                ),

                "eyes_y1": (
                    eye_bbox[1]
                ),

                "eyes_x2": (
                    eye_bbox[2]
                ),

                "eyes_y2": (
                    eye_bbox[3]
                ),
            }
        )

    feature_df = pd.DataFrame(
        rows
    )

    feature_path = (
        FEATURE_ROOT
        / f"{sample_id}.csv"
    )

    feature_df.to_csv(
        feature_path,
        index=False,
    )

    total_frames = len(
        frame_paths
    )

    detection_rate = (
        detected_frames
        / total_frames
        if total_frames
        else 0
    )

    return {
        "sample_id": sample_id,
        "total_frames": total_frames,
        "detected_frames": (
            detected_frames
        ),
        "failed_frames": (
            total_frames
            - detected_frames
        ),
        "detection_rate": (
            detection_rate
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"MediaPipe model missing:\n"
            f"{MODEL_PATH}"
        )

    sample_dirs = sorted(
        [
            p
            for p in FRAME_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    if args.limit is not None:

        sample_dirs = (
            sample_dirs[
                :args.limit
            ]
        )

    print("=" * 80)
    print("FACE / MOUTH / EYE EXTRACTION")
    print("=" * 80)

    print(
        f"Samples: "
        f"{len(sample_dirs)}"
    )

    print(
        f"FPS: "
        f"{args.fps}"
    )

    print()

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    FaceLandmarker = (
        mp.tasks.vision.FaceLandmarker
    )

    FaceLandmarkerOptions = (
        mp.tasks.vision
        .FaceLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = (
        FaceLandmarkerOptions(
            base_options=(
                BaseOptions(
                    model_asset_path=(
                        str(
                            MODEL_PATH
                        )
                    )
                )
            ),

            running_mode=(
                RunningMode.IMAGE
            ),

            num_faces=1,

            min_face_detection_confidence=0.5,

            min_face_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_face_blendshapes=False,

            output_facial_transformation_matrixes=False,
        )
    )

    summaries = []

    with (
        FaceLandmarker
        .create_from_options(
            options
        )
    ) as landmarker:

        for i, sample_dir in enumerate(
            sample_dirs,
            start=1,
        ):

            print(
                f"[{i}/{len(sample_dirs)}] "
                f"{sample_dir.name}"
            )

            summary = process_sample(
                sample_dir=sample_dir,
                landmarker=landmarker,
                fps=args.fps,
            )

            summaries.append(
                summary
            )

            print(
                f"  detected: "
                f"{summary['detected_frames']}"
                f"/"
                f"{summary['total_frames']}"
            )

            print(
                f"  detection rate: "
                f"{summary['detection_rate']:.2%}"
            )

            print()

    summary_df = pd.DataFrame(
        summaries
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Overall detection rate:",
        (
            summary_df[
                "detected_frames"
            ].sum()
            /
            summary_df[
                "total_frames"
            ].sum()
        ),
    )

    print()

    print(
        f"Saved summary:\n"
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()