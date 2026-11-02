"""Reusable face alignment and conservative lesion-match proposals."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


FACE_LANDMARK_COUNT_MIN = 468
MAX_ALIGNMENT_RESIDUAL = 0.09
SIDE_ALIGNMENT_RANSAC_THRESHOLD = 0.15
MIN_SIDE_ALIGNMENT_INLIER_RATIO = 0.55
LEFT_EYE = (33, 133)
RIGHT_EYE = (263, 362)
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHIN = 152
LEFT_JAWLINE = (234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152)
RIGHT_JAWLINE = (454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152)


@dataclass
class Box:
    image_id: str
    image_name: str
    box_id: str
    grade: str
    cx: float
    cy: float
    width: float
    height: float
    confidence: str = ""


@dataclass
class Photo:
    row: dict[str, str]
    path: Path
    width: int
    height: int
    landmarks: np.ndarray | None = None
    anchors: np.ndarray | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    eye_distance: float | None = None
    face_width: float | None = None
    failure: str = ""
    boxes: list[Box] = field(default_factory=list)
    alignment_info: dict[str, Any] = field(default_factory=dict)


def landmark_point(landmarks: np.ndarray, indices: tuple[int, ...] | int) -> np.ndarray:
    if isinstance(indices, int):
        return landmarks[indices, :2]
    return landmarks[list(indices), :2].mean(axis=0)


def estimate_pose(photo: Photo) -> None:
    if photo.landmarks is None or len(photo.landmarks) < FACE_LANDMARK_COUNT_MIN:
        photo.failure = "insufficient_face_landmarks"
        return
    lm = photo.landmarks
    image_points = np.array(
        [lm[NOSE_TIP, :2], lm[CHIN, :2], lm[33, :2], lm[263, :2], lm[MOUTH_LEFT, :2], lm[MOUTH_RIGHT, :2]],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )
    focal = float(max(photo.width, photo.height))
    camera = np.array(
        [[focal, 0.0, photo.width / 2.0], [0.0, focal, photo.height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rotation_vector, _translation = cv2.solvePnP(
        model_points,
        image_points,
        camera,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        photo.failure = "head_pose_estimation_failed"
        return
    rotation, _ = cv2.Rodrigues(rotation_vector)
    neutral_camera_rotation = np.diag([1.0, -1.0, -1.0])
    relative_rotation = rotation @ neutral_camera_rotation.T
    angles = cv2.RQDecomp3x3(relative_rotation)[0]
    photo.pitch, photo.yaw, photo.roll = (float(value) for value in angles)

    left_eye = landmark_point(lm, LEFT_EYE)
    right_eye = landmark_point(lm, RIGHT_EYE)
    photo.eye_distance = float(np.linalg.norm(left_eye - right_eye))
    face_outline = lm[[10, 152, 234, 454, 93, 323], :2]
    photo.face_width = float(np.ptp(face_outline[:, 0]))
    if photo.eye_distance < 8 or photo.face_width < 20:
        photo.failure = "visible_face_too_small"
        return
    photo.anchors = np.array(
        [
            left_eye,
            right_eye,
            landmark_point(lm, NOSE_TIP),
            (landmark_point(lm, MOUTH_LEFT) + landmark_point(lm, MOUTH_RIGHT)) / 2.0,
            landmark_point(lm, CHIN),
        ],
        dtype=np.float32,
    )


def run_face_landmarker(photos: list[Photo], model_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Face Landmarker model not found: {model_path}. "
            "Download the MediaPipe Face Landmarker task model and pass it with --landmarker."
        )
    runtime_model_path = model_path
    if os.name == "nt" and not str(model_path).isascii():
        runtime_model_path = Path(tempfile.gettempdir()) / "acne_face_landmarker.task"
        if not runtime_model_path.is_file() or runtime_model_path.stat().st_size != model_path.stat().st_size:
            shutil.copyfile(model_path, runtime_model_path)
        if not str(runtime_model_path).isascii():
            raise OSError(f"MediaPipe cannot use non-ASCII model path: {runtime_model_path}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(runtime_model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for photo in photos:
            with Image.open(photo.path) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            original_h, original_w = rgb.shape[:2]
            best_landmarks = None
            best_area = -1.0
            for quarter_turns in range(4):
                rotated = np.ascontiguousarray(np.rot90(rgb, k=quarter_turns))
                rotated_h, rotated_w = rotated.shape[:2]
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rotated)
                result = landmarker.detect(mp_image)
                if not result.face_landmarks:
                    continue
                mapped = []
                for point in result.face_landmarks[0]:
                    x_rot, y_rot = point.x * rotated_w, point.y * rotated_h
                    if quarter_turns == 0:
                        x_orig, y_orig = x_rot, y_rot
                    elif quarter_turns == 1:
                        x_orig, y_orig = original_w - y_rot, x_rot
                    elif quarter_turns == 2:
                        x_orig, y_orig = original_w - x_rot, original_h - y_rot
                    else:
                        x_orig, y_orig = y_rot, original_h - x_rot
                    mapped.append((x_orig, y_orig, point.z * original_w))
                mapped_array = np.asarray(mapped, dtype=np.float32)
                face_box = mapped_array[[10, 152, 234, 454], :2]
                area = float(np.ptp(face_box[:, 0]) * np.ptp(face_box[:, 1]))
                if area > best_area:
                    best_area = area
                    best_landmarks = mapped_array
            if best_landmarks is None:
                photo.failure = "face_not_detected"
                continue
            photo.landmarks = best_landmarks
            estimate_pose(photo)


def estimate_alignment(
    source: Photo,
    target: Photo,
    force_target_mirror: bool | None = None,
    side_view: bool = False,
) -> tuple[np.ndarray | None, float, bool]:
    if source.anchors is None or target.anchors is None:
        return None, math.inf, False

    if side_view:
        source_points, source_roles = side_alignment_points(source)
        target_points, target_roles = side_alignment_points(target)
        if source_points is None or target_points is None:
            return None, math.inf, False
        if len(source_points) != len(target_points) or not np.array_equal(source_roles, target_roles):
            return None, math.inf, False
        valid = (
            (source_points[:, 0] >= 0.015 * source.width)
            & (source_points[:, 0] <= 0.985 * source.width)
            & (source_points[:, 1] >= 0.015 * source.height)
            & (source_points[:, 1] <= 0.985 * source.height)
            & (target_points[:, 0] >= 0.015 * target.width)
            & (target_points[:, 0] <= 0.985 * target.width)
            & (target_points[:, 1] >= 0.015 * target.height)
            & (target_points[:, 1] <= 0.985 * target.height)
            & np.isfinite(source_points).all(axis=1)
            & np.isfinite(target_points).all(axis=1)
        )
        source_points = source_points[valid]
        target_points = target_points[valid]
        roles = source_roles[valid]
        jawline_mask = roles == "jawline"
        if len(source_points) < 8 or int(jawline_mask.sum()) < 6:
            source.alignment_info = {
                "anchor_strategy": "near_eye_and_near_jawline",
                "anchor_count": int(len(source_points)),
                "jawline_anchor_count": int(jawline_mask.sum()),
                "inlier_count": 0,
                "inlier_fraction": 0.0,
                "jawline_inlier_count": 0,
                "quality_gate_passed": False,
            }
            return None, math.inf, False

        mirror_options = (force_target_mirror,) if force_target_mirror is not None else (False, True)
        fits: list[tuple[tuple[int, int, int, float], np.ndarray, float, bool, dict[str, Any]]] = []
        for mirrored in mirror_options:
            target_fit_points = target_points.copy()
            if mirrored:
                target_fit_points[:, 0] = target.width - target_fit_points[:, 0]
            matrix, inliers = cv2.estimateAffinePartial2D(
                source_points,
                target_fit_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=max(3.0, SIDE_ALIGNMENT_RANSAC_THRESHOLD * float(target.eye_distance or 1.0)),
                maxIters=3000,
                confidence=0.99,
                refineIters=10,
            )
            if matrix is None or inliers is None:
                continue
            inlier_mask = inliers.reshape(-1).astype(bool)
            projected = cv2.transform(source_points[None, :, :], matrix)[0]
            errors = np.linalg.norm(projected - target_fit_points, axis=1)
            residual_norm = float(np.median(errors) / max(target.eye_distance or 1.0, 1.0))
            inlier_count = int(inlier_mask.sum())
            jawline_inlier_count = int((inlier_mask & jawline_mask).sum())
            min_inliers = max(7, math.ceil(MIN_SIDE_ALIGNMENT_INLIER_RATIO * len(source_points)))
            min_jawline_inliers = min(6, int(jawline_mask.sum()))
            passed = (
                residual_norm <= MAX_ALIGNMENT_RESIDUAL
                and inlier_count >= min_inliers
                and jawline_inlier_count >= min_jawline_inliers
            )
            info = {
                "anchor_strategy": "near_eye_and_near_jawline",
                "anchor_count": int(len(source_points)),
                "jawline_anchor_count": int(jawline_mask.sum()),
                "inlier_count": inlier_count,
                "inlier_fraction": round(inlier_count / len(source_points), 3),
                "jawline_inlier_count": jawline_inlier_count,
                "quality_gate_passed": bool(passed),
            }
            rank = (int(passed), jawline_inlier_count, inlier_count, -residual_norm)
            fits.append((rank, matrix.astype(np.float64), residual_norm, bool(mirrored), info))
        if not fits:
            return None, math.inf, False
        _rank, best_matrix, best_residual, best_mirrored, best_info = max(fits, key=lambda item: item[0])
        source.alignment_info = best_info
        return best_matrix, best_residual, best_mirrored

    best_matrix = None
    best_residual = math.inf
    best_mirrored = False
    mirror_options = (force_target_mirror,) if force_target_mirror is not None else (False, True)
    for mirrored in mirror_options:
        target_anchors = target.anchors.copy()
        if mirrored:
            target_anchors[:, 0] = target.width - target_anchors[:, 0]
            target_anchors[[0, 1]] = target_anchors[[1, 0]]
        matrix, _inliers = cv2.estimateAffinePartial2D(
            source.anchors,
            target_anchors,
            method=cv2.LMEDS,
            refineIters=10,
        )
        if matrix is None:
            continue
        projected = cv2.transform(source.anchors[None, :, :], matrix)[0]
        residual = float(np.mean(np.linalg.norm(projected - target_anchors, axis=1)))
        residual_norm = residual / max(target.eye_distance or 1.0, 1.0)
        if residual_norm < best_residual:
            best_matrix = matrix.astype(np.float64)
            best_residual = residual_norm
            best_mirrored = mirrored
    return best_matrix, best_residual, best_mirrored


def side_alignment_points(photo: Photo) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Use the camera-facing eye, mouth corner, and jaw contour for cheek views."""
    if photo.landmarks is None or len(photo.landmarks) < FACE_LANDMARK_COUNT_MIN:
        return None, None
    lm = photo.landmarks
    eye_groups = (LEFT_EYE, RIGHT_EYE)
    near_eye = min(eye_groups, key=lambda group: float(lm[list(group), 2].mean()))
    jaw_groups = (LEFT_JAWLINE, RIGHT_JAWLINE)
    near_jaw = min(jaw_groups, key=lambda group: float(lm[list(group), 2].mean()))
    mouth_corner = MOUTH_LEFT if near_jaw == LEFT_JAWLINE else MOUTH_RIGHT
    eye_center = landmark_point(lm, near_eye)
    points = np.vstack((eye_center, lm[mouth_corner, :2], lm[list(near_jaw), :2])).astype(np.float32)
    roles = np.asarray(("eye", "mouth", *("jawline" for _ in near_jaw)), dtype=object)
    return points, roles


def boxes_for_alignment(photo: Photo, matrix: np.ndarray | None = None, mirror_x: bool = False) -> list[dict[str, Any]]:
    result = []
    for box in photo.boxes:
        cx = box.cx * photo.width
        cy = box.cy * photo.height
        if mirror_x:
            cx = photo.width - cx
        if matrix is not None:
            mapped = matrix @ np.array([cx, cy, 1.0], dtype=np.float64)
            cx, cy = float(mapped[0]), float(mapped[1])
        result.append({"box": box, "cx": cx, "cy": cy})
    return result


def lesion_patch_descriptor(image: np.ndarray, cx: float, cy: float, width: float, height: float) -> np.ndarray:
    """Describe local lesion texture after face alignment and contrast normalization."""
    side = max(40, int(round(max(width, height) * 3.2)))
    half = side / 2.0
    x1, y1 = int(math.floor(cx - half)), int(math.floor(cy - half))
    x2, y2 = x1 + side, y1 + side
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - image.shape[1]), max(0, y2 - image.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        # HOG (1764) + low frequency (64) + HSV histogram (72).
        return np.zeros(1900, dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    hue_sat_hist = cv2.calcHist([hsv], [0, 1], None, [18, 4], [0, 180, 24, 256])
    hue_sat_hist = cv2.normalize(hue_sat_hist, None, norm_type=cv2.NORM_L2).reshape(-1)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(gray).reshape(-1)
    low_frequency = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    low_frequency = low_frequency - float(low_frequency.mean())
    low_frequency /= max(float(np.linalg.norm(low_frequency)), 1e-6)
    hog = hog.astype(np.float32)
    hog /= max(float(np.linalg.norm(hog)), 1e-6)
    # Give lesion color a useful role while retaining texture and shape cues.
    descriptor = np.concatenate((hog * 0.72, low_frequency * 0.22, hue_sat_hist.astype(np.float32) * 0.48))
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-6)
    return descriptor


def match_image_pair(source: Photo, target: Photo, matrix: np.ndarray, target_mirrored: bool) -> list[dict[str, Any]]:
    source_boxes = boxes_for_alignment(source, matrix)
    target_boxes = boxes_for_alignment(target, mirror_x=target_mirrored)
    if not source_boxes or not target_boxes:
        return []

    eye_distance = max(target.eye_distance or 1.0, 1.0)
    face_width = max(target.face_width or eye_distance, eye_distance)
    affine_scale = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
    invalid_cost = 1e6
    cost = np.full((len(source_boxes), len(target_boxes)), invalid_cost, dtype=np.float64)
    pair_info: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    with Image.open(source.path) as opened:
        source_image = np.asarray(opened.convert("RGB"))
    with Image.open(target.path) as opened:
        target_image = np.asarray(opened.convert("RGB"))
    aligned_source = cv2.warpAffine(
        source_image, matrix, (target.width, target.height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    aligned_target = cv2.flip(target_image, 1) if target_mirrored else target_image
    source_descriptors = []
    for item in source_boxes:
        box = item["box"]
        source_descriptors.append(lesion_patch_descriptor(
            aligned_source, item["cx"], item["cy"],
            box.width * source.width * affine_scale,
            box.height * source.height * affine_scale,
        ))
    target_descriptors = [
        lesion_patch_descriptor(
            aligned_target, item["cx"], item["cy"],
            item["box"].width * target.width,
            item["box"].height * target.height,
        )
        for item in target_boxes
    ]
    for i, source_item in enumerate(source_boxes):
        source_box: Box = source_item["box"]
        mapped_width = source_box.width * source.width * affine_scale
        mapped_height = source_box.height * source.height * affine_scale
        source_size = math.sqrt(max(mapped_width * mapped_height, 1.0))
        for j, target_item in enumerate(target_boxes):
            target_box: Box = target_item["box"]
            distance = float(np.hypot(source_item["cx"] - target_item["cx"], source_item["cy"] - target_item["cy"]))
            distance_norm = distance / face_width
            target_size = math.sqrt(max(target_box.width * target.width * target_box.height * target.height, 1.0))
            size_ratio = max(source_size, target_size) / max(min(source_size, target_size), 1.0)
            if distance_norm > 0.075 or size_ratio > 3.5:
                continue
            size_delta = abs(math.log(max(source_size, 1.0) / max(target_size, 1.0)))
            appearance_similarity = float(np.clip(np.dot(source_descriptors[i], target_descriptors[j]), 0.0, 1.0))
            if appearance_similarity < 0.30:
                continue
            # Visual evidence is the main signal. Geometry only limits the search
            # neighborhood; uncertain or ambiguous pairs are left unmatched.
            grade_order = {grade: index for index, grade in enumerate(("I", "II", "III", "IV"))}
            grade_delta = abs(grade_order.get(source_box.grade, 0) - grade_order.get(target_box.grade, 0)) / 3.0
            cost[i, j] = (
                0.55 * (1.0 - appearance_similarity)
                + 0.28 * min(distance_norm / 0.075, 1.0)
                + 0.12 * min(size_delta / math.log(3.5), 1.0)
                + 0.05 * grade_delta
            )
            pair_info[(i, j)] = (distance_norm, size_ratio, size_delta, appearance_similarity)

    # Keep only pairs that are a clearly better visual/geometric choice from
    # both sides. This intentionally prefers an unmatched box over an arbitrary
    # one-to-one assignment in dense acne regions.
    ambiguity_margin = 0.025
    raw_cost = cost.copy()
    rejected_pairs: list[tuple[int, int]] = []
    for i in range(len(source_boxes)):
        valid = np.flatnonzero(raw_cost[i] < invalid_cost)
        for j in valid:
            alternatives = [raw_cost[i, other] for other in valid if other != j]
            source_margin = (min(alternatives) - raw_cost[i, j]) if alternatives else 1.0
            column_valid = np.flatnonzero(raw_cost[:, j] < invalid_cost)
            column_alternatives = [raw_cost[other, j] for other in column_valid if other != i]
            target_margin = (min(column_alternatives) - raw_cost[i, j]) if column_alternatives else 1.0
            if source_margin < ambiguity_margin or target_margin < ambiguity_margin:
                rejected_pairs.append((i, int(j)))
    for i, j in rejected_pairs:
        cost[i, j] = invalid_cost

    try:
        from scipy.optimize import linear_sum_assignment

        # Dummy rows/columns let the optimizer leave either box unmatched.
        source_count, target_count = cost.shape
        augmented = np.full((source_count + target_count, source_count + target_count), invalid_cost, dtype=np.float64)
        augmented[:source_count, :target_count] = cost
        unmatched_cost = 0.30
        for i in range(source_count):
            augmented[i, target_count + i] = unmatched_cost
        for j in range(target_count):
            augmented[source_count + j, j] = unmatched_cost
        augmented[source_count:, target_count:] = 0.0
        rows, columns = linear_sum_assignment(augmented)
        pairs = [(int(i), int(j)) for i, j in zip(rows, columns)
                 if i < source_count and j < target_count and cost[i, j] < 2 * unmatched_cost]
        source_indices = [pair[0] for pair in pairs]
        target_indices = [pair[1] for pair in pairs]
    except Exception:
        pairs = sorted(
            ((cost[i, j], i, j) for i in range(len(source_boxes)) for j in range(len(target_boxes))),
            key=lambda item: item[0],
        )
        source_indices, target_indices = [], []
        used_source: set[int] = set()
        used_target: set[int] = set()
        for value, i, j in pairs:
            if value >= 0.60 or i in used_source or j in used_target:
                continue
            source_indices.append(i)
            target_indices.append(j)
            used_source.add(i)
            used_target.add(j)

    results = []
    for source_index, target_index in zip(source_indices, target_indices):
        info = pair_info.get((int(source_index), int(target_index)))
        if info is None:
            continue
        distance_norm, size_ratio, size_delta, appearance_similarity = info
        source_box = source_boxes[int(source_index)]["box"]
        target_box = target_boxes[int(target_index)]["box"]
        match_cost = float(cost[int(source_index), int(target_index)])
        rank_score = 100.0 * max(0.0, 1.0 - match_cost)
        results.append(
            {
                "source_box": source_box,
                "target_box": target_box,
                "distance_norm_face_width": distance_norm,
                "box_size_ratio": size_ratio,
                "appearance_similarity": appearance_similarity,
                "match_cost": match_cost,
                "rank_score": rank_score,
                "source_center_x_px": source_boxes[int(source_index)]["cx"],
                "source_center_y_px": source_boxes[int(source_index)]["cy"],
                "target_center_x_px": target_boxes[int(target_index)]["cx"],
                "target_center_y_px": target_boxes[int(target_index)]["cy"],
            }
        )
    return sorted(results, key=lambda row: row["rank_score"], reverse=True)
