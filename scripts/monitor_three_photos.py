"""Infer, annotate, compare, and summarize three same-view acne photos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from monitor_history import write_history_files
from lesion_track_store import assign_persistent_ids, load_existing_track_ids, save_track_store, write_run_track_files


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR = ROOT / "runs" / "detect" / "acne04_yolo11n_fold0" / "weights" / "best.pt"
DEFAULT_CLASSIFIER = ROOT / "runs" / "classify" / "acne_box_severity_yolo11n_v2_finetune" / "weights" / "best.pt"
DEFAULT_LANDMARKER = ROOT / "weights" / "face_landmarker.task"
REGIONS = ("左侧视角", "右侧视角", "下巴视角")
GRADES = ("0*", "I", "II", "III", "IV")
ACTIVE_GRADES = ("I", "II", "III", "IV")
GRADE_WEIGHT = {"0*": 0, "I": 1, "II": 2, "III": 2, "IV": 4}
COLORS = {
    "0*": (145, 145, 145),
    "I": (54, 170, 95),
    "II": (58, 145, 225),
    "III": (238, 143, 48),
    "IV": (220, 62, 72),
}
# Refine boxes above the 95th percentile of the source detector's training-box area.
LARGE_BOX_AREA_FRAC = 0.0045
LARGE_BOX_REFINEMENT_SCALE = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", nargs=3, type=Path, metavar=("LEFT", "RIGHT", "CHIN"),
                        help="本次三张图，顺序为左侧视角、右侧视角、下巴视角")
    parser.add_argument("--history-only", action="store_true",
                        help="汇总已保存的监测结果并生成长期趋势，不运行检测或分级模型")
    baseline_args = parser.add_mutually_exclusive_group()
    baseline_args.add_argument("--previous", nargs=3, type=Path, metavar=("LEFT", "RIGHT", "CHIN"),
                               help="旧式输入：上次原图（会重新检测）；建议使用 --baseline-run")
    baseline_args.add_argument("--baseline-run", type=Path,
                               help="上次已复核的结果目录；直接复用其中框和等级，不重跑痘痘检测/分级")
    parser.add_argument("--regions", nargs=3, default=REGIONS, metavar=("R1", "R2", "R3"),
                        help="三个视角/区域的名称")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu or 0")
    parser.add_argument("--det-conf", type=float, default=0.10)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--review-boxes", action="store_true",
        help="Open the local editor to add missed boxes, remove false detections, and correct grades before comparison",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def photo_name(photo: dict[str, Any]) -> str:
    return str(photo.get("source_filename") or photo["path"].name)


def load_reviewed_baseline(run_dir: Path, regions: list[str]) -> list[dict[str, Any]]:
    """Load the last reviewed session from a prior run without rerunning acne models."""
    result_path = run_dir / "result.json"
    lesions_path = run_dir / "lesions.csv"
    if not result_path.is_file() or not lesions_path.is_file():
        raise FileNotFoundError(f"基准目录缺少 result.json 或 lesions.csv：{run_dir}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("box_review_applied"):
        raise ValueError(f"所选基准尚未完成人工标注复核：{run_dir}")
    image_paths = result.get("current_images", [])
    saved_regions = result.get("regions", [])
    if len(image_paths) != 3 or len(saved_regions) != 3:
        raise ValueError(f"基准结果未保存完整的三视角信息：{run_dir}")
    path_by_region = {region: Path(path) for region, path in zip(saved_regions, image_paths)}
    if any(region not in path_by_region for region in regions):
        raise ValueError("基准结果的区域名称与本次设置不一致")

    with lesions_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    output: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        source_path = resolve_path(path_by_region[region])
        image_path = source_path
        if not image_path.is_file():
            clean_copy = run_dir / "detection_review" / "images" / f"current_{index + 1:02d}.jpg"
            if not clean_copy.is_file():
                raise FileNotFoundError(f"基准原图和复核副本都不存在：{source_path}")
            image_path = clean_copy.resolve()
        with Image.open(image_path) as opened:
            original = opened.convert("RGB")
        width, height = original.size
        photo_rows = [row for row in rows if row.get("session") == "current" and row.get("region") == region]
        detections: list[dict[str, Any]] = []
        for row in photo_rows:
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            grade = row["grade"]
            detections.append({
                **row,
                "session": "previous",
                "image": str(source_path),
                "grade": grade,
                "grade_weight": int(float(row.get("grade_weight") or GRADE_WEIGHT.get(grade, 0))),
                "grade_confidence": float(row.get("grade_confidence") or 0),
                "detector_confidence": float(row.get("detector_confidence") or 0),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                "width": x2 - x1, "height": y2 - y1,
                "cx_norm": (x1 + x2) / 2.0 / width,
                "cy_norm": (y1 + y2) / 2.0 / height,
                "width_norm": (x2 - x1) / width,
                "height_norm": (y2 - y1) / height,
                "candidate_track_id": "",
                "temporal_status": "",
            })
        output.append({
            "photo_id": f"P{index + 1}", "session": "previous", "region": region,
            "path": image_path, "source_path": source_path, "source_filename": source_path.name,
            "original": original, "detections": detections,
        })
    return output


def order_paths_by_named_region(paths: list[Path]) -> list[Path] | None:
    """Canonicalize a three-view set when filenames explicitly name each view."""
    if len(paths) != 3:
        return None
    found: dict[int, Path] = {}
    for path in paths:
        name = path.stem.lower().replace("-", "_").replace(" ", "_")
        if "左脸" in name or "left_face" in name or "left_cheek" in name:
            index = 0
        elif "右脸" in name or "right_face" in name or "right_cheek" in name:
            index = 1
        elif "下巴" in name or "chin" in name or "jaw" in name:
            index = 2
        else:
            return None
        if index in found:
            return None
        found[index] = path
    return [found[index] for index in range(3)] if set(found) == {0, 1, 2} else None


def grade_names(classifier: YOLO) -> dict[int, str]:
    names = classifier.names
    if isinstance(names, list):
        items = enumerate(names)
    else:
        items = names.items()
    parsed: dict[int, str] = {}
    for index, name in items:
        text = str(name).strip().lower().replace("-", "_")
        if "0star" in text or "0*" in text or text in {"0", "mark", "scar"}:
            grade = "0*"
        else:
            match = re.search(r"(?:grade_)?(iv|iii|ii|i)$", text)
            if not match:
                raise ValueError(f"无法识别分类器类别名 {name!r}，预期 0* / I / II / III / IV")
            grade = match.group(1).upper()
        parsed[int(index)] = grade
    if set(parsed.values()) != set(GRADES):
        raise ValueError(f"分类器类别必须包含 {GRADES}，实际为 {parsed}")
    return parsed


def crop_for_classifier(image: Image.Image, xyxy: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = xyxy
    side = max(96.0, 3.0 * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    left = max(0, int(round(cx - half)))
    top = max(0, int(round(cy - half)))
    right = min(width, int(round(cx + half)))
    bottom = min(height, int(round(cy + half)))
    return image.crop((left, top, right, bottom))


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def suppress_duplicate_candidates(
    candidates: list[dict[str, Any]], threshold: float = 0.70,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if any(box_iou(candidate["coords"], existing["coords"]) >= threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def refine_large_detection_boxes(
    detector: YOLO,
    image: Image.Image,
    detections: list[tuple[tuple[float, float, float, float], float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Recheck unusually large boxes in a zoomed crop and split when smaller lesions emerge."""
    image_width, image_height = image.size
    image_area = max(1.0, float(image_width * image_height))
    refined_requests: list[dict[str, Any]] = []
    base_candidates: list[dict[str, Any]] = []

    for index, (coords, confidence) in enumerate(detections):
        x1, y1, x2, y2 = coords
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / image_area < LARGE_BOX_AREA_FRAC:
            base_candidates.append({"coords": coords, "confidence": confidence, "source": "full_frame"})
            continue

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = min(
            min(image_width, image_height),
            max(224, int(round(max(x2 - x1, y2 - y1) * LARGE_BOX_REFINEMENT_SCALE))),
        )
        left = max(0, min(image_width - side, int(round(cx - side / 2))))
        top = max(0, min(image_height - side, int(round(cy - side / 2))))
        right, bottom = left + side, top + side
        refined_requests.append({
            "index": index,
            "parent": coords,
            "parent_area": area,
            "crop": image.crop((left, top, right, bottom)),
            "offset": (left, top),
            "confidence": confidence,
        })

    refined_by_parent: dict[int, list[dict[str, Any]]] = {}
    if refined_requests:
        results = detector.predict(
            source=[request["crop"] for request in refined_requests],
            conf=args.det_conf,
            iou=args.det_iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        for result, request in zip(results, refined_requests):
            boxes = result.boxes
            if boxes is None:
                continue
            left, top = request["offset"]
            parent_x1, parent_y1, parent_x2, parent_y2 = request["parent"]
            pad_x = (parent_x2 - parent_x1) * 0.15
            pad_y = (parent_y2 - parent_y1) * 0.15
            for local_box, confidence in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                child = (
                    max(0.0, float(local_box[0]) + left),
                    max(0.0, float(local_box[1]) + top),
                    min(float(image_width), float(local_box[2]) + left),
                    min(float(image_height), float(local_box[3]) + top),
                )
                child_area = max(0.0, child[2] - child[0]) * max(0.0, child[3] - child[1])
                child_cx, child_cy = (child[0] + child[2]) / 2.0, (child[1] + child[3]) / 2.0
                if child_area >= request["parent_area"] * 0.65:
                    continue
                if not (
                    parent_x1 - pad_x <= child_cx <= parent_x2 + pad_x
                    and parent_y1 - pad_y <= child_cy <= parent_y2 + pad_y
                ):
                    continue
                refined_by_parent.setdefault(request["index"], []).append({
                    "coords": child,
                    "confidence": float(confidence),
                    "source": "zoom_refined",
                })

    output: list[dict[str, Any]] = list(base_candidates)
    split_candidates: list[dict[str, Any]] = []
    for index, (coords, confidence) in enumerate(detections):
        x1, y1, x2, y2 = coords
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / image_area < LARGE_BOX_AREA_FRAC:
            continue
        pad_x = (x2 - x1) * 0.15
        pad_y = (y2 - y1) * 0.15
        for other_index, (other_coords, other_confidence) in enumerate(detections):
            if other_index == index:
                continue
            other_area = max(0.0, other_coords[2] - other_coords[0]) * max(0.0, other_coords[3] - other_coords[1])
            other_cx = (other_coords[0] + other_coords[2]) / 2.0
            other_cy = (other_coords[1] + other_coords[3]) / 2.0
            if (
                other_area < area * 0.65
                and other_area / image_area < LARGE_BOX_AREA_FRAC
                and x1 - pad_x <= other_cx <= x2 + pad_x
                and y1 - pad_y <= other_cy <= y2 + pad_y
            ):
                refined_by_parent.setdefault(index, []).append({
                    "coords": other_coords,
                    "confidence": other_confidence,
                    "source": "full_frame_inside_large",
                })
        children = suppress_duplicate_candidates(refined_by_parent.get(index, []), threshold=0.85)
        replace_parent = len(children) >= 2 or (
            len(children) == 1
            and max(0.0, children[0]["coords"][2] - children[0]["coords"][0])
            * max(0.0, children[0]["coords"][3] - children[0]["coords"][1]) <= area * 0.40
        )
        if replace_parent:
            split_candidates.extend(children)
            for base in base_candidates:
                base_cx = (base["coords"][0] + base["coords"][2]) / 2.0
                base_cy = (base["coords"][1] + base["coords"][3]) / 2.0
                base_area = max(0.0, base["coords"][2] - base["coords"][0]) * max(0.0, base["coords"][3] - base["coords"][1])
                if (
                    base_area < area * 0.65
                    and x1 - pad_x <= base_cx <= x2 + pad_x
                    and y1 - pad_y <= base_cy <= y2 + pad_y
                ):
                    base["source"] = "large_region_split"
        else:
            output.append({"coords": coords, "confidence": confidence, "source": "large_box_needs_review"})

    split_candidates = suppress_duplicate_candidates(split_candidates, threshold=0.85)
    new_split_candidates = [
        child for child in split_candidates
        if all(box_iou(child["coords"], base["coords"]) < 0.85 for base in base_candidates)
    ]
    output.extend({**child, "source": "large_region_split"} for child in new_split_candidates)
    return sorted(output, key=lambda item: (item["coords"][1], item["coords"][0]))


def infer_batch(
    paths: list[Path], regions: list[str], detector: YOLO, classifier: YOLO,
    class_map: dict[int, str], args: argparse.Namespace, session: str,
) -> list[dict[str, Any]]:
    detections_by_photo: list[list[dict[str, Any]]] = [[] for _ in paths]
    originals: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as opened:
            originals.append(opened.convert("RGB"))

    predictions = detector.predict(
        source=[str(path) for path in paths], conf=args.det_conf, iou=args.det_iou,
        imgsz=args.imgsz, device=args.device, verbose=False,
    )
    crops: list[Image.Image] = []
    pending: list[tuple[int, int, tuple[float, float, float, float], float, str]] = []
    for photo_index, result in enumerate(predictions):
        boxes = result.boxes
        if boxes is None:
            continue
        raw_detections = [
            (tuple(float(value) for value in xyxy), float(confidence))
            for xyxy, confidence in zip(boxes.xyxy.tolist(), boxes.conf.tolist())
        ]
        final_detections = refine_large_detection_boxes(detector, originals[photo_index], raw_detections, args)
        for candidate in final_detections:
            coords = candidate["coords"]
            confidence = candidate["confidence"]
            crop = crop_for_classifier(originals[photo_index], coords)
            detection_index = len(detections_by_photo[photo_index])
            detections_by_photo[photo_index].append({})
            pending.append((photo_index, detection_index, coords, float(confidence), candidate["source"]))
            crops.append(crop)

    if crops:
        classified = classifier.predict(source=crops, imgsz=224, batch=32, device=args.device, verbose=False)
        if len(classified) != len(pending):
            raise RuntimeError(f"框分类数 {len(classified)} 与候选框数 {len(pending)} 不一致")
        for result, (photo_index, detection_index, coords, det_conf, detection_source) in zip(classified, pending):
            grade_index = int(result.probs.top1)
            grade = class_map[grade_index]
            grade_conf = float(result.probs.top1conf)
            x1, y1, x2, y2 = coords
            width, height = originals[photo_index].size
            item = {
                "session": session,
                "region": regions[photo_index],
                "image": str(paths[photo_index]),
                "lesion_id": f"{session[0].upper()}{photo_index + 1}-{detection_index + 1:03d}",
                "grade": grade,
                "grade_weight": GRADE_WEIGHT[grade],
                "grade_confidence": grade_conf,
                "grade_source": "model",
                "review_note": "",
                "detector_confidence": det_conf,
                "detection_source": detection_source,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "width": x2 - x1,
                "height": y2 - y1,
                "cx_norm": ((x1 + x2) / 2.0) / width,
                "cy_norm": ((y1 + y2) / 2.0) / height,
                "width_norm": (x2 - x1) / width,
                "height_norm": (y2 - y1) / height,
                "candidate_track_id": "",
                "temporal_status": "no_previous_session",
            }
            detections_by_photo[photo_index][detection_index] = item

    output = []
    for index, path in enumerate(paths):
        photo_id = f"{session[0].upper()}{index + 1}"
        output.append({
            "photo_id": photo_id,
            "session": session,
            "region": regions[index],
            "path": path,
            "original": originals[index],
            "detections": detections_by_photo[index],
        })
    return output


def get_font(size: int) -> ImageFont.ImageFont:
    for font_path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_annotated(photo: dict[str, Any], id_by_lesion: dict[str, str] | None = None,
                   transform: Any = None, output_size: tuple[int, int] | None = None,
                   mirrored: bool = False) -> Image.Image:
    image = photo["original"].copy()
    if mirrored:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if output_size is not None and transform is not None:
        warped = cv2.warpAffine(np.asarray(image), transform, output_size, flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        image = Image.fromarray(warped)
    draw = ImageDraw.Draw(image)
    font = get_font(max(14, round(min(image.size) / 58)))
    sx = 1.0 if transform is not None else image.width / photo["original"].width
    sy = 1.0 if transform is not None else image.height / photo["original"].height
    if mirrored:
        sx = image.width / photo["original"].width
    for detection in photo["detections"]:
        cx, cy = detection["cx"], detection["cy"]
        bw, bh = detection["width"], detection["height"]
        if transform is not None:
            mapped = transform @ [cx, cy, 1.0]
            cx, cy = float(mapped[0]), float(mapped[1])
            scale = math.sqrt(abs(float(transform[0, 0] * transform[1, 1] - transform[0, 1] * transform[1, 0])))
            bw, bh = bw * scale, bh * scale
        elif mirrored:
            cx = photo["original"].width - cx
        cx, cy, bw, bh = cx * sx, cy * sy, bw * sx, bh * sy
        radius = max(8, round(max(bw, bh) / 2))
        color = COLORS[detection["grade"]]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=max(3, radius // 8))
        label = (id_by_lesion or {}).get(detection["lesion_id"], detection["lesion_id"])
        confidence_label = (
            "人工" if detection.get("grade_source") == "manual"
            else "待核" if detection.get("grade_source") == "model_uncertain"
            else f"{detection['grade_confidence']:.2f}"
        )
        label = f"{label} {detection['grade']} {confidence_label}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 6
        tx = max(0, min(image.width - tw, cx - radius))
        ty = cy - radius - th if cy - radius >= th else min(image.height - th, cy + radius)
        draw.rounded_rectangle((tx, ty, tx + tw, ty + th), radius=4, fill=color)
        draw.text((tx + 5, ty + 2), label, font=font, fill="white")
    return image


def counts_for(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(item["grade"] for item in detections)
    return {
        "comedone_I": counts["I"],
        "papule_II": counts["II"],
        "pustule_III": counts["III"],
        "nodule_IV": counts["IV"],
        "marks_0star": counts["0*"],
        "active_total": sum(counts[grade] for grade in ACTIVE_GRADES),
        "inflammatory_total": counts["II"] + counts["III"] + counts["IV"],
        "burden_points": sum(GRADE_WEIGHT[item["grade"]] for item in detections),
    }


def run_correspondence(previous: list[dict[str, Any]], current: list[dict[str, Any]],
                       landmark_path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return location-based candidates only; these are never called confirmed tracks."""
    from face_tracking_core import (
        Box, Photo, estimate_alignment, match_image_pair, run_face_landmarker,
        MAX_ALIGNMENT_RESIDUAL,
    )

    def make_photo(record: dict[str, Any], session: str) -> Any:
        pil_image = record["original"]
        photo = Photo(
            row={"image_id": record["photo_id"], "session_id": session, "capture_date": ""},
            path=record["path"], width=pil_image.width, height=pil_image.height,
        )
        for item in record["detections"]:
            if item["grade"] not in ACTIVE_GRADES:
                continue
            photo.boxes.append(Box(
                image_id=record["photo_id"], image_name=record["path"].name,
                box_id=item["lesion_id"], grade=item["grade"],
                cx=item["cx_norm"], cy=item["cy_norm"],
                width=item["width_norm"], height=item["height_norm"],
                confidence=f"{item['detector_confidence']:.4f}",
            ))
        return photo

    old_track_photos = [make_photo(item, "previous") for item in previous]
    new_track_photos = [make_photo(item, "current") for item in current]
    track_photos = old_track_photos + new_track_photos
    errors: dict[int, dict[str, Any]] = {}
    if landmark_path.is_file():
        try:
            run_face_landmarker(track_photos, landmark_path)
        except Exception as error:
            for i in range(3):
                errors[i] = {
                    "comparable": False, "reason": f"人脸关键点运行失败：{error}",
                    "source_photo": old_track_photos[i], "target_photo": new_track_photos[i],
                }
    else:
        for i in range(3):
            errors[i] = {
                "comparable": False, "reason": f"关键点模型不存在：{landmark_path}",
                "source_photo": old_track_photos[i], "target_photo": new_track_photos[i],
            }

    candidates: list[dict[str, Any]] = []
    for index, (source, target) in enumerate(zip(old_track_photos, new_track_photos)):
        if index in errors:
            continue
        if source.failure or target.failure:
            failure_labels = {
                "insufficient_face_landmarks": "关键点数量不足",
                "head_pose_estimation_failed": "头部姿态估计失败",
                "visible_face_too_small": "可见人脸区域过小",
                "face_not_detected": "未检测到人脸",
            }
            errors[index] = {
                "comparable": False,
                "reason": "; ".join(failure_labels.get(value, value) for value in (source.failure, target.failure) if value),
                "source_photo": source, "target_photo": target,
            }
            continue
        side_view = index in (0, 1)
        matrix, residual, mirrored = estimate_alignment(source, target, side_view=side_view)
        quality = source.alignment_info
        quality_failed = side_view and not quality.get("quality_gate_passed", False)
        if matrix is None or residual > MAX_ALIGNMENT_RESIDUAL or quality_failed:
            if side_view:
                reason = (
                    "侧脸锚点拟合质量不足："
                    f"下颌线内点 {quality.get('jawline_inlier_count', 0)}/{quality.get('jawline_anchor_count', 0)}，"
                    f"总内点 {quality.get('inlier_count', 0)}/{quality.get('anchor_count', 0)}，"
                    f"中位残差 {residual:.3f}（门槛 {MAX_ALIGNMENT_RESIDUAL:.3f}）"
                )
            else:
                reason = f"配准残差 {residual:.3f} > {MAX_ALIGNMENT_RESIDUAL:.3f}"
            errors[index] = {
                "comparable": False, "reason": reason, **quality,
                "source_photo": source, "target_photo": target,
            }
            continue
        matches = match_image_pair(source, target, matrix, mirrored)
        for match in matches:
            old_box, new_box = match["source_box"], match["target_box"]
            candidates.append({
                "region_index": index,
                "region": current[index]["region"],
                "previous_lesion_id": old_box.box_id,
                "current_lesion_id": new_box.box_id,
                "previous_grade": old_box.grade,
                "current_grade": new_box.grade,
                "previous_weight": GRADE_WEIGHT[old_box.grade],
                "current_weight": GRADE_WEIGHT[new_box.grade],
                "weight_delta": GRADE_WEIGHT[new_box.grade] - GRADE_WEIGHT[old_box.grade],
                "distance_norm_face_width": match["distance_norm_face_width"],
                "size_ratio": match["box_size_ratio"],
                "appearance_similarity": match["appearance_similarity"],
                "match_cost": match["match_cost"],
                "rank_score": match["rank_score"],
                "status": "candidate_same_weight" if GRADE_WEIGHT[new_box.grade] == GRADE_WEIGHT[old_box.grade]
                          else "candidate_weight_down" if GRADE_WEIGHT[new_box.grade] < GRADE_WEIGHT[old_box.grade]
                          else "candidate_weight_up",
                "alignment_residual": residual,
                "target_mirrored_for_comparison": mirrored,
            })
        errors[index] = {
            "comparable": True, "reason": "", "matrix": matrix,
            "residual": residual, "target_mirrored": mirrored, **quality,
            "source_photo": source, "target_photo": target,
        }

    # IDs are assigned after all three views are processed, for stable report ordering.
    candidates.sort(key=lambda row: (row["region_index"], row["previous_lesion_id"], row["current_lesion_id"]))
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"CAND-{index:03d}"
    return candidates, errors


def markdown_report(previous: list[dict[str, Any]] | None, current: list[dict[str, Any]],
                    candidates: list[dict[str, Any]], alignments: dict[int, dict[str, Any]],
                    output_dir: Path, longitudinal_session_count: int = 0,
                    tracking_summary: dict[str, Any] | None = None) -> str:
    region_rows = []
    for index, now in enumerate(current):
        after = counts_for(now["detections"])
        before = counts_for(previous[index]["detections"]) if previous else None
        region_rows.append((now["region"], before, after))

    lines = [
        "# 三视角痤疮模型监测结果", "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "![三视角对比总览](overview.png)", "",
        f"## 长期趋势\n\n累计 {longitudinal_session_count} 个去重时点。趋势图统计每个时点经复核的区域负担分和活动框数；逐颗编号由自动匹配候选跨相邻时点延续。自动对应可能受光线、姿势和取景影响。\n\n![长期随访趋势](longitudinal.png)", "",
        f"框拆分复核：本次拆出的小框 {sum(d.get('detection_source') == 'large_region_split' for photo in current for d in photo['detections'])} 个；无法可靠拆分、需人工检查的大框 {sum(d.get('detection_source') == 'large_box_needs_review' for photo in current for d in photo['detections'])} 个。", "",
        "## 区域统计", "",
        "负担分为透明的项目原型分：粉刺 I=1、丘疹 II=2、脓疱 III=2、结节 IV=4、0*=0。0* 只表示痘印/瘢痕候选，不计入活动病灶。该分数尚不是经临床验证的痤疮严重度量表。", "",
        "圈色：0* 灰、I 绿、II 蓝、III 橙、IV 红；圈旁标签依次显示病灶编号、类别、分类置信度。", "",
    ]
    headers = "| 区域 | 上次 I/II/III/IV | 本次 I/II/III/IV | 活动数差 | 炎症数差 | 结节差 | 负担分 上次→本次 | Δ负担分 | 负担变化率 |"
    lines.extend([headers, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for region, before, after in region_rows:
        if before is None:
            before_text = "—"
            active_delta = infl_delta = nodule_delta = burden_delta = "—"
            burden_change = "—"
            burden_pair = f"— → {after['burden_points']}"
        else:
            before_text = f"{before['comedone_I']}/{before['papule_II']}/{before['pustule_III']}/{before['nodule_IV']}"
            active_delta = f"{after['active_total'] - before['active_total']:+}"
            infl_delta = f"{after['inflammatory_total'] - before['inflammatory_total']:+}"
            nodule_delta = f"{after['nodule_IV'] - before['nodule_IV']:+}"
            burden_delta_value = after['burden_points'] - before['burden_points']
            burden_delta = f"{burden_delta_value:+}"
            burden_pair = f"{before['burden_points']} → {after['burden_points']}"
            burden_change = "N/A（上次为 0）" if before['burden_points'] == 0 else f"{(before['burden_points'] - after['burden_points']) / before['burden_points'] * 100:+.1f}%"
        after_text = f"{after['comedone_I']}/{after['papule_II']}/{after['pustule_III']}/{after['nodule_IV']}"
        lines.append(f"| {region} | {before_text} | {after_text} | {active_delta} | {infl_delta} | {nodule_delta} | {burden_pair} | {burden_delta} | {burden_change} |")

    previous_totals = counts_for([item for photo in previous for item in photo["detections"]]) if previous else None
    current_totals = counts_for([item for photo in current for item in photo["detections"]])
    lines.extend(["", "## 整体数值结论", ""])
    if previous_totals is None:
        lines.append(
            f"当前三视角检出活动病灶框 {current_totals['active_total']} 个（I={current_totals['comedone_I']}、II={current_totals['papule_II']}、III={current_totals['pustule_III']}、IV={current_totals['nodule_IV']}），"
            f"炎症性框 {current_totals['inflammatory_total']} 个，结节框 {current_totals['nodule_IV']} 个，原型负担分 {current_totals['burden_points']}。缺少上次三视角基线，无法计算时间变化。"
        )
    else:
        delta = current_totals["burden_points"] - previous_totals["burden_points"]
        change_pct = "N/A（上次总分为 0）" if previous_totals["burden_points"] == 0 else f"{(previous_totals['burden_points'] - current_totals['burden_points']) / previous_totals['burden_points'] * 100:+.1f}%"
        previous_class_counts = (previous_totals['comedone_I'], previous_totals['papule_II'], previous_totals['pustule_III'], previous_totals['nodule_IV'])
        current_class_counts = (current_totals['comedone_I'], current_totals['papule_II'], current_totals['pustule_III'], current_totals['nodule_IV'])
        previous_active = sum(counts_for(photo["detections"])["active_total"] for photo in previous)
        current_active = sum(counts_for(photo["detections"])["active_total"] for photo in current)
        previous_unpaired = sum(
            detection.get("grade") in ACTIVE_GRADES and detection.get("temporal_status") != "track_auto_matched"
            for photo in previous for detection in photo["detections"]
        )
        current_unpaired = sum(
            detection.get("grade") in ACTIVE_GRADES and detection.get("temporal_status") != "track_auto_matched"
            for photo in current for detection in photo["detections"]
        )
        previous_affected_views = sum(counts_for(photo["detections"])["active_total"] > 0 for photo in previous)
        current_affected_views = sum(counts_for(photo["detections"])["active_total"] > 0 for photo in current)
        lines.append(
            f"三视角活动病灶框 {previous_totals['active_total']} → {current_totals['active_total']}（{current_totals['active_total'] - previous_totals['active_total']:+}）；"
            f"炎症性框 {previous_totals['inflammatory_total']} → {current_totals['inflammatory_total']}（{current_totals['inflammatory_total'] - previous_totals['inflammatory_total']:+}）；"
            f"结节框 {previous_totals['nodule_IV']} → {current_totals['nodule_IV']}（{current_totals['nodule_IV'] - previous_totals['nodule_IV']:+}）；"
            f"I/II/III/IV 数量 {previous_class_counts} → {current_class_counts}；"
            f"受累视角数 {previous_affected_views}/3 → {current_affected_views}/3；"
            f"原型负担分 {previous_totals['burden_points']} → {current_totals['burden_points']}（Δ={delta:+}，变化率 {change_pct}）。"
        )
        for index, (region, before, after) in enumerate(region_rows):
            delta_region = after["burden_points"] - before["burden_points"]
            lines.append(f"- {region}：负担分 {before['burden_points']} → {after['burden_points']}（Δ={delta_region:+}），活动框 {before['active_total']} → {after['active_total']}，结节 {before['nodule_IV']} → {after['nodule_IV']}。")
        weight_deltas = [int(row.get("weight_delta", 0) or 0) for row in candidates]
        automatic_links = sum(
            detection.get("temporal_status") == "track_auto_matched"
            for photo in current for detection in photo["detections"]
        )
        lines.append(
            f"- 系统自动延续 track_id {automatic_links} 颗，符合筛选条件的候选 {len(candidates)} 对。"
            f"候选的权重下降/不变/上升分别为 {sum(delta < 0 for delta in weight_deltas)}/"
            f"{sum(delta == 0 for delta in weight_deltas)}/{sum(delta > 0 for delta in weight_deltas)} 对。"
            "相似度只用于排序，不是病灶同一性的概率。"
        )
        lines.append(f"- 未自动匹配活动框：上次 {previous_unpaired} 个，本次 {current_unpaired} 个；保留独立编号，不记作新发或消退。")
        for index, info in sorted(alignments.items()):
            if not info.get("comparable"):
                lines.append(f"- {current[index]['region']}：自动配准未通过，本区不延续逐颗编号（{info.get('reason', '未知')}）。")
            elif info.get("anchor_strategy") == "near_eye_and_near_jawline":
                lines.append(
                    f"- {current[index]['region']}：配准采用近侧眼点和下颌线，"
                    f"稳健残差 {info.get('residual', 0):.3f}，"
                    f"下颌线内点 {info.get('jawline_inlier_count', 0)}/{info.get('jawline_anchor_count', 0)}。"
                )
    if tracking_summary:
        lines.append(
            f"- 本次有稳定编号的病灶 {tracking_summary.get('current_track_count', 0)} 颗；"
            "逐时点记录已追加至本机追踪表。未确认或本次未检出的病灶不会自动判为消退。"
        )
    lines.extend([
        "", "## 病灶对应状态口径", "",
        "自动对应仅对通过配准和一对一相似度筛选的候选延续 `track_id`。光线、姿势和取景变化仍可能造成误配；未匹配病灶会保留独立编号，且不会自动判定为新发或消退。丘疹 II 与脓疱 III 的项目权重相同；两者互换时记作类别变化，不算负担分升降。", "",
        "跨视角可见同一病灶，因此三视角合计是‘检测框合计’，不能当作全脸唯一病灶数。整脸结论必须同时看分区明细、炎症性框和结节数。模型分数与类别是模型输出，不是临床诊断；模型置信度不是发生概率。", "",
        "## 圈划与对比图", "",
    ])
    for index, photo in enumerate(current):
        lines.append(f"### {photo['region']}：{photo_name(photo)}")
        lines.append("")
        lines.append(f"![{photo['region']} 本次圈划结果](current_{index + 1:02d}_annotated.jpg)")
        if previous:
            lines.append("")
            lines.append(f"![{photo['region']} 上次与本次对比](comparison_{index + 1:02d}.jpg)")
        lines.append("")
    lines.extend(["## 输出文件", "", "- `overview.png`：三视角前后对比、分区变化和总体汇总图。", "- `longitudinal.png`、`longitudinal.csv`：跨时点的区域趋势图和逐时点统计。", "- `lesions.csv`：逐病灶检测框、类别、来源、稳定追踪编号和时间状态。", "- `track_observations.csv`：本次新增的逐颗观察记录；`track_summary.csv`：各编号的观察次数和类别变化。", "- `candidate_tracks.csv`：自动对应候选、匹配指标和编号延续状态。", "- `detection_review/`：本次人工补框、删框和等级修正工作副本。", "- `result.json`：机器可读汇总。", ""])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def upright_rotation(feature_photo: Any) -> float | None:
    anchors = getattr(feature_photo, "anchors", None)
    if anchors is None:
        return None
    eyes = (anchors[0] + anchors[1]) / 2.0
    eye_axis = anchors[1] - anchors[0]
    chin_axis = anchors[4] - eyes
    base_angle = math.degrees(math.atan2(float(eye_axis[1]), float(eye_axis[0])))
    options = (base_angle, base_angle + 180.0)
    for angle in options:
        radians = math.radians(angle)
        chin_y = -math.sin(radians) * float(chin_axis[0]) + math.cos(radians) * float(chin_axis[1])
        if chin_y > 0:
            return ((angle + 180.0) % 360.0) - 180.0
    return None


def rotate_photo_record(
    photo: dict[str, Any], feature_photo: Any, fallback_angle: float | None = None,
) -> dict[str, Any]:
    angle = upright_rotation(feature_photo)
    if angle is None:
        angle = fallback_angle if fallback_angle is not None else photo.get("display_rotation_fallback")
    if angle is None or abs(angle) < 0.5:
        return photo
    image = np.asarray(photo["original"])
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    bound_width = int(math.ceil(height * abs(matrix[0, 1]) + width * abs(matrix[0, 0])))
    bound_height = int(math.ceil(height * abs(matrix[0, 0]) + width * abs(matrix[0, 1])))
    matrix[0, 2] += bound_width / 2.0 - width / 2.0
    matrix[1, 2] += bound_height / 2.0 - height / 2.0
    rotated = cv2.warpAffine(
        image, matrix, (bound_width, bound_height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    updated = dict(photo)
    updated["original"] = Image.fromarray(rotated)
    updated_detections = []
    for original in photo["detections"]:
        detection = dict(original)
        cx, cy = float(original["cx"]), float(original["cy"])
        half_w, half_h = float(original["width"]) / 2.0, float(original["height"]) / 2.0
        corners = np.asarray([
            [cx-half_w, cy-half_h], [cx+half_w, cy-half_h],
            [cx+half_w, cy+half_h], [cx-half_w, cy+half_h],
        ], dtype=np.float32)
        transformed = cv2.transform(corners[None, :, :], matrix)[0]
        x1, y1 = transformed.min(axis=0)
        x2, y2 = transformed.max(axis=0)
        detection.update({
            "cx": float((x1+x2)/2), "cy": float((y1+y2)/2),
            "width": float(x2-x1), "height": float(y2-y1),
            "cx_norm": float((x1+x2)/2/bound_width), "cy_norm": float((y1+y2)/2/bound_height),
            "width_norm": float((x2-x1)/bound_width), "height_norm": float((y2-y1)/bound_height),
        })
        updated_detections.append(detection)
    updated["detections"] = updated_detections
    return updated


def create_overview(
    previous: list[dict[str, Any]] | None,
    current: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    alignments: dict[int, dict[str, Any]],
    output_path: Path,
) -> None:
    width, outer, header_h = 2200, 42, 165
    card_h, card_gap = 850, 26
    footer_h = 220
    height = header_h + outer + 3 * card_h + 2 * card_gap + footer_h + outer
    canvas = Image.new("RGB", (width, height), "#E9EEF2")
    draw = ImageDraw.Draw(canvas)
    title_font, subtitle_font = get_font(48), get_font(25)
    card_title_font, panel_font, stat_font = get_font(34), get_font(24), get_font(22)
    draw.text((outer, 24), "三视角痤疮变化总览", font=title_font, fill="#17212B")
    draw.text((outer, 92), "侧脸配准参考近侧眼与下颌线；通过筛选的自动候选会延续稳定编号。误配可能发生；未匹配不等于消退。", font=subtitle_font, fill="#4E5C68")

    for region_index, now in enumerate(current):
        y = header_h + outer + region_index * (card_h + card_gap)
        draw.rounded_rectangle((outer, y, width-outer, y+card_h), radius=22, fill="white", outline="#CBD4DC", width=2)
        old = previous[region_index] if previous else None
        region = now["region"]
        session_label = "上次 vs 本次" if old else "本次基线"
        draw.text((outer+24, y+18), f"{region}　·　{session_label}", font=card_title_font, fill="#17212B")

        region_candidates = [row for row in candidates if row["region_index"] == region_index]
        ids_old = {
            detection["lesion_id"]: str(detection.get("track_id") or (f"?{detection['candidate_track_id']}" if detection.get("candidate_track_id") else detection["lesion_id"]))
            for detection in (old["detections"] if old else [])
        }
        ids_now = {
            detection["lesion_id"]: str(detection.get("track_id") or (f"?{detection['candidate_track_id']}" if detection.get("candidate_track_id") else detection["lesion_id"]))
            for detection in now["detections"]
        }
        info = alignments.get(region_index, {})
        if old is not None:
            feature_old, feature_now = info.get("source_photo"), info.get("target_photo")
            if region_index == 2:
                old_display = rotate_photo_record(old, feature_old)
                now_display = rotate_photo_record(now, feature_now)
                old_img = draw_annotated(old_display, ids_old)
                now_img = draw_annotated(now_display, ids_now)
            elif info.get("comparable"):
                old_img = draw_annotated(
                    old, ids_old, transform=info["matrix"],
                    output_size=(now["original"].width, now["original"].height),
                )
                now_img = draw_annotated(now, ids_now, mirrored=bool(info["target_mirrored"]))
            else:
                old_img, now_img = draw_annotated(old, ids_old), draw_annotated(now, ids_now)
        else:
            old_img = None
            now_img = draw_annotated(now)

        panel_gap = 24
        panel_width = (width - 2*outer - panel_gap) // 2
        image_y, image_h = y+118, 500
        panels = [(old_img, "基准照片"), (now_img, "本次照片")]
        for panel_index, (panel, panel_label) in enumerate(panels):
            panel_x = outer + panel_index * (panel_width + panel_gap)
            if panel is None:
                continue
            draw.text((panel_x+8, y+76), f"{panel_label}　{photo_name(old if panel_index == 0 else now)}", font=panel_font, fill="#344350")
            panel.thumbnail((panel_width-16, image_h), Image.Resampling.LANCZOS)
            paste_x = panel_x + (panel_width-panel.width)//2
            paste_y = image_y + (image_h-panel.height)//2
            canvas.paste(panel, (paste_x, paste_y))

        if old is not None:
            before, after = counts_for(old["detections"]), counts_for(now["detections"])
            stats_y = image_y + image_h + 12
            for panel_index, counts in enumerate((before, after)):
                panel_x = outer + panel_index * (panel_width + panel_gap)
                label = "上次" if panel_index == 0 else "本次"
                draw.text((panel_x+8, stats_y), f"{label}检出　0*/I/II/III/IV：{counts['marks_0star']}/{counts['comedone_I']}/{counts['papule_II']}/{counts['pustule_III']}/{counts['nodule_IV']}", font=stat_font, fill="#263541")
                draw.text((panel_x+8, stats_y+34), f"活动框 {counts['active_total']}　炎症框 {counts['inflammatory_total']}　结节 {counts['nodule_IV']}　负担分 {counts['burden_points']}", font=stat_font, fill="#263541")
            activity_delta = after["active_total"] - before["active_total"]
            infl_delta = after["inflammatory_total"] - before["inflammatory_total"]
            nodule_delta = after["nodule_IV"] - before["nodule_IV"]
            burden_delta = after["burden_points"] - before["burden_points"]
            marks_delta = after["marks_0star"] - before["marks_0star"]
            old_matched = sum(detection.get("grade") in ACTIVE_GRADES and detection.get("temporal_status") == "track_auto_matched" for detection in old["detections"])
            now_matched = sum(detection.get("grade") in ACTIVE_GRADES and detection.get("temporal_status") == "track_auto_matched" for detection in now["detections"])
            old_unmatched = max(0, before["active_total"] - old_matched)
            now_unmatched = max(0, after["active_total"] - now_matched)
            sims = [float(row["appearance_similarity"]) for row in region_candidates]
            mean_sim = f"{sum(sims) / len(sims):.2f}" if sims else "无"
            delta_text = f"本区变化　痘印 {marks_delta:+}　活动框 {activity_delta:+}　炎症框 {infl_delta:+}　结节 {nodule_delta:+}　负担分 {burden_delta:+}"
            draw.text((outer+24, y+card_h-120), delta_text, font=panel_font, fill="#185D8B")
            quality_text = (
                f"未自动匹配 {old_unmatched}/{now_unmatched} 框（上次/本次；不代表新发或消退）"
                f"　·　自动对应 {len(region_candidates)} 对　·　局部外观相似度均值 {mean_sim}"
            )
            if not info.get("comparable"):
                reason = info.get("reason", "配准质量不足")
                quality_text += f"　·　配准未通过：{reason}"
            elif info.get("anchor_strategy") == "near_eye_and_near_jawline":
                quality_text += (
                    f"　·　近侧眼/下颌线内点 "
                    f"{info.get('jawline_inlier_count', 0)}/{info.get('jawline_anchor_count', 0)}"
                )
            while draw.textlength(quality_text, font=stat_font) > width - 2 * outer - 48 and len(quality_text) > 10:
                quality_text = quality_text[:-2] + "…"
            draw.text((outer+24, y+card_h-82), quality_text, font=stat_font, fill="#A34B24" if not info.get("comparable") else "#4E5C68")
        else:
            counts = counts_for(now["detections"])
            draw.text((outer+24, y+card_h-58), f"本次基线：活动框 {counts['active_total']}　炎症框 {counts['inflammatory_total']}　结节 {counts['nodule_IV']}　负担分 {counts['burden_points']}", font=panel_font, fill="#185D8B")

    footer_y = header_h + outer + 3 * card_h + 2 * card_gap + 10
    if previous:
        before_all = counts_for([box for photo in previous for box in photo["detections"]])
        after_all = counts_for([box for photo in current for box in photo["detections"]])
        draw.rounded_rectangle((outer, footer_y, width-outer, footer_y+footer_h), radius=22, fill="#172B3A")
        draw.text((outer+24, footer_y+18), "三视角总体变化（跨视角可能重复看到同一病灶，数量是检测框合计）", font=panel_font, fill="white")
        draw.text((outer+24, footer_y+66),
                  f"活动框 {before_all['active_total']} → {after_all['active_total']}（Δ {after_all['active_total']-before_all['active_total']:+}）　"
                  f"炎症框 {before_all['inflammatory_total']} → {after_all['inflammatory_total']}　"
                  f"结节 {before_all['nodule_IV']} → {after_all['nodule_IV']}　"
                  f"负担分 {before_all['burden_points']} → {after_all['burden_points']}（Δ {after_all['burden_points']-before_all['burden_points']:+}）",
                  font=panel_font, fill="#F1F5F8")
        draw.text((outer+24, footer_y+116),
                  f"0*/I/II/III/IV：{before_all['marks_0star']}/{before_all['comedone_I']}/{before_all['papule_II']}/{before_all['pustule_III']}/{before_all['nodule_IV']} → "
                  f"{after_all['marks_0star']}/{after_all['comedone_I']}/{after_all['papule_II']}/{after_all['pustule_III']}/{after_all['nodule_IV']}　"
                  f"未确认的局部特征候选 {len(candidates)} 对。相似度是排序参考，不是病灶同一性的概率或结论。",
                  font=stat_font, fill="#C8D8E4")
    else:
        draw.rounded_rectangle((outer, footer_y, width-outer, footer_y+footer_h), radius=22, fill="#172B3A")
        totals = counts_for([box for photo in current for box in photo["detections"]])
        draw.text((outer+24, footer_y+36), f"当前基线：活动框 {totals['active_total']}　炎症框 {totals['inflammatory_total']}　结节 {totals['nodule_IV']}　负担分 {totals['burden_points']}", font=panel_font, fill="white")
    canvas.save(output_path)


def review_detection_boxes(
    previous: list[dict[str, Any]] | None,
    current: list[dict[str, Any]],
    classifier: YOLO,
    class_map: dict[int, str],
    output_dir: Path,
    device: str,
) -> None:
    """Open the existing local box editor, then apply add/delete/grade corrections."""
    review_dir = output_dir / "detection_review"
    image_dir = review_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ordered_photos = list(current)
    companions: dict[str, str] = {}
    if previous is not None:
        for index, (old_photo, new_photo) in enumerate(zip(previous, current), start=1):
            old_name, new_name = f"previous_{index:02d}.jpg", f"current_{index:02d}.jpg"
            draw_annotated(old_photo).save(image_dir / old_name, quality=95)
            new_photo["review_image"] = new_name
            companions[new_name] = old_name
    else:
        for index, photo in enumerate(current, start=1):
            photo["review_image"] = f"current_{index:02d}.jpg"

    annotation_rows: list[dict[str, Any]] = []
    suggestion_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for photo in ordered_photos:
        image_name = photo["review_image"]
        photo["original"].save(image_dir / image_name, quality=95)
        image_rows.append({
            "image": image_name,
            "source_filename": photo_name(photo),
            "companion_image": companions.get(image_name, ""),
        })
        for box_number, detection in enumerate(photo["detections"], start=1):
            box_id = str(box_number)
            annotation_rows.append({
                "image": image_name,
                "source_filename_grade": photo["path"].stem,
                "box_id": box_id,
                "source_class_id": "0",
                "x_center_norm": detection["cx_norm"],
                "y_center_norm": detection["cy_norm"],
                "width_norm": detection["width_norm"],
                "height_norm": detection["height_norm"],
                "confidence": detection["detector_confidence"],
                "detection_source": detection.get("detection_source", ""),
                "box_severity_grade": "",
                "review_note": "",
                "lesion_id": detection["lesion_id"],
                "session": detection["session"],
                "region": detection["region"],
            })
            suggestion_rows.append({
                "image": image_name,
                "box_id": box_id,
                "suggested_grade": detection["grade"],
                "top1_confidence": detection["grade_confidence"],
            })

    annotation_path = review_dir / "annotations.csv"
    suggestions_path = review_dir / "grade_suggestions.csv"
    write_csv(annotation_path, annotation_rows, [
        "image", "source_filename_grade", "box_id", "source_class_id", "x_center_norm", "y_center_norm",
        "width_norm", "height_norm", "confidence", "box_severity_grade", "review_note", "lesion_id", "session", "region",
        "detection_source",
    ])
    write_csv(suggestions_path, suggestion_rows, ["image", "box_id", "suggested_grade", "top1_confidence"])
    write_csv(review_dir / "prediction_summary.csv", image_rows, ["image", "source_filename"])
    write_csv(review_dir / "image_manifest.csv", image_rows, ["image", "source_filename", "companion_image"])

    reviewer = ROOT / "scripts" / "annotate_box_severity.py"
    subprocess.run(
        [sys.executable, "-X", "utf8", str(reviewer), "--csv", str(annotation_path), "--suggestions", str(suggestions_path)],
        cwd=ROOT,
        check=True,
    )

    with annotation_path.open("r", newline="", encoding="utf-8-sig") as handle:
        corrected_rows = [dict(row) for row in csv.DictReader(handle)]
    suggestion_by_key = {
        (row["image"], row["box_id"]): row
        for row in suggestion_rows
    }
    original_geometry_by_key = {
        (row["image"], row["box_id"]): tuple(
            float(row[name]) for name in ("x_center_norm", "y_center_norm", "width_norm", "height_norm")
        )
        for row in annotation_rows
    }
    photo_by_name = {photo["review_image"]: photo for photo in ordered_photos}
    manual_crops: list[tuple[dict[str, Any], dict[str, str], tuple[float, float, float, float]]] = []
    for row in corrected_rows:
        key = (row["image"], row["box_id"])
        if row["box_severity_grade"]:
            continue
        geometry = tuple(float(row[name]) for name in ("x_center_norm", "y_center_norm", "width_norm", "height_norm"))
        geometry_changed = key in original_geometry_by_key and any(
            abs(current_value - original_value) > 1e-6
            for current_value, original_value in zip(geometry, original_geometry_by_key[key])
        )
        if key in suggestion_by_key and not geometry_changed:
            continue
        photo = photo_by_name[row["image"]]
        width, height = photo["original"].size
        coords = (
            float(row["x_center_norm"]) * width - float(row["width_norm"]) * width / 2,
            float(row["y_center_norm"]) * height - float(row["height_norm"]) * height / 2,
            float(row["x_center_norm"]) * width + float(row["width_norm"]) * width / 2,
            float(row["y_center_norm"]) * height + float(row["height_norm"]) * height / 2,
        )
        manual_crops.append((photo, row, coords))
    manual_suggestions: dict[tuple[str, str], tuple[str, float]] = {}
    if manual_crops:
        crops = [crop_for_classifier(photo["original"], coords) for photo, _row, coords in manual_crops]
        results = classifier.predict(source=crops, imgsz=224, batch=32, device=device, verbose=False)
        if len(results) != len(manual_crops):
            raise RuntimeError(f"补画框的分类结果 {len(results)} 与新框数 {len(manual_crops)} 不一致")
        for result, (_photo, row, _coords) in zip(results, manual_crops):
            manual_suggestions[(row["image"], row["box_id"])] = (
                class_map[int(result.probs.top1)], float(result.probs.top1conf),
            )
    for (image_name, box_id), (grade, confidence) in manual_suggestions.items():
        suggestion = {
            "image": image_name, "box_id": box_id,
            "suggested_grade": grade, "top1_confidence": confidence,
        }
        suggestion_by_key[(image_name, box_id)] = suggestion
        for index, existing in enumerate(suggestion_rows):
            if (existing["image"], existing["box_id"]) == (image_name, box_id):
                suggestion_rows[index] = suggestion
                break
        else:
            suggestion_rows.append(suggestion)
    write_csv(suggestions_path, suggestion_rows, ["image", "box_id", "suggested_grade", "top1_confidence"])

    detections_by_image: dict[str, list[dict[str, Any]]] = {name: [] for name in photo_by_name}
    for row in corrected_rows:
        photo = photo_by_name[row["image"]]
        width, height = photo["original"].size
        box_w = float(row["width_norm"]) * width
        box_h = float(row["height_norm"]) * height
        cx = float(row["x_center_norm"]) * width
        cy = float(row["y_center_norm"]) * height
        key = (row["image"], row["box_id"])
        suggestion = suggestion_by_key.get(key)
        if suggestion:
            suggested_grade = suggestion["suggested_grade"]
            suggested_confidence = float(suggestion["top1_confidence"])
        else:
            suggested_grade, suggested_confidence = manual_suggestions.get(key, ("II", 0.0))
        manual_grade = row.get("box_severity_grade", "").strip()
        grade = manual_grade or suggested_grade
        lesion_id = row.get("lesion_id", "").strip() or f"{photo['photo_id']}-M{int(row['box_id']):03d}"
        detections_by_image[row["image"]].append({
            "session": photo["session"],
            "region": photo["region"],
            "image": str(photo["path"]),
            "lesion_id": lesion_id,
            "grade": grade,
            "grade_weight": GRADE_WEIGHT[grade],
            "grade_confidence": suggested_confidence,
            "grade_source": "manual" if manual_grade else "model_uncertain" if row.get("review_note", "").strip().lower() == "uncertain" else "model",
            "review_note": row.get("review_note", "").strip(),
            "detector_confidence": float(row.get("confidence") or 0.0),
            "detection_source": row.get("detection_source", "").strip(),
            "x1": cx - box_w / 2, "y1": cy - box_h / 2,
            "x2": cx + box_w / 2, "y2": cy + box_h / 2,
            "cx": cx, "cy": cy, "width": box_w, "height": box_h,
            "cx_norm": float(row["x_center_norm"]), "cy_norm": float(row["y_center_norm"]),
            "width_norm": float(row["width_norm"]), "height_norm": float(row["height_norm"]),
            "candidate_track_id": "", "temporal_status": "no_previous_session",
        })
    for photo in ordered_photos:
        photo["detections"] = detections_by_image[photo["review_image"]]


def main() -> None:
    args = parse_args()
    if args.history_only:
        if args.current or args.previous or args.baseline_run:
            raise ValueError("--history-only 不可与本次照片或基准参数同时使用")
        history_output = resolve_path(args.output_dir) if args.output_dir else ROOT / "outputs" / "acne_monitor_history"
        entries = write_history_files(ROOT / "outputs" / "acne_monitor_runs", history_output)
        print(f"Output: {history_output}")
        print(f"History sessions: {len(entries)}")
        return
    if not args.current:
        raise ValueError("请提供 --current 三张照片；若只查看长期趋势，请使用 --history-only")
    current_paths = [resolve_path(path) for path in args.current]
    previous_paths = [resolve_path(path) for path in args.previous] if args.previous else None
    baseline_run_path = resolve_path(args.baseline_run) if args.baseline_run else None
    current_named_order = order_paths_by_named_region(current_paths)
    previous_named_order = order_paths_by_named_region(previous_paths) if previous_paths else None
    if current_named_order is not None:
        current_paths = current_named_order
    if previous_named_order is not None:
        previous_paths = previous_named_order
    if len(set(current_paths)) != 3:
        raise ValueError("本次输入的三张图片必须是三个不同文件")
    if previous_paths and len(set(previous_paths)) != 3:
        raise ValueError("上次输入的三张图片必须是三个不同文件")
    for path in current_paths + (previous_paths or []):
        if not path.is_file():
            raise FileNotFoundError(path)
    if baseline_run_path is not None and not baseline_run_path.is_dir():
        raise NotADirectoryError(f"基准结果目录不存在：{baseline_run_path}")
    detector_path, classifier_path = resolve_path(args.detector), resolve_path(args.classifier)
    for model_path in (detector_path, classifier_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"模型权重不存在：{model_path}")
    output_dir = resolve_path(args.output_dir) if args.output_dir else ROOT / "outputs" / "acne_monitor_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    detector = YOLO(str(detector_path))
    classifier = YOLO(str(classifier_path))
    class_map = grade_names(classifier)
    current = infer_batch(current_paths, list(args.regions), detector, classifier, class_map, args, "current")
    if baseline_run_path is not None:
        previous = load_reviewed_baseline(baseline_run_path, list(args.regions))
        previous_paths = [photo.get("source_path", photo["path"]) for photo in previous]
    else:
        previous = infer_batch(previous_paths, list(args.regions), detector, classifier, class_map, args, "previous") if previous_paths else None
    if args.review_boxes:
        review_detection_boxes(previous, current, classifier, class_map, output_dir, args.device)

    candidates: list[dict[str, Any]] = []
    alignments: dict[int, dict[str, Any]] = {}
    if previous is not None:
        candidates, alignments = run_correspondence(previous, current, resolve_path(args.landmarker))
    track_store_path = ROOT / "outputs" / "acne_monitor_history" / "lesion_tracks.csv"
    if previous is not None:
        load_existing_track_ids(previous, track_store_path)
        for photo in previous:
            for detection in photo["detections"]:
                detection["candidate_track_id"] = ""
                detection["temporal_status"] = ""
    track_result = assign_persistent_ids(
        previous,
        current,
        candidates,
        track_store_path,
        current_run=output_dir.name,
        previous_run=baseline_run_path.name if baseline_run_path else "",
    )
    if previous:
        for index, pair in enumerate(zip(previous, current)):
            old_photo, new_photo = pair
            info = alignments.get(index, {"comparable": False, "reason": "未运行配准"})
            id_old = {
                detection["lesion_id"]: str(detection.get("track_id") or (f"?{detection['candidate_track_id']}" if detection.get("candidate_track_id") else detection["lesion_id"]))
                for detection in old_photo["detections"]
            }
            id_new = {
                detection["lesion_id"]: str(detection.get("track_id") or (f"?{detection['candidate_track_id']}" if detection.get("candidate_track_id") else detection["lesion_id"]))
                for detection in new_photo["detections"]
            }
            if index == 2:
                old_display = rotate_photo_record(old_photo, info.get("source_photo"))
                new_display = rotate_photo_record(new_photo, info.get("target_photo"))
            else:
                old_display, new_display = old_photo, new_photo
            old_img = draw_annotated(old_display, id_old)
            new_img = draw_annotated(new_display, id_new)
            old_img.save(output_dir / f"previous_{index + 1:02d}_annotated.jpg", quality=94)
            new_img.save(output_dir / f"current_{index + 1:02d}_annotated.jpg", quality=94)
            if info.get("comparable") and index != 2:
                matrix = info["matrix"]
                left = draw_annotated(old_photo, id_old, transform=matrix, output_size=(new_photo["original"].width, new_photo["original"].height))
                right = draw_annotated(new_photo, id_new, mirrored=bool(info["target_mirrored"]))
                anchor_note = (
                    f" · 下颌线内点 {info['jawline_inlier_count']}/{info['jawline_anchor_count']}"
                    if info.get("anchor_strategy") == "near_eye_and_near_jawline" else ""
                )
                alignment_note = f"已配准 · 稳健残差 {info['residual']:.3f}{anchor_note} · 相同 L 编号表示自动匹配延续"
            elif index == 2:
                left, right = old_img, new_img
                alignment_note = f"下巴方向已调整 · {info.get('reason', '候选仅供参考')} · 仅并列比较检测框"
            else:
                left, right = old_img, new_img
                alignment_note = f"未配准 · {info.get('reason', '原因未知')} · 仅并列比较检测框"
            tile_w, tile_h, header_h = 700, 700, 72
            canvas = Image.new("RGB", (tile_w * 2, tile_h + header_h), "white")
            draw = ImageDraw.Draw(canvas)
            font = get_font(22)
            draw.text((14, 8), f"{new_photo['region']}  |  上次：{photo_name(old_photo)}  →  本次：{photo_name(new_photo)}", fill=(20, 20, 20), font=font)
            draw.text((14, 39), alignment_note, fill=(70, 70, 70), font=get_font(17))
            for tile_index, panel in enumerate((left, right)):
                thumb = panel.copy()
                thumb.thumbnail((tile_w - 12, tile_h - 12), Image.Resampling.LANCZOS)
                x = tile_index * tile_w + (tile_w - thumb.width) // 2
                y = header_h + (tile_h - thumb.height) // 2
                canvas.paste(thumb, (x, y))
            canvas.save(output_dir / f"comparison_{index + 1:02d}.jpg", quality=94)
    else:
        for index, photo in enumerate(current):
            draw_annotated(photo).save(output_dir / f"current_{index + 1:02d}_annotated.jpg", quality=94)

    if previous:
        for region_index, photo in enumerate(previous):
            for detection in photo["detections"]:
                if detection.get("temporal_status") != "track_auto_matched":
                    detection["temporal_status"] = "not_seen_this_visit_unmatched"

    lesion_rows = []
    for batch in ([previous] if previous else []) + [current]:
        for photo in batch:
            for detection in photo["detections"]:
                lesion_rows.append({key: detection.get(key, "") for key in (
                    "session", "region", "image", "lesion_id", "grade", "grade_source", "review_note", "grade_weight",
                    "grade_confidence", "detector_confidence", "detection_source", "x1", "y1", "x2", "y2", "track_id",
                    "candidate_track_id", "temporal_status",
                )})
    write_csv(output_dir / "lesions.csv", lesion_rows, [
        "session", "region", "image", "lesion_id", "grade", "grade_source", "review_note", "grade_weight", "grade_confidence",
        "detector_confidence", "detection_source", "x1", "y1", "x2", "y2", "track_id", "candidate_track_id", "temporal_status",
    ])

    summary_rows = []
    for index, photo in enumerate(current):
        after = counts_for(photo["detections"])
        before = counts_for(previous[index]["detections"]) if previous else None
        row = {"region": photo["region"], "current_image": str(photo["path"]), **{f"current_{k}": v for k, v in after.items()}}
        if before is not None:
            row["previous_image"] = str(previous[index]["path"])
            row.update({f"previous_{k}": v for k, v in before.items()})
            for key in ("comedone_I", "papule_II", "pustule_III", "nodule_IV", "marks_0star", "active_total", "inflammatory_total", "burden_points"):
                row[f"delta_{key}"] = after[key] - before[key]
            row["burden_change_percent"] = "" if before["burden_points"] == 0 else round((before["burden_points"] - after["burden_points"]) / before["burden_points"] * 100, 2)
            row["alignment_comparable"] = bool(alignments.get(index, {}).get("comparable", False))
            row["alignment_reason"] = alignments.get(index, {}).get("reason", "")
        summary_rows.append(row)
    fields = list(dict.fromkeys(key for row in summary_rows for key in row))
    write_csv(output_dir / "region_summary.csv", summary_rows, fields)
    candidate_fields = [
        "candidate_id", "region", "previous_lesion_id", "current_lesion_id", "previous_grade", "current_grade",
        "previous_weight", "current_weight", "weight_delta", "distance_norm_face_width", "size_ratio",
        "appearance_similarity", "match_cost", "rank_score", "alignment_residual", "target_mirrored_for_comparison",
        "status", "tracking_status", "track_id",
    ]
    write_csv(output_dir / "candidate_tracks.csv", candidates, candidate_fields)
    create_overview(previous, current, candidates, alignments, output_dir / "overview.png")
    write_run_track_files(track_result, output_dir)

    result_json = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_images": [str(path) for path in current_paths],
        "previous_images": [str(path) for path in previous_paths] if previous_paths else [],
        "baseline_run": str(baseline_run_path) if baseline_run_path else "",
        "baseline_loaded_from_reviewed_results": baseline_run_path is not None,
        "regions": list(args.regions),
        "weight_definition": GRADE_WEIGHT,
        "weight_index_status": "engineering_prototype_not_clinically_validated",
        "detector_weights": str(detector_path),
        "classifier_weights": str(classifier_path),
        "detector_confidence_threshold": args.det_conf,
        "detector_nms_iou": args.det_iou,
        "detector_image_size": args.imgsz,
        "large_box_refinement_area_fraction": LARGE_BOX_AREA_FRAC,
        "large_box_refined_count": sum(d.get("detection_source") == "large_region_split" for photo in current for d in photo["detections"]),
        "large_box_needs_review_count": sum(d.get("detection_source") == "large_box_needs_review" for photo in current for d in photo["detections"]),
        "box_review_applied": bool(args.review_boxes),
        "overview_image": str(output_dir / "overview.png"),
        "current_region_counts": [{"region": photo["region"], **counts_for(photo["detections"])} for photo in current],
        "previous_region_counts": [{"region": photo["region"], **counts_for(photo["detections"])} for photo in previous] if previous else [],
        "candidate_track_count": len(candidates),
        "automatic_track_link_count": track_result["automatic_link_count"],
        "current_track_count": track_result["current_track_count"],
        "track_store": str(track_store_path),
        "track_continuity_requires_manual_confirmation": False,
        "candidate_matching_method": "local_hog_hsv_similarity_and_face_aligned_location; automatic_one_to_one_links_after_alignment_gate",
        "alignment": {current[index]["region"]: {k: v for k, v in info.items() if k not in {"matrix", "source_photo", "target_photo"}} for index, info in alignments.items()},
    }
    history_dir = ROOT / "outputs" / "acne_monitor_history"
    history_entries = write_history_files(
        ROOT / "outputs" / "acne_monitor_runs",
        history_dir,
        extra_current={
            "run_dir": output_dir,
            "images": [str(path) for path in current_paths],
            "counts": result_json["current_region_counts"],
            "reviewed": bool(args.review_boxes),
            "generated_at": result_json["generated_at"],
        },
    )
    for filename in ("longitudinal.png", "longitudinal.csv"):
        (output_dir / filename).write_bytes((history_dir / filename).read_bytes())
    save_track_store(track_result)
    result_json["longitudinal_session_count"] = len(history_entries)
    result_json["longitudinal_image"] = str(output_dir / "longitudinal.png")
    (output_dir / "result.json").write_text(json.dumps(result_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_dir / "report.md").write_text(
        markdown_report(previous, current, candidates, alignments, output_dir, len(history_entries), track_result),
        encoding="utf-8",
    )
    print(f"Output: {output_dir}")
    print(f"Current active detection boxes: {counts_for([x for p in current for x in p['detections']])['active_total']}")
    print(f"Candidate lesion correspondences: {len(candidates)}")
    print(f"Automatic persistent track links: {track_result['automatic_link_count']}")
    print(f"Persistent track IDs in this visit: {track_result['current_track_count']}")


if __name__ == "__main__":
    main()
