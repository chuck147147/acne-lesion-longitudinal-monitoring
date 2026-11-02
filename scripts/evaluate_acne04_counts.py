"""Evaluate per-image lesion counts on ACNE04's official fold-0 test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "acne04_yolo"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "acne04_yolo11n_fold0" / "weights" / "best.pt"
GRADE_RE = re.compile(r"^levle([0-3])_", re.IGNORECASE)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    error = y_pred.astype(float) - y_true.astype(float)
    absolute = np.abs(error)
    return {
        "images": int(len(y_true)),
        "mae_count": float(absolute.mean()),
        "rmse_count": float(np.sqrt(np.square(error).mean())),
        "bias_pred_minus_true": float(error.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.percentile(absolute, 90)),
        "within_one_lesion_fraction": float((absolute <= 1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--thresholds", default="0.25,0.5")
    parser.add_argument("--preview-threshold", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.weights}")
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is not available; use --device cpu or the project CUDA environment.")

    thresholds = sorted({float(v.strip()) for v in args.thresholds.split(",") if v.strip()})
    if not thresholds or any(v <= 0 or v >= 1 for v in thresholds):
        raise ValueError("--thresholds must be comma-separated values between 0 and 1")
    if args.preview_threshold not in thresholds:
        raise ValueError("--preview-threshold must also appear in --thresholds")

    image_dir = DATA_ROOT / "images" / args.split
    label_dir = DATA_ROOT / "labels" / args.split
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {image_dir}")

    truth: dict[str, int] = {}
    grades: dict[str, str] = {}
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        truth[image_path.name] = sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
        match = GRADE_RE.match(image_path.name)
        grades[image_path.name] = match.group(1) if match else "unknown"

    model = YOLO(str(args.weights))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf_floor,
        verbose=False,
        stream=True,
    )
    rows: list[dict[str, str | int | float]] = []
    for result in results:
        image_name = Path(result.path).name
        confidences = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.array([])
        row: dict[str, str | int | float] = {
            "filename": image_name,
            "grade_from_filename": grades[image_name],
            "ground_truth_count": truth[image_name],
        }
        for threshold in thresholds:
            count = int((confidences >= threshold).sum())
            row[f"pred_count_conf_{threshold:.2f}"] = count
            row[f"error_conf_{threshold:.2f}"] = count - truth[image_name]
        rows.append(row)

    output_dir = ROOT / "runs" / "detect" / "acne04_yolo11n_fold0"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"count_predictions_{args.split}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    y_true = np.array([int(row["ground_truth_count"]) for row in rows])
    summary = {
        "weights": str(args.weights.resolve()),
        "split": f"official ACNE04 fold 0 {args.split}",
        "images": len(rows),
        "confidence_thresholds": {},
        "note": "Choose thresholds on validation data only; do not tune on the official test split.",
    }
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        y_pred = np.array([int(row[f"pred_count_conf_{threshold:.2f}"]) for row in rows])
        summary["confidence_thresholds"][key] = metrics(y_true, y_pred)
    (output_dir / f"count_metrics_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Create a small visual audit with ground-truth boxes in green and predictions in red.
    rng = random.Random(43)
    sample_paths = image_paths[:]
    rng.shuffle(sample_paths)
    sample_paths = sample_paths[:16]
    rows_by_name = {str(row["filename"]): row for row in rows}
    cell = 320
    canvas = np.full((cell * 4, cell * 4, 3), 245, dtype=np.uint8)
    for i, image_path in enumerate(sample_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min((cell - 16) / w, (cell - 36) / h)
        nw, nh = int(w * scale), int(h * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        x0, y0 = (cell - nw) // 2, (cell - 24 - nh) // 2 + 24

        label_path = label_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, xc, yc, bw, bh = map(float, line.split())
            p1 = (int((xc - bw / 2) * nw), int((yc - bh / 2) * nh))
            p2 = (int((xc + bw / 2) * nw), int((yc + bh / 2) * nh))
            cv2.rectangle(image, p1, p2, (0, 220, 0), max(1, int(scale * 3)))

        row = rows_by_name[image_path.name]
        pred_count = int(row[f"pred_count_conf_{args.preview_threshold:.2f}"])
        # Predictions for the audit are re-created from saved results only for selected images.
        prediction = model.predict(str(image_path), imgsz=args.imgsz, device=args.device, conf=args.preview_threshold, verbose=False)[0]
        if prediction.boxes is not None:
            for box in prediction.boxes.xyxy.detach().cpu().numpy():
                p1 = (int(box[0] * scale), int(box[1] * scale))
                p2 = (int(box[2] * scale), int(box[3] * scale))
                cv2.rectangle(image, p1, p2, (0, 0, 255), max(1, int(scale * 3)))

        gx, gy = (i % 4) * cell, (i // 4) * cell
        canvas[gy : gy + 24, gx : gx + cell] = (230, 230, 230)
        cv2.putText(canvas, image_path.name, (gx + 5, gy + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        canvas[gy + y0 : gy + y0 + nh, gx + x0 : gx + x0 + nw] = image
        caption = f"GT {truth[image_path.name]} / Pred {pred_count} @{args.preview_threshold:.2f}"
        cv2.putText(canvas, caption, (gx + 6, gy + cell - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)

    preview = ROOT / "reports" / f"acne04_{args.split}_prediction_audit.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview), canvas)
    print(json.dumps(summary, indent=2))
    print(f"Predictions: {output_dir / f'count_predictions_{args.split}.csv'}")
    print(f"Visual audit: {preview}")


if __name__ == "__main__":
    main()
