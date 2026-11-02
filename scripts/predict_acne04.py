"""Run the trained single-class ACNE04 YOLO baseline on one local image."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs" / "detect" / "acne04_yolo11n_fold0" / "weights" / "best.pt"

CONFIDENCE = 0.10
NMS_IOU = 0.50  # Suppress more overlapping candidate boxes to avoid double circles.
IMAGE_SIZE = 1024
DEVICE = "0"  # Use "cpu" if you do not want to use the NVIDIA GPU.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Image file to analyze")
    parser.add_argument("--weights", type=Path, default=MODEL_PATH, help="Trained detector checkpoint")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "detect" / "acne04_predict_circles")
    parser.add_argument("--device", default=DEVICE, help="Ultralytics device, for example cpu or 0")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    model_path = args.weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model weights not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=args.device,
        save=False,
    )

    result = results[0]
    boxes = result.boxes
    annotated = result.orig_img.copy()
    scores: list[float] = []
    for box in boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        score = float(box.conf.item())
        scores.append(score)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max(8, round(max(x2 - x1, y2 - y1) / 2))
        cv2.circle(annotated, center, radius, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_circles.jpg"
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save prediction image: {output_path}")

    print(f"Confidence cutoff: {CONFIDENCE:.2f}")
    print(f"Detected {len(scores)} acne lesion(s); confidence scores are listed below, off-image:")
    for index, score in enumerate(scores, start=1):
        print(f"  {index}. {score:.3f}")
    print(f"Prediction image saved to: {output_path}")
    if os.name == "nt":
        os.startfile(str(output_path))


if __name__ == "__main__":
    main()
