"""Copy local photos, run an acne detector, and prepare a box-labeling pilot."""

from __future__ import annotations

import csv
import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "photo_inputs"
DEFAULT_OUTPUT = ROOT / "data" / "photo_annotation_pilot"
OUTPUT = DEFAULT_OUTPUT
IMAGE_DIR = OUTPUT / "images"
OVERLAY_DIR = OUTPUT / "overlays"
DEFAULT_MODEL_PATH = ROOT / "runs" / "detect" / "acne04_yolo11n_fold0" / "weights" / "best.pt"
CONFIDENCE = 0.10
NMS_IOU = 0.50
IMAGE_SIZE = 1024


def draw_predictions(image_path: Path, boxes: list[tuple[int, float, float, float, float, float]]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size = max(18, min(image.size) // 75)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    line_width = max(3, min(image.size) // 350)
    for box_id, x1, y1, x2, y2, _confidence in boxes:
        draw.ellipse((x1, y1, x2, y2), outline="#00B8D9", width=line_width)
        label = str(box_id)
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        pad = max(4, font_size // 6)
        tag_y = max(0, y1 - height - pad * 2)
        draw.rounded_rectangle(
            (x1, tag_y, x1 + width + 2 * pad, tag_y + height + 2 * pad),
            radius=pad,
            fill="#102A43",
            outline="#FFFFFF",
            width=max(1, line_width // 2),
        )
        draw.text((x1 + pad, tag_y + pad - bounds[1]), label, fill="white", font=font)
    image.save(OVERLAY_DIR / f"{image_path.stem}_boxes.jpg", quality=95)


def main() -> None:
    global OUTPUT, IMAGE_DIR, OVERLAY_DIR
    parser = argparse.ArgumentParser(description="Run an acne detector on local photos and prepare a box-labeling pilot.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Folder of JPG/PNG photos to process")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="New output folder; existing folders are not overwritten")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Local Ultralytics detection checkpoint")
    args = parser.parse_args()
    source_dir = args.source_dir if args.source_dir.is_absolute() else (ROOT / args.source_dir).resolve()
    OUTPUT = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    model_path = args.model if args.model.is_absolute() else (ROOT / args.model).resolve()
    IMAGE_DIR = OUTPUT / "images"
    OVERLAY_DIR = OUTPUT / "overlays"

    if not model_path.is_file():
        raise FileNotFoundError(f"Detection checkpoint not found: {model_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source photo folder not found: {source_dir}")
    source_photos = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not source_photos:
        raise FileNotFoundError(f"No JPG/PNG photos found in: {source_dir}")
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT}")

    IMAGE_DIR.mkdir(parents=True)
    OVERLAY_DIR.mkdir(parents=True)
    copied_images: list[Path] = []
    image_manifest: list[dict[str, str | int]] = []
    for index, source in enumerate(source_photos, start=1):
        target = IMAGE_DIR / f"photo_{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied_images.append(target)
        with Image.open(target) as copied:
            width, height = copied.size
        image_manifest.append(
            {
                "image": target.name,
                "source_filename": source.name,
                "width": width,
                "height": height,
            }
        )

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in copied_images],
        conf=CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        save=False,
        verbose=False,
    )

    annotation_rows: list[dict[str, str | int | float]] = []
    summary_rows: list[dict[str, str | int | float]] = []
    for image_path, result in zip(copied_images, results):
        detections: list[tuple[int, float, float, float, float, float]] = []
        boxes = result.boxes
        if boxes is not None:
            for box_id, box in enumerate(boxes, start=1):
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                xc, yc, width, height = (float(value) for value in box.xywhn[0].tolist())
                confidence = float(box.conf.item())
                source_class_id = int(box.cls.item())
                detections.append((box_id, x1, y1, x2, y2, confidence))
                annotation_rows.append(
                    {
                        "image": image_path.name,
                        "source_filename_grade": "",
                        "box_id": box_id,
                        "source_class_id": source_class_id,
                        "x_center_norm": xc,
                        "y_center_norm": yc,
                        "width_norm": width,
                        "height_norm": height,
                        "confidence": confidence,
                        "box_severity_grade": "",
                        "review_note": "",
                    }
                )
        draw_predictions(image_path, detections)
        mean_confidence = sum(item[5] for item in detections) / len(detections) if detections else 0.0
        summary_rows.append(
            {
                "image": image_path.name,
                "predicted_boxes": len(detections),
                "mean_confidence": round(mean_confidence, 4),
                "model": model_path.name,
                "confidence_cutoff": CONFIDENCE,
            }
        )

    headers = [
        "image", "source_filename_grade", "box_id", "source_class_id", "x_center_norm",
        "y_center_norm", "width_norm", "height_norm", "confidence", "box_severity_grade", "review_note",
    ]
    with (OUTPUT / "annotations.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(annotation_rows)
    with (OUTPUT / "prediction_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "predicted_boxes", "mean_confidence", "model", "confidence_cutoff"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUTPUT / "image_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "source_filename", "width", "height"])
        writer.writeheader()
        writer.writerows(image_manifest)
    readme = (
        "# 本地照片逐框标注试点\n\n"
        f"图片数：{len(copied_images)}。检测框数：{len(annotation_rows)}。\n\n"
        f"使用本地检测器 `{model_path.name}`，置信度阈值 {CONFIDENCE:.2f}、NMS IoU {NMS_IOU:.2f}、输入尺寸 {IMAGE_SIZE}。模型只提出痤疮候选框；框的 0*–IV 严重程度由标注者逐框判断，不能把检测置信度当成严重程度。\n\n"
        "查看 `overlays/` 中带编号的框。打开 `annotations.csv` 对应的标注窗口后，点击框或编号并选择 0*–IV；进度会自动保存。误检可删除并撤销，漏检可在窗口中拖动补框。\n"
    )
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Images: {len(copied_images)}")
    for row in summary_rows:
        print(f"{row['image']}: {row['predicted_boxes']} boxes; mean confidence={row['mean_confidence']:.3f}")
    print(f"Total boxes: {len(annotation_rows)}")
    print(f"Results: {OUTPUT}")


if __name__ == "__main__":
    main()
