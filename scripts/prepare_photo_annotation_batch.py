"""Stage a local image folder and prepare a fresh YOLO box-labeling batch."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "photo_batch_inputs"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "photo_batch"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_photo_annotation_pilot.py"
RENDER_SCRIPT = ROOT / "scripts" / "render_photo_contact_sheet.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="Folder containing the local photos")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Temporary staged input directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="New annotation batch directory")
    parser.add_argument("--model", type=Path, help="Optional local Ultralytics detection checkpoint")
    return parser.parse_args()


def rooted(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    source_dir = rooted(args.source_dir)
    input_dir = rooted(args.input_dir)
    output_dir = rooted(args.output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Photo folder not found: {source_dir}")
    photos = sorted(
        (path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    if not photos:
        raise FileNotFoundError(f"No supported image files found in: {source_dir}")
    if input_dir.exists() or output_dir.exists():
        existing = input_dir if input_dir.exists() else output_dir
        raise FileExistsError(f"Refusing to overwrite existing batch path: {existing}")

    input_dir.mkdir(parents=True)
    staged_rows: list[dict[str, str | int]] = []
    for order, source in enumerate(photos, start=1):
        staged_name = f"{order:04d}__{source.name}"
        shutil.copy2(source, input_dir / staged_name)
        staged_rows.append({
            "attachment_order": order,
            "input_filename": staged_name,
            "source_filename": source.name,
        })
    with (input_dir / "source_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attachment_order", "input_filename", "source_filename"])
        writer.writeheader()
        writer.writerows(staged_rows)

    prepare_command = [
        sys.executable, str(PREPARE_SCRIPT), "--source-dir", str(input_dir), "--output-dir", str(output_dir),
    ]
    if args.model is not None:
        prepare_command.extend(["--model", str(rooted(args.model))])
    subprocess.run(prepare_command, cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), "--batch-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
    )

    source_by_staged_name = {row["input_filename"]: row for row in staged_rows}
    manifest_path = output_dir / "image_manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        prepared_rows = list(csv.DictReader(handle))
    for row in prepared_rows:
        source = source_by_staged_name[row["source_filename"]]
        row["attachment_order"] = str(source["attachment_order"])
        row["source_filename"] = str(source["source_filename"])
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "attachment_order", "source_filename", "width", "height"])
        writer.writeheader()
        writer.writerows(prepared_rows)

    with (output_dir / "prediction_summary.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        detection_count = sum(int(row["predicted_boxes"]) for row in csv.DictReader(handle))
    readme = (
        f"# Local photo annotation batch\n\n"
        f"Images: {len(photos)}. Detector candidate boxes: {detection_count}.\n\n"
        "Review `overlays/` and annotate the boxes in `annotations.csv` with the project's 0*–IV labels. "
        "The detector confidence is not a severity score. Keep the source photos and outputs private unless the subject has explicitly authorized publication.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Prepared {len(photos)} photos at: {output_dir}")
    print(f"Annotations: {output_dir / 'annotations.csv'}")
    print(f"Preview: {output_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
