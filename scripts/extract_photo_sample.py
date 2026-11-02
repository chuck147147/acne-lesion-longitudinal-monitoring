"""Extract a small, labeled-aware photo sample from the local acne dataset.

Run this file with the project's selected Python interpreter in VS Code.
The script copies source images (never moves them) into data/sample_inspection,
copies matching YOLO labels when available, and creates a contact sheet/manifest.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it in the selected project interpreter first.") from exc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "sample_inspection"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "runs", "sample_inspection",
    "predictions", "prediction", "outputs", "output", "results", "result",
}
SAMPLE_SIZE = 12
SEED = 20261026


def is_excluded(path: Path) -> bool:
    return any(part.lower() in EXCLUDED_PARTS for part in path.parts)


def collect_images() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not is_excluded(path)
    )


def matching_label(image_path: Path) -> Path | None:
    """Find the conventional YOLO labels/split/name.txt counterpart."""
    parts = list(image_path.parts)
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts):
        if part == "images":
            label_parts = parts.copy()
            label_parts[index] = "labels"
            candidate = Path(*label_parts).with_suffix(".txt")
            if candidate.is_file():
                return candidate
    same_stem = image_path.with_suffix(".txt")
    return same_stem if same_stem.is_file() else None


def label_summary(label_path: Path | None) -> tuple[int, str]:
    if label_path is None:
        return 0, ""
    classes: Counter[str] = Counter()
    rows = 0
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 5:
            rows += 1
            classes[fields[0]] += 1
    class_text = "; ".join(f"class {key}: {count}" for key, count in sorted(classes.items()))
    return rows, class_text


def select_source(images: list[Path]) -> tuple[str, list[Path]]:
    """Prefer the previously prepared ACNE04 YOLO data, if it exists."""
    preferred = [p for p in images if "acne04_yolo" in str(p).lower()]
    if preferred:
        return "data/processed/acne04_yolo", preferred
    labeled = [p for p in images if matching_label(p) is not None]
    if labeled:
        return "local images with matching YOLO labels", labeled
    return "all local image files", images


def make_contact_sheet(rows: list[dict[str, str]]) -> None:
    thumb_w, thumb_h, caption_h = 320, 250, 48
    columns = 3
    count = len(rows)
    row_count = (count + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, row_count * (thumb_h + caption_h)), "#f4f2ee")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for index, row in enumerate(rows):
        image = Image.open(row["sample_path"]).convert("RGB")
        image.thumbnail((thumb_w - 16, thumb_h - 16))
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + caption_h)
        image_x = x + (thumb_w - image.width) // 2
        image_y = y + (thumb_h - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        caption = f"{Path(row['sample_path']).name}\n{row['label_status']} | boxes: {row['box_count']}"
        draw.multiline_text((x + 8, y + thumb_h + 2), caption, fill="#242424", font=font, spacing=3)

    sheet.save(OUTPUT / "contact_sheet.jpg", quality=90)


def main() -> None:
    images = collect_images()
    if not images:
        print(f"No local photos found under: {ROOT}")
        return

    by_folder: defaultdict[str, int] = defaultdict(int)
    for image_path in images:
        by_folder[str(image_path.parent.relative_to(ROOT))] += 1

    print(f"Project: {ROOT}")
    print(f"Found {len(images)} image files in {len(by_folder)} folders:")
    for folder, count in sorted(by_folder.items(), key=lambda item: (-item[1], item[0]))[:25]:
        print(f"  {count:>5}  {folder}")

    source_name, candidates = select_source(images)
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    selected = sorted(candidates[: min(SAMPLE_SIZE, len(candidates))])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample_images = OUTPUT / "images"
    sample_labels = OUTPUT / "labels"
    sample_images.mkdir(parents=True, exist_ok=True)
    sample_labels.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for index, image_path in enumerate(selected, start=1):
        # Prefixing the number prevents collisions when source folders reuse names.
        sample_name = f"sample_{index:02d}_{image_path.name}"
        sample_path = sample_images / sample_name
        shutil.copy2(image_path, sample_path)
        label_path = matching_label(image_path)
        box_count, class_text = label_summary(label_path)
        label_status = "has YOLO label" if label_path else "no matched label"
        copied_label = ""
        if label_path:
            copied_label_path = sample_labels / f"{Path(sample_name).stem}.txt"
            shutil.copy2(label_path, copied_label_path)
            copied_label = str(copied_label_path.relative_to(OUTPUT))
        rows.append({
            "sample_path": str(sample_path),
            "source_path": str(image_path),
            "label_status": label_status,
            "label_path": copied_label,
            "box_count": str(box_count),
            "classes": class_text,
        })

    manifest_path = OUTPUT / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_path", "source_path", "label_status", "label_path", "box_count", "classes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    make_contact_sheet(rows)
    label_total = sum(row["label_status"] == "has YOLO label" for row in rows)
    print(f"\nSample source: {source_name}")
    print(f"Extracted {len(rows)} photos; {label_total} have matching YOLO label files.")
    print(f"Images: {sample_images}")
    print(f"Labels: {sample_labels}")
    print(f"Manifest: {manifest_path}")
    print(f"Contact sheet: {OUTPUT / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
