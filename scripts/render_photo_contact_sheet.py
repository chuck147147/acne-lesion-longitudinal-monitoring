"""Render a compact review sheet for locally predicted photo boxes."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "photo_annotation_pilot"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=Path("data/photo_annotation_pilot"))
    args = parser.parse_args()
    pilot = args.batch_dir if args.batch_dir.is_absolute() else (ROOT / args.batch_dir).resolve()
    with (pilot / "prediction_summary.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        summary = list(csv.DictReader(handle))
    columns, tile_width, tile_height, header_height = 3, 420, 590, 68
    rows = (len(summary) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, header_height + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 18), "Previous model output · boxes are candidates · severity is still for manual labeling", fill="#222222")
    for index, item in enumerate(summary):
        image_path = pilot / "overlays" / f"{Path(item['image']).stem}_boxes.jpg"
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.contain(image, (400, 520), method=Image.Resampling.LANCZOS)
        x0, y0 = (index % columns) * tile_width, header_height + (index // columns) * tile_height
        draw.text((x0 + 12, y0 + 8), f"{item['image']} · {item['predicted_boxes']} boxes", fill="#222222")
        x = x0 + (tile_width - image.width) // 2
        sheet.paste(image, (x, y0 + 42))
    sheet.save(pilot / "contact_sheet.jpg", quality=92)
    print(f"Contact sheet: {pilot / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
