"""Map AcneSCU Pascal VOC labels to this project's acne taxonomy and export YOLO data.

Source-to-project mapping:
  open_comedo        -> open_comedo
  closed_comedo      -> closed_comedo
  papule             -> papule
  pustule            -> pustule
  nodule             -> nodule_or_cyst (nodule examples only; no cyst class exists)
  atrophic_scar      -> scar
  hypertrophic_scar  -> scar

AcneSCU's melasma and nevus labels are outside the project taxonomy. Images
containing `other` are sent to a review queue and excluded from YOLO export.
Acne marks have no direct AcneSCU source label and remain a manual-labeling gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import stat
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "AcneSCU"
DEFAULT_ARCHIVE = ROOT / "data" / "raw" / "AcneSCU_2.zip"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "acnescu_yolo"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_CLASSES = (
    "open_comedo",
    "closed_comedo",
    "papule",
    "pustule",
    "nodule_or_cyst",
    "scar",
)
CLASS_ID = {name: index for index, name in enumerate(MODEL_CLASSES)}
PROJECT_CLASSES = MODEL_CLASSES + ("acne_mark",)

SOURCE_TO_TARGET = {
    "open_comedo": "open_comedo",
    "closed_comedo": "closed_comedo",
    "papule": "papule",
    "pustule": "pustule",
    "nodule": "nodule_or_cyst",
    "atrophic_scar": "scar",
    "hypertrophic_scar": "scar",
}
IGNORED_SOURCE_CLASSES = {"melasma", "nevus"}
REVIEW_SOURCE_CLASSES = {"other"}
KNOWN_SOURCE_CLASSES = set(SOURCE_TO_TARGET) | IGNORED_SOURCE_CLASSES | REVIEW_SOURCE_CLASSES

SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "test": "test"}
COLORS = {
    "open_comedo": "#00C2FF",
    "closed_comedo": "#FFB000",
    "papule": "#FF4D6D",
    "pustule": "#9B5DE5",
    "nodule_or_cyst": "#F15BB5",
    "scar": "#00A878",
}


@dataclass
class Box:
    source_class: str
    target_class: str | None
    line: str | None


@dataclass
class ImageRecord:
    xml_path: Path
    image_path: Path
    split: str | None
    boxes: list[Box] = field(default_factory=list)
    review_classes: set[str] = field(default_factory=set)

    @property
    def target_classes(self) -> set[str]:
        return {box.target_class for box in self.boxes if box.target_class is not None}


def normalize_class(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").split())


def split_from_path(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts[:-1]:
        normalized = part.lower()
        if normalized in SPLIT_NAMES:
            return SPLIT_NAMES[normalized]
    return None


def build_image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            name_key = path.name.lower()
            if name_key in by_name:
                raise ValueError(f"Duplicate image filename in source: {path.name}")
            by_name[name_key] = path
            by_stem[path.stem.lower()].append(path)
    return by_name, by_stem


def match_image(xml_root: ET.Element, xml_path: Path, by_name: dict[str, Path], by_stem: dict[str, list[Path]]) -> Path:
    filename = (xml_root.findtext("filename") or "").strip()
    if filename:
        exact = by_name.get(Path(filename).name.lower())
        if exact is not None:
            return exact
        stem = Path(filename).stem.lower()
    else:
        stem = xml_path.stem.lower()
    candidates = by_stem.get(stem, [])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No image matches annotation {xml_path}")
    raise ValueError(f"Several images match {xml_path.name}; provide unique image filenames")


def normalize_box(obj: ET.Element, width: int, height: int) -> str:
    box = obj.find("bndbox")
    if box is None:
        raise ValueError("VOC object has no bndbox")
    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))
    xmin = min(max(xmin, 0.0), float(width))
    xmax = min(max(xmax, 0.0), float(width))
    ymin = min(max(ymin, 0.0), float(height))
    ymax = min(max(ymax, 0.0), float(height))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate VOC box: {(xmin, ymin, xmax, ymax)}")
    xc = (xmin + xmax) / (2.0 * width)
    yc = (ymin + ymax) / (2.0 * height)
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    return f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def read_records(source_root: Path) -> tuple[list[ImageRecord], Counter[str]]:
    xml_files = sorted(source_root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No Pascal VOC XML annotations found under {source_root}")
    by_name, by_stem = build_image_index(source_root)
    if not by_name:
        raise FileNotFoundError(f"No image files found under {source_root}")

    records: list[ImageRecord] = []
    source_counts: Counter[str] = Counter()
    unknown: set[str] = set()
    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        image_path = match_image(root, xml_path, by_name, by_stem)
        size = root.find("size")
        if size is None:
            raise ValueError(f"VOC annotation has no <size>: {xml_path}")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions in {xml_path}")

        record = ImageRecord(xml_path, image_path, split_from_path(image_path, source_root))
        for obj in root.findall("object"):
            source_class = normalize_class(obj.findtext("name", ""))
            if not source_class:
                raise ValueError(f"Object without a class name in {xml_path}")
            source_counts[source_class] += 1
            if source_class not in KNOWN_SOURCE_CLASSES:
                unknown.add(source_class)
                continue
            if source_class in REVIEW_SOURCE_CLASSES:
                record.review_classes.add(source_class)
                record.boxes.append(Box(source_class, None, None))
                continue
            if source_class in IGNORED_SOURCE_CLASSES:
                record.boxes.append(Box(source_class, None, None))
                continue
            line = normalize_box(obj, width, height)
            target_class = SOURCE_TO_TARGET[source_class]
            record.boxes.append(Box(source_class, target_class, line))
        records.append(record)

    if unknown:
        raise ValueError(
            "Unmapped AcneSCU source classes found: "
            + ", ".join(sorted(unknown))
            + ". Add an explicit mapping or ignore rule before conversion."
        )
    return records, source_counts


def assign_splits(records: list[ImageRecord], seed: int) -> None:
    all_splits = {"train", "val", "test"}
    present = {record.split for record in records if record.split is not None}
    if present == all_splits:
        for record in records:
            if record.split is None:
                record.split = "train"
        return

    # A single `train` folder is not a usable train/validation/test split.
    # In that case, create a deterministic image-level split for all records.
    if not present or present == {"train"} or "train" not in present:
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * 0.1)) if n >= 10 else 0
        n_val = max(1, round(n * 0.1)) if n >= 10 else 0
        for index, record in enumerate(shuffled):
            if index < n_test:
                record.split = "test"
            elif index < n_test + n_val:
                record.split = "val"
            else:
                record.split = "train"
        return

    for record in records:
        if record.split is None:
            record.split = "train"
    rng = random.Random(seed)
    for missing_split in ("val", "test"):
        if missing_split not in present:
            train_records = [record for record in records if record.split == "train"]
            rng.shuffle(train_records)
            move_count = max(1, round(len(records) * 0.1)) if len(train_records) > 2 else 0
            for record in train_records[:move_count]:
                record.split = missing_split


def write_dataset(records: list[ImageRecord], source_counts: Counter[str], output_root: Path, sample_count: int, seed: int) -> None:
    if output_root.exists():
        raise FileExistsError(f"Output already exists; move it or choose another path: {output_root}")
    exported = [record for record in records if not record.review_classes]
    excluded = [record for record in records if record.review_classes]
    if not exported:
        raise ValueError("Every image contains a review-only `other` annotation; inspect the source labels first.")
    output_root.mkdir(parents=True)

    target_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for record in exported:
        assert record.split in {"train", "val", "test"}
        split = record.split
        split_counts[split] += 1
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dest = image_dir / record.image_path.name
        shutil.copy2(record.image_path, image_dest)
        yolo_lines: list[str] = []
        for box in record.boxes:
            if box.target_class is None or box.line is None:
                continue
            target_counts[box.target_class] += 1
            yolo_lines.append(f"{CLASS_ID[box.target_class]} {box.line}")
        (label_dir / f"{record.image_path.stem}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )

    with (output_root / "review_queue.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "source_classes", "action"])
        writer.writeheader()
        for record in excluded:
            writer.writerow(
                {
                    "image": str(record.image_path),
                    "source_classes": ",".join(sorted(record.review_classes)),
                    "action": "Review `other` objects; confirm whether to relabel or discard this image.",
                }
            )

    names_text = "\n".join(f"  {index}: {name}" for index, name in enumerate(MODEL_CLASSES))
    (output_root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + names_text
        + "\n",
        encoding="utf-8",
    )

    image_class_counts: Counter[str] = Counter()
    for record in exported:
        image_class_counts.update(record.target_classes)
    report = {
        "source": "AcneSCU Pascal VOC annotations",
        "split_unit": "image-level; source has no usable subject IDs",
        "source_images": len(records),
        "exported_images_by_split": dict(split_counts),
        "images_excluded_pending_review": len(excluded),
        "source_object_counts": dict(sorted(source_counts.items())),
        "mapped_object_counts": {name: target_counts.get(name, 0) for name in MODEL_CLASSES},
        "images_containing_target_class": {name: image_class_counts.get(name, 0) for name in MODEL_CLASSES},
        "project_classes_without_source_labels": ["acne_mark"],
        "partial_coverage": {
            "nodule_or_cyst": "Source has nodule annotations but no cyst class; cyst examples need manual labels.",
        },
        "ignored_source_classes": sorted(IGNORED_SOURCE_CLASSES),
        "review_source_classes": sorted(REVIEW_SOURCE_CLASSES),
        "class_ids": CLASS_ID,
        "notes": [
            "`melasma` and `nevus` are not treated as acne marks.",
            "Atrophic and hypertrophic scars are merged into `scar` and remain separate from active lesions.",
            "Add `acne_mark` only after manually labeling examples; append it as class ID 6 to preserve current IDs.",
        ],
    }
    (output_root / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_mapping_notes(output_root)
    write_preview(exported, output_root, sample_count, seed)


def write_mapping_notes(output_root: Path) -> None:
    rows = [
        ("open_comedo", "open_comedo", "direct"),
        ("closed_comedo", "closed_comedo", "direct"),
        ("papule", "papule", "direct"),
        ("pustule", "pustule", "direct"),
        ("nodule", "nodule_or_cyst", "partial: no cyst annotations"),
        ("atrophic_scar", "scar", "merge scar subtypes"),
        ("hypertrophic_scar", "scar", "merge scar subtypes"),
        ("melasma", "", "ignore: outside project taxonomy"),
        ("nevus", "", "ignore: outside project taxonomy"),
        ("other", "", "exclude image and send to review queue"),
        ("acne_mark", "acne_mark", "missing source class; manual annotation required"),
    ]
    with (output_root / "taxonomy_mapping.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_class", "project_class", "handling"])
        writer.writerows(rows)


def select_preview_records(records: list[ImageRecord], sample_count: int, seed: int) -> list[ImageRecord]:
    eligible = [record for record in records if record.target_classes]
    if not eligible or sample_count <= 0:
        return []
    image_frequency: Counter[str] = Counter()
    for record in eligible:
        image_frequency.update(record.target_classes)
    selected: list[ImageRecord] = []
    ordered_classes = sorted(MODEL_CLASSES, key=lambda name: image_frequency[name] or 10**9)
    for class_name in ordered_classes:
        candidates = [record for record in eligible if class_name in record.target_classes and record not in selected]
        if candidates and len(selected) < sample_count:
            candidates.sort(key=lambda record: (-len(record.target_classes), record.image_path.name.lower()))
            selected.append(candidates[0])
    remaining = [record for record in eligible if record not in selected]
    random.Random(seed).shuffle(remaining)
    selected.extend(remaining[: max(0, sample_count - len(selected))])
    return selected[:sample_count]


def write_preview(records: list[ImageRecord], output_root: Path, sample_count: int, seed: int) -> None:
    selected = select_preview_records(records, sample_count, seed)
    if not selected:
        return
    tile_w, tile_h = 360, 300
    columns = 3
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, 48 + rows * tile_h), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text((14, 14), "AcneSCU mapped ground-truth annotations — review class mapping", fill="#222222")
    sample_rows: list[dict[str, str]] = []

    for index, record in enumerate(selected):
        split = record.split
        assert split in {"train", "val", "test"}
        dest_image = output_root / "images" / split / record.image_path.name
        image = Image.open(dest_image).convert("RGB")
        draw = ImageDraw.Draw(image)
        labels: set[str] = set()
        for box in record.boxes:
            if box.target_class is None or box.line is None:
                continue
            class_id = CLASS_ID[box.target_class]
            fields = box.line.split()
            xc, yc, bw, bh = (float(value) for value in fields)
            xmin = (xc - bw / 2) * image.width
            ymin = (yc - bh / 2) * image.height
            xmax = (xc + bw / 2) * image.width
            ymax = (yc + bh / 2) * image.height
            color = COLORS[box.target_class]
            draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=max(2, min(image.size) // 450))
            tag = f"{class_id}:{box.target_class}"
            text_box = draw.textbbox((0, 0), tag)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            tag_y = max(0, ymin - text_h - 4)
            draw.rectangle((xmin, tag_y, xmin + text_w + 6, tag_y + text_h + 4), fill=color)
            draw.text((xmin + 3, tag_y + 2), tag, fill="white")
            labels.add(box.target_class)

        image.thumbnail((tile_w - 12, tile_h - 40))
        x = (index % columns) * tile_w + (tile_w - image.width) // 2
        y = 48 + (index // columns) * tile_h + 30
        sheet.paste(image, (x, y))
        sheet_draw.text(((index % columns) * tile_w + 8, 48 + (index // columns) * tile_h + 7), record.image_path.name, fill="#222222")
        sample_rows.append(
            {
                "image": record.image_path.name,
                "split": str(split),
                "mapped_classes": ",".join(sorted(labels)),
            }
        )

    preview_dir = output_root / "annotation_sample"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_dir / "contact_sheet.jpg", quality=94)
    with (preview_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "split", "mapped_classes"])
        writer.writeheader()
        writer.writerows(sample_rows)


def extract_voc_files(archive_path: Path, destination: Path) -> None:
    """Extract only VOC XML and image files, rejecting traversal and symlinks."""
    allowed_suffixes = IMAGE_EXTENSIONS | {".xml"}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
            if member.is_dir() or relative.suffix.lower() not in allowed_suffixes:
                continue
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Refusing symlink in source archive: {member.filename}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def convert_source(source_root: Path, output_root: Path, sample_count: int, seed: int) -> None:
    records, source_counts = read_records(source_root)
    assign_splits([record for record in records if not record.review_classes], seed)
    write_dataset(records, source_counts, output_root, sample_count, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map AcneSCU VOC annotations to the project's YOLO classes.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE, help="Extracted AcneSCU dataset root")
    parser.add_argument("--source-zip", type=Path, help="AcneSCU ZIP archive; Python extracts only images and VOC XML")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=12, help="Number of mapped images in the annotation preview")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    archive_path = args.source_zip.resolve() if args.source_zip else DEFAULT_ARCHIVE
    if args.source_zip or archive_path.is_file():
        if not archive_path.is_file():
            raise FileNotFoundError(f"AcneSCU archive not found: {archive_path}")
        with tempfile.TemporaryDirectory(prefix="acnescu_voc_") as temporary:
            source_root = Path(temporary)
            extract_voc_files(archive_path, source_root)
            convert_source(source_root, output_root, args.sample_count, args.seed)
    else:
        source_root = args.source_root.resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"AcneSCU files are not extracted at {source_root}. Pass the archive with --source-zip "
                "or use the extracted folder with --source-root."
            )
        convert_source(source_root, output_root, args.sample_count, args.seed)
    print(f"Mapped dataset written to: {output_root}")
    print(f"Image preview: {output_root / 'annotation_sample' / 'contact_sheet.jpg'}")
    print(f"Manual review queue: {output_root / 'review_queue.csv'}")


if __name__ == "__main__":
    main()
