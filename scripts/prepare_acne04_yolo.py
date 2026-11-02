"""Convert the official ACNE04 VOC annotations into a reproducible YOLO split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ACNE04"
OUTPUT = ROOT / "data" / "processed" / "acne04_yolo"
IMAGE_TAR = RAW / "Classification.tar"
LABEL_TAR = RAW / "Detection.tar"
LEVEL_RE = re.compile(r"^levle([0-3])_", re.IGNORECASE)


def safe_member(tar: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    """Return an expected regular tar member and reject path traversal."""
    member = tar.getmember(name)
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not member.isfile():
        raise ValueError(f"Unsafe or non-file archive entry: {member.name}")
    return member


def read_split(tar: tarfile.TarFile, split: str, fold: int) -> list[str]:
    name = f"Detection/VOC2007/ImageSets/Main/NNEW_{split}_{fold}.txt"
    data = tar.extractfile(safe_member(tar, name))
    assert data is not None
    filenames = [line.split()[0] for line in data.read().decode("utf-8-sig").splitlines() if line.strip()]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Duplicate names in official split {name}")
    return filenames


def grade(filename: str) -> str:
    match = LEVEL_RE.match(filename)
    return match.group(1) if match else "unknown"


def stratified_train_val(filenames: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    groups: dict[str, list[str]] = {}
    for filename in filenames:
        groups.setdefault(grade(filename), []).append(filename)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    for label, names in sorted(groups.items()):
        rng.shuffle(names)
        n_val = max(1, round(len(names) * fraction)) if len(names) > 1 else 0
        val.extend(names[:n_val])
        train.extend(names[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def convert_box(obj: ET.Element, width: int, height: int) -> tuple[str, bool]:
    box = obj.find("bndbox")
    if box is None:
        raise ValueError("Object has no bndbox")
    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))
    old = (xmin, ymin, xmax, ymax)
    xmin = min(max(xmin, 0.0), float(width))
    xmax = min(max(xmax, 0.0), float(width))
    ymin = min(max(ymin, 0.0), float(height))
    ymax = min(max(ymax, 0.0), float(height))
    clipped = old != (xmin, ymin, xmax, ymax)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate box after clipping: {old}")
    xc = (xmin + xmax) / (2.0 * width)
    yc = (ymin + ymax) / (2.0 * height)
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}", clipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if not IMAGE_TAR.is_file() or not LABEL_TAR.is_file():
        raise FileNotFoundError(f"Expected official archives in {RAW}")

    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (OUTPUT / kind / split).mkdir(parents=True, exist_ok=True)

    with tarfile.open(IMAGE_TAR, "r:*") as image_tar, tarfile.open(LABEL_TAR, "r:*") as label_tar:
        trainval = read_split(label_tar, "trainval", args.fold)
        test = read_split(label_tar, "test", args.fold)
        if set(trainval) & set(test):
            raise ValueError("Official trainval and test split overlap")
        train, val = stratified_train_val(trainval, args.val_fraction, args.seed)
        split_map = {"train": train, "val": val, "test": test}

        image_member_names = set(image_tar.getnames())
        xml_member_names = set(label_tar.getnames())
        image_count = len([n for n in image_member_names if n.startswith("Classification/JPEGImages/") and n.lower().endswith(".jpg")])
        xml_count = len([n for n in xml_member_names if n.startswith("Detection/VOC2007/Annotations/") and n.lower().endswith(".xml")])
        all_names = train + val + test
        if len(set(all_names)) != len(all_names):
            raise ValueError("Prepared splits overlap")
        if image_count != 1457 or xml_count != 1457 or len(all_names) != 1457:
            raise ValueError(f"Unexpected archive/split totals: images={image_count}, xml={xml_count}, selected={len(all_names)}")

        object_counts: dict[str, Counter[str]] = {s: Counter() for s in split_map}
        level_counts: dict[str, Counter[str]] = {s: Counter() for s in split_map}
        clipped_boxes = 0
        empty_images = 0
        manifest: list[dict[str, str | int]] = []

        for split, names in split_map.items():
            for filename in names:
                image_name = f"Classification/JPEGImages/{filename}"
                xml_name = f"Detection/VOC2007/Annotations/{Path(filename).stem}.xml"
                image_member = safe_member(image_tar, image_name)
                xml_member = safe_member(label_tar, xml_name)
                image_src = image_tar.extractfile(image_member)
                xml_src = label_tar.extractfile(xml_member)
                assert image_src is not None and xml_src is not None

                image_path = OUTPUT / "images" / split / filename
                label_path = OUTPUT / "labels" / split / f"{Path(filename).stem}.txt"
                image_path.write_bytes(image_src.read())

                root = ET.fromstring(xml_src.read())
                size = root.find("size")
                if size is None:
                    raise ValueError(f"Missing size in {xml_name}")
                width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid image size in {xml_name}")
                rows = []
                for obj in root.findall("object"):
                    source_class = (obj.findtext("name") or "").strip()
                    if source_class != "fore":
                        raise ValueError(f"Unexpected ACNE04 class {source_class!r} in {xml_name}")
                    row, clipped = convert_box(obj, width, height)
                    rows.append(row)
                    object_counts[split]["lesion"] += 1
                    clipped_boxes += int(clipped)
                if not rows:
                    empty_images += 1
                label_path.write_text("\n".join(rows), encoding="utf-8")
                level = grade(filename)
                level_counts[split][level] += 1
                manifest.append({"filename": filename, "split": split, "official_fold": args.fold, "hayashi_level_from_name": level, "lesion_boxes": len(rows)})

    data_root = OUTPUT.resolve().as_posix()
    (OUTPUT / "data.yaml").write_text(
        f"path: '{data_root}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: acne_lesion\n",
        encoding="utf-8",
    )
    with (OUTPUT / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "source": "ACNE04 official release",
        "official_fold": args.fold,
        "seed": args.seed,
        "validation_fraction_of_official_trainval": args.val_fraction,
        "class_mapping": {"fore": "acne_lesion"},
        "images_by_split": {k: len(v) for k, v in split_map.items()},
        "lesion_boxes_by_split": {k: int(v["lesion"]) for k, v in object_counts.items()},
        "grade_from_filename_by_split": {k: dict(v) for k, v in level_counts.items()},
        "empty_images": empty_images,
        "boxes_clipped_to_image_bounds": clipped_boxes,
        "limitations": [
            "ACNE04 annotations contain one class and do not identify lesion types.",
            "The dataset does not provide reliable subject IDs; image-level splits cannot guarantee subject-level independence.",
            "Validation is a seeded random subset of the official trainval fold; test remains the official fold test split.",
        ],
    }
    (OUTPUT / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
