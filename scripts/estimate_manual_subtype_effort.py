"""Estimate subtype relabeling effort from existing single-class ACNE04 boxes."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LABELS = ROOT / "data" / "processed" / "acne04_yolo" / "labels" / "train"
GRADE_RE = re.compile(r"^levle([0-3])_", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-grade", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.images_per_grade <= 0:
        raise ValueError("--images-per-grade must be positive")
    if not TRAIN_LABELS.is_dir():
        raise FileNotFoundError(f"ACNE04 YOLO labels not found: {TRAIN_LABELS}")

    by_grade: dict[str, list[Path]] = {str(level): [] for level in range(4)}
    for label_path in TRAIN_LABELS.glob("*.txt"):
        match = GRADE_RE.match(label_path.stem)
        if match:
            by_grade[match.group(1)].append(label_path)

    rng = random.Random(args.seed)
    chosen: list[tuple[str, Path]] = []
    for grade, paths in by_grade.items():
        if len(paths) < args.images_per_grade:
            raise ValueError(f"Grade {grade} has only {len(paths)} train images")
        rng.shuffle(paths)
        chosen.extend((grade, path) for path in paths[: args.images_per_grade])

    total_boxes = 0
    print(f"Pilot: {len(chosen)} train images ({args.images_per_grade} per severity grade)")
    for grade in sorted(by_grade):
        grade_items = [(g, path) for g, path in chosen if g == grade]
        box_count = sum(
            sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            for _, path in grade_items
        )
        total_boxes += box_count
        print(f"Grade {grade}: {len(grade_items)} images, {box_count} existing lesion boxes")
    print(f"Total subtype decisions needed for this pilot: {total_boxes}")
    print(f"Average per image: {total_boxes / len(chosen):.1f} boxes")
    print("Each box already has coordinates; this is class relabeling, not redrawing the box.")


if __name__ == "__main__":
    main()
