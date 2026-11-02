"""Append prepared independent-test photos to the subject split manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "data" / "independent_test_batch_01"
DEFAULT_MANIFEST = ROOT / "data" / "subject_split_manifest.csv"
FIELDS = [
    "image_id", "relative_path", "participant_id", "session_id", "capture_date", "capture_time",
    "time_precision", "sequence_order", "split", "notes",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="把已准备的测试照片登记到受试者划分清单")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    batch_dir = args.batch_dir if args.batch_dir.is_absolute() else ROOT / args.batch_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    images_dir = batch_dir / "images"
    source_manifest = batch_dir / "image_manifest.csv"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"找不到图片目录：{images_dir}")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"找不到图片清单：{source_manifest}；先运行 prepare_photo_annotation_pilot.py")

    with source_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"图片清单为空：{source_manifest}")

    prepared: list[dict[str, str]] = []
    prefix = batch_dir.name
    for row in source_rows:
        image_name = (row.get("image") or "").strip()
        image_path = images_dir / image_name
        if not image_name or image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            raise ValueError(f"图片清单中的图片缺失或格式不支持：{image_name!r}")
        prepared.append(
            {
                "image_id": f"{prefix}_{image_path.stem}",
                "relative_path": relative_to_root(image_path),
                "participant_id": "",
                "session_id": "",
                "capture_date": "",
                "capture_time": "",
                "time_precision": "",
                "sequence_order": "",
                "split": "test",
                "notes": "新独立测试照片；填写匿名受试者和会话编号",
            }
        )

    existing: list[dict[str, str]] = []
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) != set(FIELDS):
                raise ValueError(f"现有划分清单列不匹配：需要 {FIELDS}")
            existing = list(reader)

    existing_paths = {
        (row.get("relative_path") or "").replace("\\", "/").casefold()
        for row in existing
    }
    existing_ids = {row.get("image_id", "") for row in existing}
    duplicates = [
        row["relative_path"]
        for row in prepared
        if row["relative_path"].casefold() in existing_paths or row["image_id"] in existing_ids
    ]
    if duplicates:
        raise ValueError("这些照片已登记，未修改清单：\n" + "\n".join(duplicates))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(prepared)
    os.replace(temp_path, manifest_path)

    print(f"已登记 {len(prepared)} 张测试照片：{manifest_path}")
    print("接着给新增行填写 participant_id 和 session_id，再运行 audit_subject_split_manifest.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
