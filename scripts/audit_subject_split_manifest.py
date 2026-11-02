"""Check image-manifest completeness and prevent participant leakage across splits."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "subject_split_manifest.csv"
REQUIRED_COLUMNS = {"image_id", "relative_path", "participant_id", "session_id", "split"}
VALID_SPLITS = {"train", "val", "test"}


def main() -> int:
    parser = argparse.ArgumentParser(description="按匿名受试者检查 train/val/test 是否隔离")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if not manifest.is_file():
        parser.error(f"找不到清单：{manifest}")

    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            parser.error(f"清单缺少列：{', '.join(sorted(missing))}")
        rows = list(reader)

    errors: list[str] = []
    image_ids: set[str] = set()
    participant_splits: dict[str, set[str]] = defaultdict(set)
    images_per_split: dict[str, int] = defaultdict(int)
    subjects_per_split: dict[str, set[str]] = defaultdict(set)
    sessions_unrecorded = 0
    times_recorded = 0

    for line_number, row in enumerate(rows, start=2):
        image_id = (row.get("image_id") or "").strip()
        relative_path = (row.get("relative_path") or "").strip()
        participant_id = (row.get("participant_id") or "").strip()
        session_id = (row.get("session_id") or "").strip()
        split = (row.get("split") or "").strip().lower()
        if (row.get("capture_date") or "").strip() or (row.get("sequence_order") or "").strip():
            times_recorded += 1

        if not image_id:
            errors.append(f"第 {line_number} 行缺少 image_id")
        elif image_id in image_ids:
            errors.append(f"第 {line_number} 行 image_id 重复：{image_id}")
        image_ids.add(image_id)

        if not relative_path:
            errors.append(f"第 {line_number} 行缺少 relative_path")
        elif not (ROOT / relative_path).is_file():
            errors.append(f"第 {line_number} 行找不到图片：{relative_path}")

        if not participant_id:
            errors.append(f"第 {line_number} 行缺少匿名 participant_id：{image_id or '(无 image_id)'}")
        if not session_id:
            sessions_unrecorded += 1
        if split not in VALID_SPLITS:
            errors.append(f"第 {line_number} 行 split 必须是 train/val/test：{image_id or '(无 image_id)'}")
        else:
            images_per_split[split] += 1
            if participant_id:
                participant_splits[participant_id].add(split)
                subjects_per_split[split].add(participant_id)

    for participant_id, splits in sorted(participant_splits.items()):
        if len(splits) > 1:
            errors.append(f"受试者 {participant_id} 跨集合出现：{', '.join(sorted(splits))}")

    print(f"清单：{manifest}")
    print(f"图片数：{len(rows)}")
    print(f"已记录日期或先后顺序：{times_recorded}/{len(rows)} 张")
    for split in sorted(VALID_SPLITS):
        print(f"{split}: {images_per_split[split]} 张，{len(subjects_per_split[split])} 位受试者")

    if errors:
        print("\n发现的问题：")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\n通过：图片路径有效，匿名受试者未跨集合。")
    if sessions_unrecorded:
        print(f"提示：{sessions_unrecorded} 张照片未记录 session_id；可用于受试者隔离检查，暂不适合做时间点/随访分析。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
