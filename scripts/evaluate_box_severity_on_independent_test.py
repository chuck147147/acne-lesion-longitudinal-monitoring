"""Evaluate the V2 crop classifier on a participant-isolated, labeled test batch."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "data" / "independent_test_batch_01"
DEFAULT_CLASSIFIER = (
    ROOT / "runs" / "classify" / "acne_box_severity_yolo11n_v2_finetune" / "weights" / "best.pt"
)
DEFAULT_MANIFEST = ROOT / "data" / "subject_split_manifest.csv"
LABELS = ("0*", "I", "II", "III", "IV")
ACTIVE_GRADES = ("I", "II", "III", "IV")
EXCLUDED_GRADE = "0*"
FOLDER_TO_LABEL = {
    "grade_0star": "0*",
    "grade_I": "I",
    "grade_II": "II",
    "grade_III": "III",
    "grade_IV": "IV",
}


def normalized_relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix().casefold()


def load_test_image_manifest(manifest_path: Path, batch_dir: Path) -> tuple[set[str], set[str]]:
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "participant_id", "session_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"受试者清单缺少列：{', '.join(sorted(missing))}")
        rows = list(reader)

    by_path: dict[str, dict[str, str]] = {}
    participant_splits: dict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows, start=2):
        relative = (row.get("relative_path") or "").strip().replace("\\", "/")
        participant = (row.get("participant_id") or "").strip()
        session = (row.get("session_id") or "").strip()
        split = (row.get("split") or "").strip().lower()
        if not relative:
            continue
        if not participant:
            raise ValueError(f"受试者清单第 {line_number} 行缺少 participant_id：{relative}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"受试者清单第 {line_number} 行 split 必须是 train/val/test：{relative}")
        key = relative.casefold()
        if key in by_path:
            raise ValueError(f"受试者清单第 {line_number} 行 relative_path 重复：{relative}")
        by_path[key] = row
        if participant and split in {"train", "val", "test"}:
            participant_splits[participant].add(split)
        if split == "test" and not participant:
            raise ValueError(f"测试图片缺少 participant_id：{relative}")

    overlap = {
        participant: splits
        for participant, splits in participant_splits.items()
        if "test" in splits and ("train" in splits or "val" in splits)
    }
    if overlap:
        details = "; ".join(f"{pid}: {','.join(sorted(splits))}" for pid, splits in sorted(overlap.items()))
        raise ValueError(f"test 受试者与 train/val 重叠：{details}")

    batch_prefix = normalized_relative_path(batch_dir) + "/"
    test_paths = {
        key
        for key, row in by_path.items()
        if (row.get("split") or "").strip().lower() == "test"
        and key.startswith(batch_prefix)
    }
    if not test_paths:
        raise ValueError(f"清单中没有指向该测试批次的 test 图片：{batch_dir}")
    return test_paths, set(participant_splits)


def crop_box(image: Image.Image, row: dict[str, str]) -> Image.Image:
    width, height = image.size
    cx = float(row["x_center_norm"]) * width
    cy = float(row["y_center_norm"]) * height
    box_w = float(row["width_norm"]) * width
    box_h = float(row["height_norm"]) * height
    side = max(96.0, 3.0 * max(box_w, box_h))
    half = side / 2
    x1 = max(0, int(round(cx - half)))
    y1 = max(0, int(round(cy - half)))
    x2 = min(width, int(round(cx + half)))
    y2 = min(height, int(round(cy + half)))
    return image.crop((x1, y1, x2, y2))


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="在受试者隔离的独立测试集上评估逐框分类器")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    batch_dir = args.batch_dir if args.batch_dir.is_absolute() else ROOT / args.batch_dir
    images_dir = batch_dir / "images"
    annotations_path = args.annotations or batch_dir / "annotations.csv"
    if not annotations_path.is_absolute():
        annotations_path = ROOT / annotations_path
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    classifier_path = args.classifier if args.classifier.is_absolute() else ROOT / args.classifier
    output_dir = args.output_dir or (ROOT / "runs" / "evaluation" / "box_severity_v2_independent_test")
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    for required_path in (images_dir, annotations_path, manifest_path, classifier_path):
        if not required_path.exists():
            raise FileNotFoundError(f"找不到所需路径：{required_path}")

    test_paths, _ = load_test_image_manifest(manifest_path, batch_dir)
    image_path_by_name: dict[str, Path] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    excluded_uncertain = 0
    with annotations_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image", "box_id", "x_center_norm", "y_center_norm", "width_norm", "height_norm",
            "box_severity_grade",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"标注表缺少列：{', '.join(sorted(missing))}")
        for row in reader:
            image_name = (row.get("image") or "").strip()
            image_path = images_dir / image_name
            rel = normalized_relative_path(image_path)
            if rel not in test_paths:
                raise ValueError(f"标注图片未在受试者清单中标记为独立 test：{image_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"标注引用的图片不存在：{image_path}")
            image_path_by_name[image_name] = image_path
            grade = (row.get("box_severity_grade") or "").strip()
            review_note = (row.get("review_note") or "").strip().lower()
            if review_note == "uncertain" or grade in {"", "uncertain"}:
                excluded_uncertain += 1
                continue
            if grade not in LABELS:
                raise ValueError(f"未知标签 {grade!r}：{image_name} 框 {row.get('box_id')}")
            grouped[image_name].append(row)

    total_rows = sum(len(rows) for rows in grouped.values())
    if total_rows == 0:
        raise ValueError("没有可评估的已标注框。请完成独立测试批次的人工标注。")

    model = YOLO(str(classifier_path))
    model_labels = {
        index: FOLDER_TO_LABEL.get(str(name), str(name))
        for index, name in model.names.items()
    }
    predictions: list[dict[str, str | int | float]] = []
    for image_name, rows in sorted(grouped.items()):
        with Image.open(image_path_by_name[image_name]) as opened:
            image = opened.convert("RGB")
        crops = [crop_box(image, row) for row in rows]
        results = model.predict(
            source=crops,
            imgsz=224,
            batch=args.batch_size,
            device=args.device,
            verbose=False,
        )
        for row, result in zip(rows, results):
            probabilities = result.probs
            predicted_index = int(probabilities.top1)
            predictions.append(
                {
                    "image": image_name,
                    "box_id": row["box_id"],
                    "true_grade": row["box_severity_grade"],
                    "predicted_grade": model_labels[predicted_index],
                    "confidence": round(float(probabilities.top1conf), 6),
                }
            )

    confusion = {truth: Counter() for truth in LABELS}
    supports = Counter()
    predictions_by_grade = Counter()
    for row in predictions:
        truth = str(row["true_grade"])
        predicted = str(row["predicted_grade"])
        confusion[truth][predicted] += 1
        supports[truth] += 1
        predictions_by_grade[predicted] += 1

    metrics: dict[str, dict[str, float | int]] = {}
    active_rows = [row for row in predictions if row["true_grade"] in ACTIVE_GRADES]
    correct = sum(confusion[grade][grade] for grade in ACTIVE_GRADES)
    for grade in ACTIVE_GRADES:
        tp = confusion[grade][grade]
        fp = sum(confusion[truth][grade] for truth in LABELS if truth != grade)
        fn = sum(confusion[grade][predicted] for predicted in LABELS if predicted != grade)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        metrics[grade] = {
            "support": supports[grade],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_precision = sum(float(metrics[label]["precision"]) for label in ACTIVE_GRADES) / len(ACTIVE_GRADES)
    macro_recall = sum(float(metrics[label]["recall"]) for label in ACTIVE_GRADES) / len(ACTIVE_GRADES)
    macro_f1 = sum(float(metrics[label]["f1"]) for label in ACTIVE_GRADES) / len(ACTIVE_GRADES)
    accuracy = safe_ratio(correct, len(active_rows))
    excluded_rows = [row for row in predictions if row["true_grade"] == EXCLUDED_GRADE]
    excluded_correct = sum(row["predicted_grade"] == EXCLUDED_GRADE for row in excluded_rows)
    active_rejected = sum(row["predicted_grade"] == EXCLUDED_GRADE for row in active_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["image", "box_id", "true_grade", "predicted_grade", "confidence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    confusion_path = output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *LABELS])
        for truth in ACTIVE_GRADES:
            writer.writerow([truth, *(confusion[truth][predicted] for predicted in LABELS)])

    participant_ids = set()
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            relative = (row.get("relative_path") or "").strip().replace("\\", "/").casefold()
            if relative in test_paths and (row.get("split") or "").strip().lower() == "test":
                participant_ids.add((row.get("participant_id") or "").strip())

    report_lines = [
        "# 独立测试结果",
        "",
        f"- 清单中的测试照片数：{len(test_paths)}",
        f"- 有已评分框的照片数：{len(grouped)}",
        f"- 受试者数：{len(participant_ids)}",
        f"- 活动病灶框数（I–IV，仅此进入活动统计）：{len(active_rows)}",
        f"- 痘印/瘢痕 `0*` 排除于活动病灶统计；过滤核查：正确排除 {excluded_correct}/{len(excluded_rows)} 个，活动病灶误排除 {active_rejected} 个",
        f"- 因不确定/未标注排除的框数：{excluded_uncertain}",
        f"- 活动等级 Top-1 accuracy：{accuracy:.4f}",
        f"- Macro precision：{macro_precision:.4f}",
        f"- Macro recall：{macro_recall:.4f}",
        f"- Macro F1：{macro_f1:.4f}",
        f"- 分类器权重：`{classifier_path}`",
        f"- 参考框标注：`{annotations_path}`",
        "",
        "| 活动病灶等级 | 真值框数 | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for grade in ACTIVE_GRADES:
        metric = metrics[grade]
        report_lines.append(
            f"| {grade} | {metric['support']} | {metric['precision']:.4f} | "
            f"{metric['recall']:.4f} | {metric['f1']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "本结果只评价分类器在人工参考框裁剪图上的分类表现，不包含自动检测器的漏检/误检，也不等同于临床诊断性能。",
            "测试集应按受试者与训练/验证集隔离；若受试者数量很少，结果仅作探索性报告。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"独立测试完成：{len(test_paths)} 张测试照片，{len(active_rows)} 个活动病灶参考框，Top-1={accuracy:.1%}，Macro-F1={macro_f1:.3f}")
    print(f"结果目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
