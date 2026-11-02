"""Build longitudinal trend artifacts from saved three-view monitoring runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REGIONS = ("左侧视角", "右侧视角", "下巴视角")
GRADES = ("0*", "I", "II", "III", "IV")
GRADE_WEIGHTS = {"0*": 0, "I": 1, "II": 2, "III": 2, "IV": 4}
COUNT_KEYS = {
    "0*": "marks_0star",
    "I": "comedone_I",
    "II": "papule_II",
    "III": "pustule_III",
    "IV": "nodule_IV",
}
REGION_COLORS = {
    "左侧视角": "#2878B5",
    "右侧视角": "#078A83",
    "下巴视角": "#8255A5",
}
PERIODS = {
    "morning": ("上午", 9),
    "noon": ("中午", 12),
    "afternoon": ("下午", 15),
    "evening": ("晚上", 20),
}


def normalize_region(value: object) -> str | None:
    text = str(value or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if text in {"左侧视角", "左脸", "left", "leftface", "leftcheek"}:
        return "左侧视角"
    if text in {"右侧视角", "右脸", "right", "rightface", "rightcheek"}:
        return "右侧视角"
    if text in {"下巴视角", "下巴", "chin", "jaw"}:
        return "下巴视角"
    return None


def parse_generated_at(value: object, run_dir: Path) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed
    except (TypeError, ValueError):
        match = re.fullmatch(r"(\d{8})_(\d{6})", run_dir.name)
        if match:
            return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        return datetime.fromtimestamp(run_dir.stat().st_mtime)


def capture_metadata(image_paths: list[str], generated_at: datetime) -> dict[str, Any]:
    combined = " ".join(image_paths).replace("\\", "/").lower()
    date_match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]?\d)[-_]?([0-3]?\d)(?!\d)", combined)
    capture_date: str | None = None
    period_name: str | None = None
    sort_time = generated_at
    label = generated_at.strftime("%m/%d %H:%M（记录时间）")
    if date_match:
        year, month, day = (int(part) for part in date_match.groups())
        try:
            date_value = datetime(year, month, day)
            capture_date = date_value.strftime("%Y-%m-%d")
            if any(token in combined for token in ("上午", "早上", "清晨", "morning")):
                period_name = "morning"
            elif any(token in combined for token in ("中午", "noon", "midday")):
                period_name = "noon"
            elif any(token in combined for token in ("下午", "afternoon")):
                period_name = "afternoon"
            elif any(token in combined for token in ("晚上", "夜间", "夜晚", "evening", "night")):
                period_name = "evening"
            hour = PERIODS[period_name][1] if period_name else generated_at.hour
            minute = generated_at.minute if period_name is None and generated_at.date() == date_value.date() else 0
            sort_time = date_value.replace(hour=hour, minute=minute)
            label = f"{month}/{day} {PERIODS[period_name][0]}" if period_name else f"{month}/{day}（时段未标）"
        except ValueError:
            capture_date = None

    batch_match = re.search(r"(?:batch|批次)[ _-]*([a-z0-9一二三四五六七八九十]+)", combined)
    batch = batch_match.group(1) if batch_match else ""
    return {
        "capture_date": capture_date,
        "period": period_name or "",
        "batch": batch,
        "sort_time": sort_time,
        "label": label,
        "has_capture_time": bool(capture_date and period_name),
    }


def path_session_key(image_paths_by_region: dict[str, str]) -> str:
    normalized = []
    for region in REGIONS:
        value = image_paths_by_region.get(region, "")
        normalized.append(os.path.normcase(os.path.normpath(value)) if value else "")
    return "paths:" + "|".join(normalized)


def content_session_key(image_paths_by_region: dict[str, str]) -> str:
    """Identify a capture set by image bytes so separate same-day batches stay separate."""
    digests: list[str] = []
    for region in REGIONS:
        value = image_paths_by_region.get(region, "")
        path = Path(value) if value else None
        try:
            if path is None or not path.is_file():
                raise OSError("source image unavailable")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests.append(digest.hexdigest())
        except OSError:
            digests.append(os.path.normcase(os.path.normpath(value)) if value else "missing")
    return "images:" + "|".join(digests)


def parse_counts(raw_counts: object) -> dict[str, dict[str, int]]:
    result = {
        region: {key: 0 for key in (*COUNT_KEYS.values(), "active_total", "inflammatory_total", "nodule_IV", "burden_points")}
        for region in REGIONS
    }
    if not isinstance(raw_counts, list):
        return result
    for item in raw_counts:
        if not isinstance(item, dict):
            continue
        region = normalize_region(item.get("region"))
        if region is None:
            continue
        for key in result[region]:
            try:
                result[region][key] = int(round(float(item.get(key, 0) or 0)))
            except (TypeError, ValueError):
                result[region][key] = 0
        result[region]["active_total"] = sum(result[region][COUNT_KEYS[grade]] for grade in GRADES if grade != "0*")
        result[region]["inflammatory_total"] = sum(result[region][COUNT_KEYS[grade]] for grade in ("II", "III", "IV"))
        result[region]["nodule_IV"] = result[region][COUNT_KEYS["IV"]]
        result[region]["burden_points"] = sum(result[region][COUNT_KEYS[grade]] * GRADE_WEIGHTS[grade] for grade in GRADES)
    return result


def count_lesions_csv(run_dir: Path, session: str) -> dict[str, dict[str, int]]:
    path = run_dir / "lesions.csv"
    if not path.is_file():
        return {}
    counts: dict[str, dict[str, int]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("session", "").strip().lower() != session:
                continue
            region = normalize_region(row.get("region"))
            grade = row.get("grade", "").strip()
            if region is None or grade not in GRADES:
                continue
            item = counts.setdefault(
                region,
                {key: 0 for key in (*COUNT_KEYS.values(), "active_total", "inflammatory_total", "nodule_IV", "burden_points")},
            )
            item[COUNT_KEYS[grade]] += 1
    for item in counts.values():
        item["active_total"] = sum(item[COUNT_KEYS[grade]] for grade in GRADES if grade != "0*")
        item["inflammatory_total"] = sum(item[COUNT_KEYS[grade]] for grade in ("II", "III", "IV"))
        item["nodule_IV"] = item[COUNT_KEYS["IV"]]
        item["burden_points"] = sum(item[COUNT_KEYS[grade]] * GRADE_WEIGHTS[grade] for grade in GRADES)
    return counts


def annotations_include_previous(run_dir: Path) -> bool:
    path = run_dir / "detection_review" / "annotations.csv"
    if not path.is_file():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return any(str(row.get("image", "")).startswith("previous_") for row in csv.DictReader(handle))
    except OSError:
        return False


def make_entry(
    run_dir: Path,
    result: dict[str, Any],
    session: str,
    paths: list[str],
    raw_counts: object,
    generated_at: datetime,
) -> dict[str, Any] | None:
    if len(paths) != 3:
        return None
    regions = [str(value) for value in result.get("regions", [])]
    path_by_region = {
        normalized: str(path)
        for normalized, path in (
            (normalize_region(region), path) for region, path in zip(regions, paths)
        )
        if normalized is not None
    }
    if set(path_by_region) != set(REGIONS):
        return None
    if session == "current":
        reviewed = result.get("box_review_applied") if isinstance(result.get("box_review_applied"), bool) else None
    elif result.get("baseline_loaded_from_reviewed_results"):
        reviewed = True
    elif annotations_include_previous(run_dir):
        reviewed = True
    else:
        reviewed = None

    raw_regions = {
        normalize_region(item.get("region"))
        for item in raw_counts if isinstance(item, dict)
    } if isinstance(raw_counts, list) else set()
    if set(REGIONS).issubset(raw_regions):
        metrics = parse_counts(raw_counts)
    else:
        metrics = count_lesions_csv(run_dir, session)
        if set(metrics) != set(REGIONS):
            return None

    meta = capture_metadata(paths, generated_at)
    # Untimed, unreviewed legacy runs cannot be safely positioned in a time series.
    if not meta["capture_date"] and reviewed is not True:
        return None
    if meta["capture_date"] and not meta["period"] and reviewed is not True:
        return None
    ordered_paths = {region: path_by_region[region] for region in REGIONS}
    if meta["capture_date"] and meta["period"]:
        signature = content_session_key(ordered_paths)
        key = f"capture:{meta['capture_date']}:{meta['period']}:{meta['batch']}:{signature}"
    else:
        key = path_session_key(ordered_paths)
    return {
        "session_key": key,
        "session_label": meta["label"],
        "capture_date": meta["capture_date"] or generated_at.date().isoformat(),
        "period": meta["period"],
        "sort_time": meta["sort_time"],
        "review_status": "已复核" if reviewed is True else "未复核" if reviewed is False else "状态未知",
        "review_rank": 2 if reviewed is True else 0 if reviewed is False else 1,
        "source_run": run_dir.name,
        "source_session": session,
        "images": ordered_paths,
        "image_paths": paths,
        "photo_count": len(paths),
        "active_boxes": sum(metrics[region]["active_total"] for region in REGIONS),
        "source_type": "复核后监测结果",
        "region_metrics_available": True,
        "regions": metrics,
        "generated_at": generated_at,
    }


def collect_legacy_manual_history(outputs_dir: Path) -> list[dict[str, Any]]:
    """Read the user's earlier, manually confirmed single-subject label sessions."""
    candidates = list(outputs_dir.glob("*_longitudinal_tracking*_review*/face_alignment_summary.csv"))
    if not candidates:
        return []

    def version_key(path: Path) -> tuple[int, float]:
        match = re.search(r"_review_v(\d+)$", path.parent.name)
        return (int(match.group(1)) if match else 0, path.stat().st_mtime)

    summary_path = max(candidates, key=version_key)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                session_id = str(row.get("session_id", "")).strip()
                date_text = str(row.get("capture_date", "")).strip()
                try:
                    date_value = datetime.strptime(date_text, "%Y-%m-%d")
                    active_boxes = int(float(row.get("active_labeled_boxes") or 0))
                except (TypeError, ValueError):
                    continue
                if not session_id:
                    continue
                item = grouped.setdefault((date_text, session_id), {
                    "date": date_value,
                    "active_boxes": 0,
                    "image_paths": [],
                })
                item["active_boxes"] += active_boxes
                relative_path = str(row.get("relative_path", "")).strip()
                if relative_path:
                    item["image_paths"].append(relative_path)
    except OSError:
        return []

    same_date: dict[str, list[tuple[str, str]]] = {}
    for key in grouped:
        same_date.setdefault(key[0], []).append(key)
    output: list[dict[str, Any]] = []
    generated_at = datetime.fromtimestamp(summary_path.stat().st_mtime)
    for (date_text, session_id), item in sorted(grouped.items(), key=lambda pair: (pair[1]["date"], pair[0][1])):
        date_value = item["date"]
        session_label = f"{date_value.month}/{date_value.day} · {len(item['image_paths'])}图"
        siblings = sorted(same_date[date_text], key=lambda key: key[1])
        if len(siblings) > 1:
            session_label = f"{date_value.month}/{date_value.day} 批次{siblings.index((date_text, session_id)) + 1} · {len(item['image_paths'])}图"
        output.append({
            "session_key": f"legacy:{session_id}",
            "session_label": session_label,
            "capture_date": date_text,
            "period": "",
            "sort_time": date_value.replace(hour=12),
            "review_status": "已复核",
            "review_rank": 2,
            "source_run": summary_path.parent.name,
            "source_session": session_id,
            "images": {},
            "image_paths": item["image_paths"],
            "photo_count": len(item["image_paths"]),
            "active_boxes": item["active_boxes"],
            "source_type": "历史人工标注",
            "region_metrics_available": False,
            "regions": {},
            "generated_at": generated_at,
        })
    return output


def collect_history(runs_dir: Path, extra_current: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    if runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            result_path = run_dir / "result.json"
            if not run_dir.is_dir() or not result_path.is_file():
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(result, dict):
                continue
            generated_at = parse_generated_at(result.get("generated_at"), run_dir)
            events = (
                ("previous", result.get("previous_images", []), result.get("previous_region_counts", [])),
                ("current", result.get("current_images", []), result.get("current_region_counts", [])),
            )
            for session, paths, raw_counts in events:
                if not isinstance(paths, list):
                    continue
                entry = make_entry(run_dir, result, session, [str(path) for path in paths], raw_counts, generated_at)
                if entry is None:
                    continue
                existing = by_key.get(entry["session_key"])
                rank = (entry["review_rank"], entry["generated_at"])
                existing_rank = (existing["review_rank"], existing["generated_at"]) if existing else None
                if existing is None or rank > existing_rank:
                    by_key[entry["session_key"]] = entry

    legacy_entries = collect_legacy_manual_history(runs_dir.parent)
    for legacy in legacy_entries:
        duplicate = any(
            item.get("region_metrics_available")
            and item["capture_date"] == legacy["capture_date"]
            and item["photo_count"] == legacy["photo_count"]
            and item["active_boxes"] == legacy["active_boxes"]
            for item in by_key.values()
        )
        if not duplicate:
            by_key[legacy["session_key"]] = legacy

    if extra_current:
        now = extra_current.get("generated_at")
        if isinstance(now, str):
            try:
                now = datetime.fromisoformat(now.replace("Z", "+00:00"))
                now = now.astimezone().replace(tzinfo=None) if now.tzinfo else now
            except ValueError:
                now = datetime.now()
        if not isinstance(now, datetime):
            now = datetime.now()
        extra_run = Path(str(extra_current.get("run_dir") or "current_session"))
        extra_result = {"regions": list(REGIONS), "box_review_applied": bool(extra_current.get("reviewed"))}
        entry = make_entry(
            extra_run,
            extra_result,
            "current",
            [str(path) for path in extra_current.get("images", [])],
            extra_current.get("counts", []),
            now,
        )
        if entry is not None:
            existing = by_key.get(entry["session_key"])
            rank = (entry["review_rank"], entry["generated_at"])
            existing_rank = (existing["review_rank"], existing["generated_at"]) if existing else None
            if existing is None or rank > existing_rank:
                by_key[entry["session_key"]] = entry

    entries = sorted(by_key.values(), key=lambda item: (item["sort_time"], item["session_key"]))
    # The user has multiple capture batches on one date. Keep them as separate
    # points and make the duplicate date/time labels legible in the chart.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        bucket = entry["period"] or ("legacy" if entry["source_type"] == "历史人工标注" else "")
        if bucket:
            grouped.setdefault((entry["capture_date"], bucket), []).append(entry)
    for group in grouped.values():
        if len(group) < 2:
            continue
        for index, entry in enumerate(sorted(group, key=lambda item: (item["sort_time"], item["session_key"])), start=1):
            if "批次" not in entry["session_label"]:
                entry["session_label"] += f" · 批次{index}"
    return entries


def history_csv_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row: dict[str, Any] = {
            "session_label": entry["session_label"],
            "capture_date": entry["capture_date"],
            "period": entry["period"],
            "review_status": entry["review_status"],
            "source_type": entry["source_type"],
            "source_run": entry["source_run"],
            "source_session": entry["source_session"],
            "photo_count": entry["photo_count"],
            "active_boxes_across_photos": entry["active_boxes"],
            "region_metrics_available": entry["region_metrics_available"],
            "images": " | ".join(entry.get("image_paths", [])) or " | ".join(entry["images"].get(region, "") for region in REGIONS),
        }
        metric_keys = (*COUNT_KEYS.values(), "active_total", "inflammatory_total", "nodule_IV", "burden_points")
        all_metrics = {key: 0 for key in metric_keys}
        for region in REGIONS:
            metrics = entry["regions"].get(region, {})
            for key in metric_keys:
                value = metrics.get(key, "")
                row[f"{region}_{key}"] = value
                if value != "":
                    all_metrics[key] += value
        for key, value in all_metrics.items():
            row[f"三视角_{key}"] = value if entry["region_metrics_available"] else ""
        rows.append(row)
    return rows


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc") if bold
        else ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_trend_panel(
    draw: ImageDraw.ImageDraw,
    entries: list[dict[str, Any]],
    values: list[int],
    point_labels: list[str],
    title: str,
    note: str,
    color: str,
    top: int,
    panel_height: int,
    panel_x: int,
    panel_width: int,
    title_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(
        (panel_x, top, panel_x + panel_width, top + panel_height),
        radius=20, fill="white", outline="#D5DDE3", width=2,
    )
    draw.text((panel_x + 28, top + 18), title, font=title_font, fill="#1D2A34")
    draw.text((panel_x + 390, top + 27), note, font=small_font, fill="#697782")
    if not entries:
        draw.text((panel_x + 30, top + 100), "暂无可用记录。", font=small_font, fill="#697782")
        return

    maximum = max(values + [1])
    y_max = max(10, math.ceil(maximum * 1.16 / 10) * 10)
    plot_left, plot_right = panel_x + 92, panel_x + panel_width - 38
    plot_top, plot_bottom = top + 78, top + panel_height - 58
    tick_step = max(1, math.ceil(y_max / 4 / 5) * 5)
    for tick in range(0, y_max + 1, tick_step):
        ty = plot_bottom - (tick / y_max) * (plot_bottom - plot_top)
        draw.line((plot_left, ty, plot_right, ty), fill="#E6EBEF", width=2)
        draw.text((panel_x + 34, ty - 11), str(tick), font=small_font, fill="#73818B")

    count = len(entries)
    xs = [(plot_left + plot_right) / 2] if count == 1 else [
        plot_left + ((plot_right - plot_left) * i / (count - 1)) for i in range(count)
    ]
    ys = [plot_bottom - (value / y_max) * (plot_bottom - plot_top) for value in values]
    if len(xs) > 1:
        draw.line(list(zip(xs, ys)), fill=color, width=5, joint="curve")

    stride = max(1, math.ceil(count / 9))
    date_counts: dict[str, int] = {}
    for entry in entries:
        date_counts[entry["capture_date"]] = date_counts.get(entry["capture_date"], 0) + 1
    batch_indices = {
        index for index, entry in enumerate(entries)
        if date_counts.get(entry["capture_date"], 0) > 1 and "批次" in entry["session_label"]
    }
    for index, (x, point_y) in enumerate(zip(xs, ys)):
        reviewed = entries[index]["review_status"] == "已复核"
        radius = 8
        if reviewed:
            draw.ellipse((x-radius, point_y-radius, x+radius, point_y+radius), fill=color, outline="white", width=2)
        else:
            draw.ellipse((x-radius, point_y-radius, x+radius, point_y+radius), fill="white", outline=color, width=3)
        if count <= 15 or index % stride == 0 or index == count - 1 or index in batch_indices:
            label = point_labels[index]
            label_width = draw.textlength(label, font=small_font)
            label_x = max(plot_left, min(plot_right - label_width, x - label_width / 2))
            label_y = point_y - 30 if point_y - 34 >= plot_top else point_y + 12
            draw.text((label_x, label_y), label, font=small_font, fill="#273943")

    label_indices = set(range(0, count, stride)) | batch_indices | {count - 1}
    for index in sorted(label_indices):
        label = entries[index]["session_label"]
        batch_match = re.match(r"^(\d+/\d+ 批次\d+)", label)
        if batch_match:
            label = batch_match.group(1)
        if len(label) > 14:
            label = label[:13] + "…"
        label_width = draw.textlength(label, font=small_font)
        label_x = max(plot_left, min(plot_right - label_width, xs[index] - label_width / 2))
        draw.text((label_x, plot_bottom + 12), label, font=small_font, fill="#52616D")
    if (count - 1) % stride:
        label = entries[-1]["session_label"]
        label_width = draw.textlength(label, font=small_font)
        label_x = max(plot_left, min(plot_right - label_width, xs[-1] - label_width / 2))
        draw.text((label_x, plot_bottom + 12), label, font=small_font, fill="#52616D")


def render_history_chart(entries: list[dict[str, Any]], output_path: Path) -> None:
    width, height = 2100, 1880
    canvas = Image.new("RGB", (width, height), "#EEF2F5")
    draw = ImageDraw.Draw(canvas)
    title_font, subtitle_font = load_font(48, True), load_font(24)
    panel_title_font, small_font = load_font(29, True), load_font(17)
    outer = 52
    draw.text((outer, 34), "痤疮长期随访趋势", font=title_font, fill="#162431")
    if entries:
        draw.text(
            (outer, 100),
            f"共 {len(entries)} 个去重时点　·　{entries[0]['session_label']} 至 {entries[-1]['session_label']}　·　同一组三张照片自动去重",
            font=subtitle_font,
            fill="#52616D",
        )
    else:
        draw.text((outer, 100), "未找到可用历史结果。完成一次三视角复核后，趋势记录会从该时点开始积累。", font=subtitle_font, fill="#52616D")

    legend_y = 166
    draw.ellipse((outer, legend_y + 4, outer + 22, legend_y + 26), fill="#2878B5")
    draw.text((outer + 34, legend_y), "实心点：已复核", font=small_font, fill="#394955")
    draw.ellipse((outer + 220, legend_y + 4, outer + 242, legend_y + 26), outline="#687780", width=3, fill="white")
    draw.text((outer + 254, legend_y), "空心点：未复核或状态未知", font=small_font, fill="#394955")
    draw.text((outer + 590, legend_y), "活动框合计会受纳入照片数量和拍摄角度影响；不是全脸去重数。", font=small_font, fill="#687780")

    panel_x, panel_width = outer, width - 2 * outer
    total_top, total_height = 212, 290
    total_labels = [f"{entry['active_boxes']} ({entry['photo_count']}图)" for entry in entries]
    draw_trend_panel(
        draw, entries, [entry["active_boxes"] for entry in entries], total_labels,
        "全时段活动框历史", "点标签：活动框数（照片数）· 包含历史人工标注与近期复核结果",
        "#4C6B88", total_top, total_height, panel_x, panel_width, panel_title_font, small_font,
    )

    region_entries = [entry for entry in entries if entry["region_metrics_available"]]
    region_top = total_top + total_height + 24
    region_height, region_gap = 330, 20
    for region_index, region in enumerate(REGIONS):
        y = region_top + region_index * (region_height + region_gap)
        values = [entry["regions"][region]["burden_points"] for entry in region_entries]
        labels = [f"{entry['regions'][region]['burden_points']} ({entry['regions'][region]['active_total']})" for entry in region_entries]
        draw_trend_panel(
            draw, region_entries, values, labels,
            region.replace("视角", "") + " · 三视角负担分趋势",
            "点标签：负担分（活动框数）· 仅统计三视角复核结果",
            REGION_COLORS[region], y, region_height, panel_x, panel_width, panel_title_font, small_font,
        )

    footer_y = region_top + 3 * region_height + 2 * region_gap + 10
    draw.text((outer, footer_y), "最近时点明细（活动框是纳入照片的框数；早期记录可能只有部分视角）", font=panel_title_font, fill="#1D2A34")
    table_y = footer_y + 44
    row_h = 30
    headers = ((outer, "时点"), (outer + 520, "纳入照片活动框"), (outer + 870, "照片数"), (outer + 1080, "来源"), (outer + 1430, "复核状态"))
    for x, label in headers:
        draw.text((x, table_y), label, font=small_font, fill="#53616B")
    for index, entry in enumerate(entries[-5:]):
        row_y = table_y + 26 + index * row_h
        values = (
            (outer, entry["session_label"]),
            (outer + 520, str(entry["active_boxes"])),
            (outer + 870, str(entry["photo_count"])),
            (outer + 1080, entry["source_type"]),
            (outer + 1430, entry["review_status"]),
        )
        for x, value in values:
            draw.text((x, row_y), value, font=small_font, fill="#263641")
    canvas.save(output_path)


def write_history_files(runs_dir: Path, output_dir: Path, extra_current: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    entries = collect_history(runs_dir, extra_current=extra_current)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "longitudinal.csv"
    rows = history_csv_rows(entries)
    metric_keys = (*COUNT_KEYS.values(), "active_total", "inflammatory_total", "nodule_IV", "burden_points")
    fields = [
        "session_label", "capture_date", "period", "review_status", "source_type", "source_run", "source_session",
        "photo_count", "active_boxes_across_photos", "region_metrics_available", "images",
        *[f"{region}_{key}" for region in REGIONS for key in metric_keys],
        *[f"三视角_{key}" for key in metric_keys],
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    render_history_chart(entries, output_dir / "longitudinal.png")
    return entries
