"""Persistent, human-confirmed lesion IDs for the local follow-up history."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


TRACK_FIELDS = [
    "track_id", "visit_id", "visit_label", "capture_date", "region", "lesion_id", "grade",
    "image_path", "x1", "y1", "x2", "y2", "observation_status", "matched_from_track_id",
    "candidate_id", "source_run",
]


def read_track_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def visit_identity(photos: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Use image bytes to keep a visit ID stable when a run is repeated."""
    digests = []
    paths: dict[str, str] = {}
    labels = []
    for photo in photos:
        region = str(photo.get("region", ""))
        path = Path(photo.get("source_path") or photo["path"])
        paths[region] = str(path)
        labels.append(path.name)
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests.append(f"{region}:{digest.hexdigest()}")
        except OSError:
            digests.append(f"{region}:{os.path.normcase(os.path.normpath(str(path)))}")
    visit_id = "visit:" + hashlib.sha256("|".join(digests).encode("utf-8")).hexdigest()
    label = " / ".join(labels)
    date_match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]?\d)[-_]?([0-3]?\d)(?!\d)", " ".join(labels))
    capture_date = ""
    if date_match:
        try:
            capture_date = datetime(*(int(part) for part in date_match.groups())).date().isoformat()
        except ValueError:
            pass
    return visit_id, label, capture_date


def lesion_key(visit_id: str, region: str, lesion_id: str) -> tuple[str, str, str]:
    return visit_id, region, lesion_id


def load_existing_track_ids(photos: list[dict[str, Any]], store_path: Path) -> None:
    """Annotate an already tracked baseline with its saved IDs before review UI opens."""
    if not photos:
        return
    visit_id, _label, _capture_date = visit_identity(photos)
    stored = read_track_rows(store_path)
    by_key = {
        lesion_key(row.get("visit_id", ""), row.get("region", ""), row.get("lesion_id", "")): row.get("track_id", "")
        for row in stored
    }
    for photo in photos:
        for detection in photo.get("detections", []):
            detection["track_id"] = by_key.get(
                lesion_key(visit_id, str(photo.get("region", "")), str(detection.get("lesion_id", ""))),
                "",
            )


def observation_row(
    track_id: str,
    visit_id: str,
    visit_label: str,
    capture_date: str,
    photo: dict[str, Any],
    detection: dict[str, Any],
    status: str,
    matched_from_track_id: str = "",
    candidate_id: str = "",
    source_run: str = "",
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "visit_id": visit_id,
        "visit_label": visit_label,
        "capture_date": capture_date,
        "region": str(photo.get("region", "")),
        "lesion_id": str(detection.get("lesion_id", "")),
        "grade": str(detection.get("grade", "")),
        "image_path": str(photo.get("source_path") or photo["path"]),
        "x1": detection.get("x1", ""), "y1": detection.get("y1", ""),
        "x2": detection.get("x2", ""), "y2": detection.get("y2", ""),
        "observation_status": status,
        "matched_from_track_id": matched_from_track_id,
        "candidate_id": candidate_id,
        "source_run": source_run,
    }


def assign_persistent_ids(
    previous: list[dict[str, Any]] | None,
    current: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    store_path: Path,
    current_run: str,
    previous_run: str = "",
) -> dict[str, Any]:
    """Apply unambiguous automatic links, assign new IDs, and return a local store snapshot."""
    stored = read_track_rows(store_path)
    stored_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    used_ids: set[str] = set()
    for row in stored:
        key = lesion_key(row.get("visit_id", ""), row.get("region", ""), row.get("lesion_id", ""))
        stored_by_key[key] = row
        if row.get("track_id"):
            used_ids.add(row["track_id"])
    numeric_ids = [int(match.group(1)) for value in used_ids if (match := re.fullmatch(r"L(\d+)", value))]
    next_number = max(numeric_ids, default=0) + 1

    def new_id() -> str:
        nonlocal next_number
        value = f"L{next_number:05d}"
        next_number += 1
        used_ids.add(value)
        return value

    current_visit, current_label, current_date = visit_identity(current)
    if previous:
        previous_visit, previous_label, previous_date = visit_identity(previous)
    else:
        previous_visit, previous_label, previous_date = "", "", ""

    current_detection_keys = {
        (str(photo["region"]), str(detection["lesion_id"]))
        for photo in current for detection in photo["detections"]
    }
    stored = [
        row for row in stored
        if row.get("visit_id") != current_visit
        or (row.get("region", ""), row.get("lesion_id", "")) in current_detection_keys
    ]
    stored_by_key = {
        lesion_key(row.get("visit_id", ""), row.get("region", ""), row.get("lesion_id", "")): row
        for row in stored
    }

    new_store_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    def ensure_previous_tracks() -> dict[tuple[str, str], str]:
        mapping: dict[tuple[str, str], str] = {}
        if not previous:
            return mapping
        for photo in previous:
            for detection in photo["detections"]:
                key = lesion_key(previous_visit, str(photo["region"]), str(detection["lesion_id"]))
                row = stored_by_key.get(key)
                track_id = row.get("track_id", "") if row else ""
                if not track_id:
                    track_id = new_id()
                    row = observation_row(
                        track_id, previous_visit, previous_label, previous_date, photo, detection,
                        "seeded_from_selected_baseline", source_run=previous_run,
                    )
                    stored_by_key[key] = row
                    new_store_by_key[key] = row
                detection["track_id"] = track_id
                mapping[(str(photo["region"]), str(detection["lesion_id"]))] = track_id
        return mapping

    previous_tracks = ensure_previous_tracks()
    current_by_key = {
        (str(photo["region"]), str(detection["lesion_id"])): (photo, detection)
        for photo in current for detection in photo["detections"]
    }
    previous_by_key = {
        (str(photo["region"]), str(detection["lesion_id"])): (photo, detection)
        for photo in (previous or []) for detection in photo["detections"]
    }
    automatic_pairs = [
        {
            "region_index": int(candidate.get("region_index", -1)),
            "previous_lesion_id": str(candidate.get("previous_lesion_id", "")),
            "current_lesion_id": str(candidate.get("current_lesion_id", "")),
            "candidate_id": str(candidate.get("candidate_id", "")),
        }
        for candidate in candidates
    ]
    linked_candidate_ids: set[str] = set()
    current_status: dict[tuple[str, str], tuple[str, str, str]] = {}
    merged_tracks: dict[str, str] = {}

    for pair in automatic_pairs:
        region_index = int(pair.get("region_index", -1))
        if not previous or region_index < 0 or region_index >= len(previous) or region_index >= len(current):
            continue
        region = str(current[region_index]["region"])
        old_key = (region, str(pair.get("previous_lesion_id", "")))
        new_key = (region, str(pair.get("current_lesion_id", "")))
        if old_key not in previous_by_key or new_key not in current_by_key:
            continue
        old_track = previous_tracks.get(old_key, "")
        if not old_track:
            old_track = new_id()
            previous_tracks[old_key] = old_track
        current_visit_key = lesion_key(current_visit, region, new_key[1])
        existing_current = stored_by_key.get(current_visit_key)
        old_current_track = current_by_key[new_key][1].get("track_id", "") or (existing_current.get("track_id", "") if existing_current else "")
        if old_current_track and old_current_track != old_track:
            merged_tracks[old_current_track] = old_track
        photo, detection = current_by_key[new_key]
        detection["track_id"] = old_track
        previous_by_key[old_key][1]["temporal_status"] = "track_auto_matched"
        candidate_id = str(pair.get("candidate_id", ""))
        if candidate_id:
            linked_candidate_ids.add(candidate_id)
        detection["candidate_track_id"] = candidate_id
        detection["temporal_status"] = "track_auto_matched"
        current_status[new_key] = ("track_auto_matched", old_track, candidate_id)

    # An automatic link merges the current provisional identity into the older one everywhere.
    if merged_tracks:
        for row in stored:
            while row.get("track_id") in merged_tracks:
                row["track_id"] = merged_tracks[row["track_id"]]
        for key, row in list(stored_by_key.items()):
            while row.get("track_id") in merged_tracks:
                row["track_id"] = merged_tracks[row["track_id"]]
        for key, row in list(new_store_by_key.items()):
            while row.get("track_id") in merged_tracks:
                row["track_id"] = merged_tracks[row["track_id"]]

    candidate_by_current: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        region = str(candidate.get("region", ""))
        current_key = (region, str(candidate.get("current_lesion_id", "")))
        candidate_by_current.setdefault(current_key, candidate)

    current_records: list[dict[str, Any]] = []
    for photo in current:
        for detection in photo["detections"]:
            region = str(photo["region"])
            lesion_id = str(detection["lesion_id"])
            key = (region, lesion_id)
            visit_key = lesion_key(current_visit, region, lesion_id)
            prior_row = stored_by_key.get(visit_key)
            linked_status = current_status.get(key)
            if linked_status:
                status, matched_from, candidate_id = linked_status
            elif prior_row:
                # Re-running the exact same photo set preserves previously assigned IDs.
                detection["track_id"] = prior_row.get("track_id", "")
                status, matched_from, candidate_id = prior_row.get("observation_status", "existing_visit"), prior_row.get("matched_from_track_id", ""), prior_row.get("candidate_id", "")
            else:
                track_id = new_id()
                detection["track_id"] = track_id
                candidate = candidate_by_current.get(key)
                candidate_id = str(candidate.get("candidate_id", "")) if candidate else ""
                if not previous:
                    status = "baseline_first_observation"
                else:
                    status = "unmatched_new_track"
                matched_from = ""
            detection.setdefault("track_id", "")
            if detection.get("track_id"):
                if linked_status:
                    detection["track_id"] = linked_status[1]
                detection["candidate_track_id"] = candidate_id or detection.get("candidate_track_id", "")
                detection["temporal_status"] = status
                row = observation_row(
                    str(detection["track_id"]), current_visit, current_label, current_date, photo, detection,
                    status, matched_from_track_id=matched_from, candidate_id=candidate_id, source_run=current_run,
                )
                current_records.append(row)
                new_store_by_key[visit_key] = row
                stored_by_key[visit_key] = row

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        region = str(candidate.get("region", ""))
        current_detection = current_by_key.get((region, str(candidate.get("current_lesion_id", ""))))
        candidate["tracking_status"] = "auto_linked" if candidate_id in linked_candidate_ids else "not_linked"
        candidate["track_id"] = current_detection[1].get("track_id", "") if current_detection else ""

    # Merge updated rows with the stored history; one observation per visit/region/detection.
    merged_rows = {lesion_key(row.get("visit_id", ""), row.get("region", ""), row.get("lesion_id", "")): row for row in stored}
    merged_rows.update({lesion_key(row["visit_id"], row["region"], row["lesion_id"]): row for row in new_store_by_key.values()})
    final_rows = list(merged_rows.values())
    final_rows.sort(key=lambda row: (row.get("capture_date", ""), row.get("visit_id", ""), row.get("region", ""), row.get("track_id", "")))
    summary: dict[str, dict[str, Any]] = {}
    for row in final_rows:
        item = summary.setdefault(row["track_id"], {
            "track_id": row["track_id"], "region": row["region"], "first_capture_date": row.get("capture_date", ""),
            "last_capture_date": row.get("capture_date", ""), "observation_count": 0, "grades_over_time": [],
        })
        item["observation_count"] += 1
        if row.get("capture_date"):
            if not item["first_capture_date"] or row["capture_date"] < item["first_capture_date"]:
                item["first_capture_date"] = row["capture_date"]
            if row["capture_date"] > item["last_capture_date"]:
                item["last_capture_date"] = row["capture_date"]
        item["grades_over_time"].append(f"{row.get('capture_date') or row.get('visit_id', '')}:{row.get('grade', '')}")
    summary_rows = []
    for track_id in sorted(summary):
        item = summary[track_id]
        item["grades_over_time"] = " | ".join(item["grades_over_time"])
        summary_rows.append(item)
    return {
        "store_path": store_path,
        "observations": final_rows,
        "new_observations": current_records,
        "summary_rows": summary_rows,
        "automatic_link_count": len(current_status),
        "current_track_count": len({row.get("track_id", "") for row in current_records}),
        "visit_id": current_visit,
        "visit_label": current_label,
    }


def save_track_store(result: dict[str, Any]) -> None:
    store_path = Path(result["store_path"])
    atomic_write_csv(store_path, result["observations"], TRACK_FIELDS)
    summary_path = store_path.with_name("track_summary.csv")
    atomic_write_csv(summary_path, result["summary_rows"], [
        "track_id", "region", "first_capture_date", "last_capture_date", "observation_count", "grades_over_time",
    ])


def write_run_track_files(result: dict[str, Any], output_dir: Path) -> None:
    atomic_write_csv(output_dir / "track_observations.csv", result["new_observations"], TRACK_FIELDS)
    atomic_write_csv(output_dir / "track_summary.csv", result["summary_rows"], [
        "track_id", "region", "first_capture_date", "last_capture_date", "observation_count", "grades_over_time",
    ])
