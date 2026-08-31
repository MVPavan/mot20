from __future__ import annotations

import configparser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sequence:
    name: str
    root: Path
    image_dir: Path
    length: int
    width: int
    height: int

    def image_names(self, start_frame: int, stop_frame: int) -> list[str]:
        if not 1 <= start_frame <= stop_frame <= self.length:
            raise ValueError(f"invalid frame range for {self.name}: {start_frame}-{stop_frame}")
        return [f"{frame:06d}.jpg" for frame in range(start_frame, stop_frame + 1)]


def read_sequences(dataset_root: Path) -> list[Sequence]:
    """Read and validate MOT20 test sequence contracts without mutating files."""
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    sequences: list[Sequence] = []
    for sequence_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        config_path = sequence_root / "seqinfo.ini"
        if not config_path.is_file():
            continue
        parser = configparser.ConfigParser()
        parser.read(config_path)
        if "Sequence" not in parser:
            raise ValueError(f"missing [Sequence] in {config_path}")
        values = parser["Sequence"]
        try:
            name = values["name"]
            image_dir = sequence_root / values["imDir"]
            length = int(values["seqLength"])
            width = int(values["imWidth"])
            height = int(values["imHeight"])
            extension = values["imExt"]
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid sequence metadata: {config_path}") from error
        if name != sequence_root.name or extension.lower() != ".jpg" or length < 1 or width < 1 or height < 1:
            raise ValueError(f"unsupported MOT20 sequence contract: {config_path}")
        names = sorted(path.name for path in image_dir.glob("*.jpg")) if image_dir.is_dir() else []
        expected = [f"{frame:06d}.jpg" for frame in range(1, length + 1)]
        if names != expected:
            raise ValueError(f"image count or names do not match seqinfo.ini: {sequence_root}")
        sequences.append(Sequence(name, sequence_root, image_dir, length, width, height))
    if not sequences:
        raise ValueError(f"no MOT20 sequences found under {dataset_root}")
    return sequences


def build_task_plan(sequences: list[Sequence], reviewers: list[str], *, max_images_per_task: int) -> dict[str, Any]:
    """Create a deterministic, balanced, contiguous-frame assignment plan."""
    if not reviewers or len(set(reviewers)) != len(reviewers) or any(not name for name in reviewers):
        raise ValueError("reviewers must be non-empty and unique")
    if not 1 <= max_images_per_task <= 1000:
        raise ValueError("max_images_per_task must be between 1 and 1000")
    candidates: list[tuple[str, int, int]] = []
    for sequence in sequences:
        for start in range(1, sequence.length + 1, max_images_per_task):
            candidates.append((sequence.name, start, min(start + max_images_per_task - 1, sequence.length)))
    loads = {reviewer: 0 for reviewer in reviewers}
    assignments: list[dict[str, Any]] = []
    for sequence_name, start, stop in sorted(candidates, key=lambda item: (-(item[2] - item[1] + 1), *item)):
        assignee = min(reviewers, key=lambda reviewer: (loads[reviewer], reviewers.index(reviewer)))
        loads[assignee] += stop - start + 1
        assignments.append({
            "task_name": f"mot20-test__{sequence_name}__frames-{start:06d}-{stop:06d}",
            "sequence": sequence_name,
            "start_frame": start,
            "stop_frame": stop,
            "assignee": assignee,
        })
    assignments.sort(key=lambda item: (item["sequence"], item["start_frame"]))
    return {"format": "mot20.cvat.assignments.v1", "max_images_per_task": max_images_per_task, "assignments": assignments}


def validate_task_plan(plan: dict[str, Any], sequences: list[Sequence]) -> None:
    """Reject incomplete, overlapping, or identity-changing planned tasks."""
    if plan.get("format") != "mot20.cvat.assignments.v1" or not isinstance(plan.get("assignments"), list):
        raise ValueError("unsupported assignment plan")
    indexed = {sequence.name: sequence for sequence in sequences}
    covered: dict[str, set[int]] = defaultdict(set)
    task_names: set[str] = set()
    for assignment in plan["assignments"]:
        if not isinstance(assignment, dict):
            raise ValueError("assignment must be an object")
        try:
            task_name = assignment["task_name"]
            sequence = indexed[assignment["sequence"]]
            start = int(assignment["start_frame"])
            stop = int(assignment["stop_frame"])
            assignee = assignment["assignee"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("assignment is incomplete") from error
        if not isinstance(task_name, str) or not task_name or task_name in task_names or not isinstance(assignee, str) or not assignee:
            raise ValueError("assignment task names and assignees must be unique/non-empty")
        sequence.image_names(start, stop)
        frames = set(range(start, stop + 1))
        if covered[sequence.name] & frames:
            raise ValueError(f"overlapping assignment frames: {sequence.name}")
        covered[sequence.name].update(frames)
        task_names.add(task_name)
    for sequence in sequences:
        expected = set(range(1, sequence.length + 1))
        if covered[sequence.name] != expected:
            raise ValueError(f"assignment plan does not cover {sequence.name} exactly")


def parse_mot_rows(text: str) -> list[dict[str, float | int]]:
    """Parse MOTChallenge rows, preserving track IDs and geometry verbatim."""
    rows: list[dict[str, float | int]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        columns = [column.strip() for column in raw.split(",")]
        if len(columns) < 6:
            raise ValueError(f"MOT row {line_number} has fewer than six columns")
        try:
            frame, track_id = int(columns[0]), int(columns[1])
            left, top, width, height = (float(value) for value in columns[2:6])
        except ValueError as error:
            raise ValueError(f"invalid MOT row {line_number}") from error
        if frame < 1 or track_id < 1 or width <= 0 or height <= 0:
            raise ValueError(f"invalid MOT geometry or identity at row {line_number}")
        rows.append({"frame": frame, "track_id": track_id, "left": left, "top": top, "width": width, "height": height})
    if not rows:
        raise ValueError("annotation file has no MOT rows")
    return rows


def build_tracks(rows: list[dict[str, float | int]], *, label_id: int, start_frame: int, stop_frame: int) -> list[dict[str, Any]]:
    """Convert one task's MOT rows into CVAT tracks with task-local frame indexes."""
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frame = int(row["frame"])
        if not start_frame <= frame <= stop_frame:
            continue
        left, top, width, height = (float(row[name]) for name in ("left", "top", "width", "height"))
        tracks[int(row["track_id"])].append({
            "type": "rectangle", "frame": frame - start_frame, "outside": False,
            "occluded": False, "z_order": 0, "rotation": 0,
            "points": [left, top, left + width, top + height], "attributes": [],
        })
    return [
        {"label_id": label_id, "frame": shapes[0]["frame"], "group": 0, "source": "manual", "attributes": [], "shapes": shapes}
        for _, shapes in sorted(tracks.items())
    ]
