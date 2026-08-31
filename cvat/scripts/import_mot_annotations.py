#!/usr/bin/env python3
"""Import supplied MOT-format tracks only after every target task passes preflight."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mot20_cvat.client import CvatClient, full_id  # noqa: E402
from mot20_cvat.contracts import build_tracks, parse_mot_rows, read_sequences, validate_task_plan  # noqa: E402


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def annotation_file(root: Path, sequence: str) -> Path:
    for candidate in (root / sequence / "gt" / "gt.txt", root / f"{sequence}.txt"):
        if candidate.is_file():
            return candidate
    raise ValueError(f"no annotation file for {sequence}; expected {root}/{sequence}/gt/gt.txt or {root}/{sequence}.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "config/project.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "config/assignments.json")
    parser.add_argument("--cvat-url", default=os.environ.get("CVAT_URL", "http://127.0.0.1:8082"))
    parser.add_argument("--admin-user", default=os.environ.get("CVAT_ADMIN_USER", "admin"))
    parser.add_argument("--admin-password-env", default="CVAT_ADMIN_PASSWORD")
    parser.add_argument("--replace-existing", action="store_true", help="required to replace any existing task annotations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project, plan = read_json(args.project), read_json(args.plan)
    sequences = read_sequences(Path(plan.get("dataset_root", "datasets/MOT20/test")))
    validate_task_plan(plan, sequences)
    rows = {sequence.name: parse_mot_rows(annotation_file(args.annotations_root, sequence.name).read_text()) for sequence in sequences}
    if args.dry_run:
        print(json.dumps({"tasks": len(plan["assignments"]), "annotation_rows": {name: len(value) for name, value in rows.items()}, "dry_run": True}, indent=2))
        return 0
    password = os.environ.get(args.admin_password_env)
    if not password:
        parser.error(f"environment variable is unset: {args.admin_password_env}")
    client = CvatClient(args.cvat_url, args.admin_user, password)
    cvat_project = client.ensure_project(project)
    project_id = full_id(cvat_project)
    labels = client.project_labels(project_id)
    pedestrian = next((label for label in labels if label.get("name") == "pedestrian" and label.get("type") == "rectangle"), None)
    if not pedestrian:
        raise RuntimeError("CVAT project has no pedestrian rectangle label")
    prepared = []
    for assignment in plan["assignments"]:
        task = client.get_task(assignment["task_name"], project_id)
        task_id = full_id(task)
        sequence = next(item for item in sequences if item.name == assignment["sequence"])
        expected_names = sequence.image_names(assignment["start_frame"], assignment["stop_frame"])
        if task.get("size") != len(expected_names) or client.task_frame_names(task_id) != expected_names:
            raise RuntimeError(f"task frame contract differs: {assignment['task_name']}")
        existing = client.annotations(task_id)
        has_annotations = any(existing.get(key) for key in ("tags", "shapes", "tracks"))
        if has_annotations and not args.replace_existing:
            raise RuntimeError(f"task already has annotations (refusing to replace): {assignment['task_name']}")
        tracks = build_tracks(rows[sequence.name], label_id=full_id(pedestrian), start_frame=assignment["start_frame"], stop_frame=assignment["stop_frame"])
        prepared.append((task_id, assignment["task_name"], {"version": 0, "tags": [], "shapes": [], "tracks": tracks}))
    for task_id, name, payload in prepared:
        client.replace_annotations(task_id, payload)
        print(json.dumps({"task_id": task_id, "task_name": name, "tracks": len(payload["tracks"]), "status": "imported"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
