#!/usr/bin/env python3
"""Provision and verify the isolated ByteTrack-X MOT20-06/08 CVAT project."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CVAT_ROOT = REPO_ROOT / "cvat"
sys.path.insert(0, str(CVAT_ROOT))

from mot20_cvat.client import CvatClient, full_id, shared_file_fields, task_data_form  # noqa: E402
from mot20_cvat.contracts import read_sequences  # noqa: E402
from bytetrack_x_import import annotation_payload, read_yolo_shapes  # noqa: E402


REQUIRED_SEQUENCES = {"MOT20-06", "MOT20-08"}
REQUIRED_REVIEWERS = {"calanit", "tamar", "haim", "ran", "tamir", "yohai", "pavan", "deepak", "sree", "sathish", "raj"}
EXPECTED_FRAMES_BY_REVIEWER = {
    "calanit": 100,
    "tamar": 202,
    "haim": 100,
    "ran": 202,
    "tamir": 100,
    "yohai": 202,
    "pavan": 202,
    "deepak": 100,
    "sree": 202,
    "sathish": 202,
    "raj": 202,
}


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def validate_plan(plan: dict[str, Any], sequences: dict[str, Any]) -> list[dict[str, Any]]:
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 12:
        raise ValueError("debug plan must contain the expected twelve assignments")
    rows: list[dict[str, Any]] = []
    task_names: set[str] = set()
    assignees: set[str] = set()
    covered = {name: set() for name in REQUIRED_SEQUENCES}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("debug plan assignment must be an object")
        sequence = assignment.get("sequence")
        task_name = assignment.get("task_name")
        assignee = assignment.get("assignee")
        start_frame = assignment.get("start_frame")
        stop_frame = assignment.get("stop_frame")
        if not all(isinstance(value, str) and value for value in (sequence, task_name, assignee)) or not isinstance(start_frame, int) or not isinstance(stop_frame, int):
            raise ValueError("debug plan assignment is incomplete")
        if sequence not in REQUIRED_SEQUENCES or sequence not in sequences or task_name in task_names or assignee not in REQUIRED_REVIEWERS:
            raise ValueError(f"invalid debug sequence: {sequence}")
        if not 1 <= start_frame <= stop_frame <= sequences[sequence].length:
            raise ValueError(f"invalid debug frame range: {sequence} {start_frame}-{stop_frame}")
        frames = set(range(start_frame, stop_frame + 1))
        if covered[sequence] & frames:
            raise ValueError(f"overlapping debug frame range: {sequence}")
        covered[sequence].update(frames)
        task_names.add(task_name)
        assignees.add(assignee)
        rows.append({"sequence": sequence, "start_frame": start_frame, "stop_frame": stop_frame, "task_name": task_name, "assignee": assignee})
    if assignees != REQUIRED_REVIEWERS:
        raise ValueError("debug plan reviewers must exactly match the configured reviewer group")
    loads = {reviewer: 0 for reviewer in REQUIRED_REVIEWERS}
    for row in rows:
        loads[row["assignee"]] += row["stop_frame"] - row["start_frame"] + 1
    if loads != EXPECTED_FRAMES_BY_REVIEWER:
        raise ValueError(f"debug plan frame loads differ: {loads}")
    for sequence in REQUIRED_SEQUENCES:
        if covered[sequence] != set(range(1, sequences[sequence].length + 1)):
            raise ValueError(f"debug plan does not cover {sequence} exactly")
    return sorted(rows, key=lambda item: (item["sequence"], item["start_frame"]))


def find_label_id(client: CvatClient, project_id: int) -> int:
    labels = client.project_labels(project_id)
    matches = [label for label in labels if label.get("name") == "pedestrian" and label.get("type") == "rectangle"]
    if len(matches) != 1:
        raise RuntimeError("project must contain exactly one pedestrian rectangle label")
    return full_id(matches[0])


def ensure_debug_task(
    client: CvatClient,
    *,
    task_name: str,
    project_id: int,
    image_names: list[str],
    server_files: list[str],
    assignee_id: int,
) -> tuple[int, str]:
    """Repair only the blank task created by the pre-fix shared-file request."""
    existing = client._find("tasks", task_name, project_id=project_id)
    if not existing:
        return client.ensure_task(
            task_name=task_name,
            project_id=project_id,
            image_names=image_names,
            server_files=server_files,
            assignee_id=assignee_id,
        )
    task_id = full_id(existing)
    task = client.get_task(task_name, project_id)
    jobs = client._list("jobs", task_id=task_id)
    if (
        task.get("name") != task_name
        or task.get("project_id") != project_id
        or task.get("size") is not None
        or jobs
    ):
        raise RuntimeError(f"existing task is not the known blank task: {task_name}")
    client._check(
        client.session.post(
            f"{client.base}/api/tasks/{task_id}/data",
            data=task_data_form(),
            files=shared_file_fields(server_files),
            headers={"Upload-Start": "true", "Upload-Finish": "true"},
            timeout=client.timeout,
        ),
        f"repair task {task_name} images",
    )
    client._wait_for_data(task_id)
    jobs = client._list("jobs", task_id=task_id)
    if len(jobs) != 1:
        raise RuntimeError(f"repaired task must have exactly one job: {task_name}")
    client._check(
        client.session.patch(
            f"{client.base}/api/jobs/{full_id(jobs[0])}",
            json={"assignee": assignee_id},
            timeout=client.timeout,
        ),
        f"assign repaired task {task_name}",
    )
    client._verify_task(task_id, task_name, project_id, image_names, assignee_id)
    return task_id, "repaired"


def remove_unreviewed_project_tasks(client: CvatClient, *, project_id: int, project_name: str) -> list[int]:
    """Delete project tasks only if every one remains untouched model output."""
    removed: list[int] = []
    tasks = client._list("tasks", project_id=project_id)
    if not tasks:
        return removed
    for task in tasks:
        task_id = full_id(task)
        task_name = task.get("name")
        if not isinstance(task_name, str) or not task_name.startswith(f"{project_name}__"):
            raise RuntimeError(f"project task is outside the ByteTrack-X replacement scope: {task_id}")
        jobs = client._list("jobs", task_id=task_id)
        if len(jobs) != 1 or jobs[0].get("state") != "new" or jobs[0].get("stage") != "annotation":
            raise RuntimeError(f"project task job is no longer untouched: {task_name}")
        annotations = client.annotations(task_id)
        if annotations.get("tags") or annotations.get("tracks") or any(shape.get("source") != "semi-auto" for shape in annotations.get("shapes", [])):
            raise RuntimeError(f"project task annotations are no longer untouched: {task_name}")
        client._check(client.session.delete(f"{client.base}/api/tasks/{task_id}", timeout=client.timeout), f"remove unreviewed task {task_name}")
        removed.append(task_id)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=HERE / "project.json")
    parser.add_argument("--plan", type=Path, default=HERE / "project-plan.json")
    parser.add_argument("--cvat-url", default=os.environ.get("CVAT_URL", "http://127.0.0.1:8082"))
    parser.add_argument("--admin-user", default=os.environ.get("CVAT_ADMIN_USER", "admin"))
    parser.add_argument("--admin-password-env", default="CVAT_ADMIN_PASSWORD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-unreviewed-project-tasks", action="store_true", help="remove only untouched model-only tasks in this project before seeding the updated plan")
    args = parser.parse_args()

    project = read_object(args.project, "project config")
    plan = read_object(args.plan, "debug plan")
    dataset_root = REPO_ROOT / str(plan.get("dataset_root", ""))
    labels_root = REPO_ROOT / str(plan.get("labels_root", ""))
    sequences = {sequence.name: sequence for sequence in read_sequences(dataset_root)}
    assignments = validate_plan(plan, sequences)
    source_shapes = {
        assignment["task_name"]: read_yolo_shapes(
            labels_root / assignment["sequence"],
            sequence_length=sequences[assignment["sequence"]].length,
            image_width=sequences[assignment["sequence"]].width,
            image_height=sequences[assignment["sequence"]].height,
            label_id=0,
            start_frame=assignment["start_frame"],
            stop_frame=assignment["stop_frame"],
        )
        for assignment in assignments
    }
    summary: dict[str, Any] = {
        "project": project.get("name"),
        "dry_run": args.dry_run,
        "tasks": [
            {
                "task_name": assignment["task_name"],
                "sequence": assignment["sequence"],
                "start_frame": assignment["start_frame"],
                "stop_frame": assignment["stop_frame"],
                "frames": assignment["stop_frame"] - assignment["start_frame"] + 1,
                "source_shapes": len(source_shapes[assignment["task_name"]]),
                "assignee": assignment["assignee"],
            }
            for assignment in assignments
        ],
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    password = os.environ.get(args.admin_password_env)
    if not password:
        parser.error(f"environment variable is unset: {args.admin_password_env}")
    client = CvatClient(args.cvat_url, args.admin_user, password, timeout=(10.0, 900.0))
    cvat_project = client.ensure_project(project)
    project_id = full_id(cvat_project)
    label_id = find_label_id(client, project_id)
    if args.replace_unreviewed_project_tasks:
        summary["removed_task_ids"] = remove_unreviewed_project_tasks(
            client,
            project_id=project_id,
            project_name=str(project["name"]),
        )
    results = []
    for assignment in assignments:
        sequence = sequences[assignment["sequence"]]
        user = client.find_user(assignment["assignee"])
        if not user:
            raise RuntimeError(f"CVAT reviewer does not exist: {assignment['assignee']}")
        names = sequence.image_names(assignment["start_frame"], assignment["stop_frame"])
        server_files = [f"mot20-test/{sequence.name}/img1/{name}" for name in names]
        task_id, status = ensure_debug_task(
            client,
            task_name=assignment["task_name"],
            project_id=project_id,
            image_names=names,
            server_files=server_files,
            assignee_id=full_id(user),
        )
        existing = client.annotations(task_id)
        if any(existing.get(key) for key in ("tags", "shapes", "tracks")):
            raise RuntimeError(f"refusing to replace non-empty task annotations: {assignment['task_name']}")
        shapes = [dict(shape, label_id=label_id) for shape in source_shapes[assignment["task_name"]]]
        client.replace_annotations(task_id, annotation_payload(shapes))
        imported = client.annotations(task_id)
        if len(imported.get("shapes", [])) != len(shapes) or imported.get("tracks") or imported.get("tags"):
            raise RuntimeError(f"annotation verification failed: {assignment['task_name']}")
        results.append({
            "task_name": assignment["task_name"],
            "task_id": task_id,
            "status": status,
            "assignee": assignment["assignee"],
            "frames": len(names),
            "imported_shapes": len(shapes),
        })
    summary.update({"project_id": project_id, "results": results})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
