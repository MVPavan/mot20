#!/usr/bin/env python3
"""Preflight then create or safely reuse one assigned CVAT job per MOT20 task range."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mot20_cvat.client import CvatClient, full_id  # noqa: E402
from mot20_cvat.contracts import read_sequences, validate_task_plan  # noqa: E402


def read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "config/project.json")
    parser.add_argument("--reviewers", type=Path, default=ROOT / "config/reviewers.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "config/assignments.json")
    parser.add_argument("--cvat-url", default=os.environ.get("CVAT_URL", "http://127.0.0.1:8082"))
    parser.add_argument("--admin-user", default=os.environ.get("CVAT_ADMIN_USER", "admin"))
    parser.add_argument("--admin-password-env", default="CVAT_ADMIN_PASSWORD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = read_object(args.project, "project config")
    plan = read_object(args.plan, "assignment plan")
    dataset_root = Path(plan.get("dataset_root", "datasets/MOT20/test"))
    sequences = read_sequences(dataset_root)
    validate_task_plan(plan, sequences)
    planned = {assignment["assignee"] for assignment in plan["assignments"]}
    indexed = {sequence.name: sequence for sequence in sequences}
    summary = {"project": project.get("name"), "tasks": len(plan["assignments"]), "frames": sum(sequence.length for sequence in sequences), "dry_run": args.dry_run}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    reviewers_config = read_object(args.reviewers, "reviewer config")
    reviewer_rows = reviewers_config.get("reviewers")
    if not isinstance(reviewer_rows, list):
        raise ValueError("reviewer config needs a reviewers list")
    reviewers = {row.get("username"): row for row in reviewer_rows if isinstance(row, dict) and isinstance(row.get("username"), str)}
    if planned != set(reviewers):
        raise ValueError("reviewer config usernames must exactly match plan assignees")
    password = os.environ.get(args.admin_password_env)
    if not password:
        parser.error(f"environment variable is unset: {args.admin_password_env}")
    client = CvatClient(args.cvat_url, args.admin_user, password)
    cvat_project = client.ensure_project(project)
    project_id = full_id(cvat_project)
    user_ids = {}
    for username in sorted(planned):
        user = client.find_user(username)
        if not user:
            raise ValueError(f"CVAT reviewer does not exist; run bootstrap_users.py first: {username}")
        user_ids[username] = full_id(user)
    results = []
    for assignment in plan["assignments"]:
        sequence = indexed[assignment["sequence"]]
        names = sequence.image_names(assignment["start_frame"], assignment["stop_frame"])
        server_files = [f"mot20-test/{sequence.name}/img1/{name}" for name in names]
        task_id, status = client.ensure_task(task_name=assignment["task_name"], project_id=project_id, image_names=names, server_files=server_files, assignee_id=user_ids[assignment["assignee"]])
        results.append({"task_name": assignment["task_name"], "task_id": task_id, "assignee": assignment["assignee"], "status": status})
    summary["project_id"] = project_id
    summary["results"] = results
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
