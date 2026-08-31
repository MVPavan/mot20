#!/usr/bin/env python3
"""Generate an ignored, reviewable MOT20 CVAT assignment plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mot20_cvat.contracts import build_task_plan, read_sequences, validate_task_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewers", required=True, help="comma-separated CVAT usernames")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/MOT20/test"))
    parser.add_argument("--max-images-per-task", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing local plan")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"output already exists (refusing to overwrite): {args.output}")
    reviewers = [value.strip() for value in args.reviewers.split(",") if value.strip()]
    sequences = read_sequences(args.dataset_root)
    plan = build_task_plan(sequences, reviewers, max_images_per_task=args.max_images_per_task)
    plan["dataset_root"] = str(args.dataset_root)
    validate_task_plan(plan, sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    totals = {reviewer: 0 for reviewer in reviewers}
    for assignment in plan["assignments"]:
        totals[assignment["assignee"]] += assignment["stop_frame"] - assignment["start_frame"] + 1
    print(json.dumps({"output": str(args.output), "tasks": len(plan["assignments"]), "frames_by_reviewer": totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
