"""Phase 8: evaluate L3 tracks with the vendored TrackEval.

TrackEval expects a fixed directory layout
(``<trackers>/<BENCHMARK>-<SPLIT>/<tracker>/data/<sequence>.txt``) that differs
from the combination-keyed artifact tree, so a thin staging tree of symlinks is
built per evaluation rather than reorganising the artifact store around one
evaluator.

Ground truth is BoostTrack's vendored ``results/gt/MOT20-val``, which is
span-identical to ``datasets/val_half``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOSTTRACK_ROOT = REPO_ROOT / "repos" / "BoostTrack"

sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import METRICS_FORMAT, TrackingArtifacts  # noqa: E402
from mot20_tracking.variants import Combination  # noqa: E402

HEADLINE = ("HOTA", "MOTA", "IDF1", "IDSW", "AssA", "DetA", "MT", "ML", "FP", "FN", "Frag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combination", required=True, help="det__reid__tracker")
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--benchmark", default="MOT20")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combination = Combination.parse(args.combination)
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)

    tracks_dir = artifacts.tracks_dir(combination, args.split)
    if not artifacts.manifest_path(tracks_dir).exists():
        raise SystemExit(f"tracks not produced: {tracks_dir}")
    level_dir = artifacts.metrics_dir(combination, args.split)
    if artifacts.manifest_path(level_dir).exists():
        raise SystemExit(f"metrics already produced: {level_dir}")

    # TrackEval's val split for MOT20 is BoostTrack's vendored half-val ground
    # truth, which matches datasets/val_half in span and frame numbering.
    eval_split = "val"
    staging = level_dir / "trackeval"
    tracker_dir = staging / f"{args.benchmark}-{eval_split}" / str(combination) / "data"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    sequences = []
    for source in sorted(tracks_dir.glob("*.txt")):
        link = tracker_dir / source.name
        if not link.exists():
            link.symlink_to(source.resolve())
        sequences.append(source.stem)
    if not sequences:
        raise SystemExit(f"no track files under {tracks_dir}")

    os.chdir(BOOSTTRACK_ROOT)
    command = [
        sys.executable,
        "external/TrackEval/scripts/run_mot_challenge.py",
        "--SPLIT_TO_EVAL", eval_split,
        "--GT_FOLDER", "results/gt/",
        "--TRACKERS_FOLDER", str(staging.resolve()),
        "--BENCHMARK", args.benchmark,
        "--TRACKERS_TO_EVAL", str(combination),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", str(args.cores),
        "--PRINT_CONFIG", "False",
    ]
    print(" ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(completed.stdout[-4000:])
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr[-4000:])
        raise SystemExit(f"TrackEval failed with exit code {completed.returncode}")

    summary_path = (
        staging / f"{args.benchmark}-{eval_split}" / str(combination) / "pedestrian_summary.txt"
    )
    if not summary_path.is_file():
        raise SystemExit(f"TrackEval produced no summary at {summary_path}")
    with summary_path.open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter=" "))
    metrics = {key: float(value) for key, value in zip(rows[0], rows[1]) if value}

    headline = {key: metrics[key] for key in HEADLINE if key in metrics}
    print("\n" + json.dumps(headline, indent=2, sort_keys=True))

    digest = artifacts.write_manifest(
        level_dir,
        {
            "format": METRICS_FORMAT,
            "combination": str(combination),
            "split": args.split,
            "benchmark": args.benchmark,
            "evaluator": "vendored TrackEval, run_mot_challenge.py",
            "ground_truth": "repos/BoostTrack/results/gt/MOT20-val",
            "sequences": sequences,
            "command": command,
            "headline": headline,
            "metrics": metrics,
        },
    )
    print(f"\nmanifest {artifacts.manifest_path(level_dir)} sha256 {digest}")


if __name__ == "__main__":
    main()
