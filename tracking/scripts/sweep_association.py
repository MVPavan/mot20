"""Run immutable BoostTrack++ association sweep points and compare their metrics.

Each point is a comma-separated list of upstream ``NAME=VALUE`` overrides.
The point's canonical spelling is hashed into its tracker slug, while the L3
manifest written by ``run_boosttrack.py`` retains the human-readable values.
This keeps artifact paths valid and stable without losing experiment context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tracking" / "scripts" / "run_boosttrack.py"
EVALUATOR = REPO_ROOT / "tracking" / "scripts" / "evaluate_tracks.py"

sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import TrackingArtifacts  # noqa: E402
from mot20_tracking.variants import Combination, validate_slug  # noqa: E402

METRICS = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW")


@dataclass(frozen=True)
class SweepPoint:
    """One canonicalized association configuration and its immutable slug."""

    overrides: tuple[str, ...]
    tracker: str

    @property
    def display(self) -> str:
        """Return the deterministic, human-readable point representation."""
        return ",".join(self.overrides)


def parse_point(raw_point: str, base_tracker: str) -> SweepPoint:
    """Validate and canonicalize a comma-separated list of ``NAME=VALUE`` pairs.

    Value typing and upstream-setting validation deliberately remain in the
    runner, the one place that imports the vendored current settings. This
    driver only establishes stable point identity before dispatching it.
    """
    if not raw_point:
        raise ValueError("sweep point must not be empty")

    overrides: list[str] = []
    names: set[str] = set()
    for raw_override in raw_point.split(","):
        if raw_override.count("=") != 1:
            raise ValueError(f"point override must be NAME=VALUE, not {raw_override!r}")
        name, value = raw_override.split("=", 1)
        if not name or not value:
            raise ValueError(f"point override must be NAME=VALUE, not {raw_override!r}")
        if name in names:
            raise ValueError(f"point specifies {name!r} more than once")
        names.add(name)
        overrides.append(f"{name}={value}")

    canonical = tuple(sorted(overrides))
    digest = hashlib.sha256(",".join(canonical).encode("utf-8")).hexdigest()[:12]
    tracker = f"{base_tracker}-assoc-{digest}"
    validate_slug(tracker, "derived tracker")
    return SweepPoint(overrides=canonical, tracker=tracker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, help="detection variant slug")
    parser.add_argument("--reid", required=True, help="ReID variant slug, or 'none'")
    parser.add_argument("--base-tracker", required=True, help="base tracker slug for derived variants")
    parser.add_argument(
        "--point",
        action="append",
        required=True,
        metavar="NAME=VALUE,...",
        help="repeatable association point; values are validated by run_boosttrack.py",
    )
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    parser.add_argument("--cores", type=int, default=8, help="TrackEval worker count")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print derived slugs and subprocess argument lists without launching either job",
    )
    return parser.parse_args()


def runner_command(args: argparse.Namespace, point: SweepPoint) -> list[str]:
    """Build the tracking command as an argument list, never a shell string."""
    command = [
        sys.executable,
        str(RUNNER),
        "--detector",
        args.detector,
        "--reid",
        args.reid,
        "--tracker",
        point.tracker,
        "--split",
        args.split,
        "--artifact-root",
        args.artifact_root,
    ]
    for override in point.overrides:
        command.extend(("--set", override))
    return command


def evaluator_command(args: argparse.Namespace, combination: Combination) -> list[str]:
    """Build the TrackEval wrapper command as an argument list."""
    return [
        sys.executable,
        str(EVALUATOR),
        "--combination",
        str(combination),
        "--split",
        args.split,
        "--artifact-root",
        args.artifact_root,
        "--cores",
        str(args.cores),
    ]


def load_metrics(manifest_path: Path) -> dict[str, float]:
    """Read the seven reported metrics from an existing L4 manifest."""
    with manifest_path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics manifest has no metrics object: {manifest_path}")
    missing = [name for name in METRICS if name not in metrics]
    if missing:
        raise ValueError(f"metrics manifest lacks {', '.join(missing)}: {manifest_path}")
    return {name: float(metrics[name]) for name in METRICS}


def print_summary(rows: list[tuple[SweepPoint, str, dict[str, float] | None]]) -> None:
    """Print the requested metrics for completed or skipped points."""
    headers = ("tracker", "point", "status", *METRICS)
    print("\n" + " | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for point, status, metrics in rows:
        values = [point.tracker, point.display, status]
        for name in METRICS:
            if metrics is None:
                values.append("-")
            elif name == "IDSW":
                values.append(str(round(metrics[name])))
            else:
                values.append(f"{metrics[name]:.2f}")
        print(" | ".join(values))


def run_command(command: list[str]) -> bool:
    """Run one command, preserving its output and returning success status."""
    print("$ " + shlex.join(command))
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode == 0


def main() -> None:
    args = parse_args()
    validate_slug(args.detector, "detector")
    validate_slug(args.reid, "reid")
    validate_slug(args.base_tracker, "base tracker")
    if args.cores < 1:
        raise SystemExit("--cores must be at least 1")

    try:
        points = [parse_point(raw_point, args.base_tracker) for raw_point in args.point]
    except ValueError as error:
        raise SystemExit(f"invalid --point: {error}") from error
    if len({point.tracker for point in points}) != len(points):
        raise SystemExit("duplicate sweep points resolve to the same tracker slug")

    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    rows: list[tuple[SweepPoint, str, dict[str, float] | None]] = []
    failed = False
    for point in points:
        combination = Combination(args.detector, args.reid, point.tracker)
        metrics_manifest = artifacts.manifest_path(artifacts.metrics_dir(combination, args.split))
        tracks_manifest = artifacts.manifest_path(artifacts.tracks_dir(combination, args.split))

        if metrics_manifest.exists():
            print(f"skip {point.tracker}: metrics manifest exists at {metrics_manifest}")
            rows.append((point, "skipped", load_metrics(metrics_manifest)))
            continue

        if args.dry_run:
            print(f"dry run {point.tracker}: {point.display}")
            if tracks_manifest.exists():
                print("  L3 manifest exists; would run evaluator only")
            else:
                print("  $ " + shlex.join(runner_command(args, point)))
            print("  $ " + shlex.join(evaluator_command(args, combination)))
            rows.append((point, "dry-run", None))
            continue

        if not tracks_manifest.exists() and not run_command(runner_command(args, point)):
            rows.append((point, "tracking-failed", None))
            failed = True
            continue
        if not tracks_manifest.exists():
            print(f"tracking produced no manifest for {point.tracker}")
            rows.append((point, "tracking-failed", None))
            failed = True
            continue
        if not run_command(evaluator_command(args, combination)):
            rows.append((point, "evaluation-failed", None))
            failed = True
            continue
        if not metrics_manifest.exists():
            print(f"evaluation produced no manifest for {point.tracker}")
            rows.append((point, "evaluation-failed", None))
            failed = True
            continue
        rows.append((point, "completed", load_metrics(metrics_manifest)))

    print_summary(rows)
    if failed:
        raise SystemExit("one or more sweep points failed")


if __name__ == "__main__":
    main()
