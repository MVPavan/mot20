"""Stage-wise diagnosis of where a detection variant helps or hurts tracking.

A detector can improve detection metrics and still degrade tracking, so this
instruments BoostTrack's per-frame pipeline and records what each stage did:

1. ``raw``          detections handed to the tracker, below its own threshold
2. ``boosted``      after ``dlo_confidence_boost`` and ``duo_confidence_boost``
3. ``promoted``     boxes pushed from below ``det_thresh`` to above it by boosting
4. ``entering``     detections that survive the ``det_thresh`` cut
5. ``matched``      associated with an existing track
6. ``new``          unmatched detections, each of which can start a track
7. ``output``       boxes emitted after aspect-ratio and area filtering

The counts are written per frame so a single frame can be inspected, and
aggregated per sequence so the comparison is not anecdotal. Nothing here
changes tracker behaviour: every hook records and delegates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOSTTRACK_ROOT = REPO_ROOT / "repos" / "BoostTrack"

sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import TrackingArtifacts  # noqa: E402
from mot20_tracking.detections import read_detections  # noqa: E402
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import Combination  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tracking" / "scripts"))
from run_boosttrack import PrecomputedEmbedder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--reid", default="osnet-ain-msdc")
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--det-thresh", type=float, default=None)
    parser.add_argument("--sequences", nargs="*", default=None, help="subset of sequences")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    parser.add_argument("--output-prefix", required=True, help="path prefix for csv/json output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    combination = Combination(args.detector, args.reid, "diagnostic")
    sequences = read_split(args.split, repo_root=REPO_ROOT)
    if args.sequences:
        sequences = [s for s in sequences if s.name in set(args.sequences)]
        if not sequences:
            raise SystemExit("no matching sequences")

    os.chdir(BOOSTTRACK_ROOT)
    sys.path.insert(0, str(BOOSTTRACK_ROOT))
    import external  # noqa: F401
    import cv2
    import torch
    import utils
    from default_settings import GeneralSettings
    from tracker import boost_track as boost_track_module
    from tracker.boost_track import BoostTrack

    GeneralSettings.values["dataset"] = "mot20"
    GeneralSettings.values["test_dataset"] = False
    GeneralSettings.values["use_embedding"] = combination.uses_reid
    GeneralSettings.values["use_ecc"] = True
    if args.det_thresh is not None:
        GeneralSettings.values["det_thresh"] = args.det_thresh
        GeneralSettings.dataset_specific_settings["mot20"]["det_thresh"] = args.det_thresh
    det_thresh = GeneralSettings["det_thresh"]

    frame_state: dict[str, float | int] = {}

    original_dlo = BoostTrack.dlo_confidence_boost
    original_duo = BoostTrack.duo_confidence_boost
    original_associate = boost_track_module.associate

    def dlo_hook(self, detections, *rest):  # type: ignore[no-untyped-def]
        frame_state["scores_before_boost"] = detections[:, 4].copy() if detections.size else np.empty(0)
        return original_dlo(self, detections, *rest)

    def duo_hook(self, detections):  # type: ignore[no-untyped-def]
        result = original_duo(self, detections)
        frame_state["scores_after_boost"] = result[:, 4].copy() if result.size else np.empty(0)
        return result

    def associate_hook(dets, trks, *rest, **kwargs):  # type: ignore[no-untyped-def]
        matched, unmatched_dets, unmatched_trks, sym = original_associate(dets, trks, *rest, **kwargs)
        frame_state["matched"] = len(matched)
        frame_state["new"] = len(unmatched_dets)
        frame_state["unmatched_tracks"] = len(unmatched_trks)
        frame_state["live_tracks"] = len(trks)
        return matched, unmatched_dets, unmatched_trks, sym

    BoostTrack.dlo_confidence_boost = dlo_hook
    BoostTrack.duo_confidence_boost = duo_hook
    boost_track_module.associate = associate_hook

    rows = []
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    try:
        for sequence in sequences:
            detections = read_detections(
                artifacts.detections_file(args.detector, args.split, sequence.name),
                sequence.name,
                sequence.length,
            )
            tracker = BoostTrack(video_name=sequence.name)
            if combination.uses_reid:
                payload = np.load(
                    artifacts.embeddings_file(args.detector, args.reid, args.split, sequence.name)
                )
                tracker.embedder = PrecomputedEmbedder(detections.frames, payload["embeddings"])

            limit = min(sequence.length, args.max_frames or sequence.length)
            for frame_id in range(1, limit + 1):
                image = cv2.imread(str(sequence.frame_path(frame_id)))
                raw = detections.rows_for(frame_id)
                frame_state.clear()
                shape_tensor = torch.zeros((1, 3, image.shape[0], image.shape[1]))
                targets = tracker.update(
                    raw, shape_tensor, image, f"{sequence.name}:{frame_id}"
                )
                tlwhs, ids, _ = utils.filter_targets(
                    targets,
                    GeneralSettings["aspect_ratio_thresh"],
                    GeneralSettings["min_box_area"],
                )
                before = frame_state.get("scores_before_boost", np.empty(0))
                after = frame_state.get("scores_after_boost", np.empty(0))
                promoted = 0
                if isinstance(before, np.ndarray) and isinstance(after, np.ndarray) and before.size == after.size and before.size:
                    promoted = int(((before < det_thresh) & (after >= det_thresh)).sum())
                entering = int((after >= det_thresh).sum()) if isinstance(after, np.ndarray) and after.size else 0
                record = {
                    "sequence": sequence.name,
                    "frame": frame_id,
                    "raw": int(raw.shape[0]),
                    "raw_above_thresh": int((raw[:, 4] >= det_thresh).sum()) if raw.size else 0,
                    "entering": entering,
                    "promoted": promoted,
                    "live_tracks": int(frame_state.get("live_tracks", 0)),
                    "matched": int(frame_state.get("matched", 0)),
                    "new": int(frame_state.get("new", 0)),
                    "unmatched_tracks": int(frame_state.get("unmatched_tracks", 0)),
                    "output": len(tlwhs),
                }
                rows.append(record)
                bucket = totals[sequence.name]
                for key, value in record.items():
                    if key not in ("sequence", "frame"):
                        bucket[key] += value
                bucket["frames"] += 1
            tracker.dump_cache()
    finally:
        BoostTrack.dlo_confidence_boost = original_dlo
        BoostTrack.duo_confidence_boost = original_duo
        boost_track_module.associate = original_associate

    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = REPO_ROOT / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    combined: dict[str, float] = defaultdict(float)
    for bucket in totals.values():
        for key, value in bucket.items():
            combined[key] += value
    summary = {
        "format": "mot20.tracking.stage-diagnosis.v1",
        "detector_variant": args.detector,
        "reid_variant": args.reid,
        "split": args.split,
        "det_thresh": det_thresh,
        "per_sequence": {name: dict(bucket) for name, bucket in totals.items()},
        "combined": dict(combined),
        "per_frame_means": {
            key: round(value / max(combined["frames"], 1), 3)
            for key, value in combined.items()
            if key != "frames"
        },
        "frame_csv": str(csv_path.relative_to(REPO_ROOT)),
    }
    json_path = prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(summary["per_frame_means"], indent=2, sort_keys=True))
    print(f"\nframes {int(combined['frames'])} | csv {csv_path} | json {json_path}")


if __name__ == "__main__":
    main()
