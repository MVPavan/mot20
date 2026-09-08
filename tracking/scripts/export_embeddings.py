"""Phase 6: precompute L2 ReID embeddings for every exported detection.

Embeddings are extracted with BoostTrack's own ``EmbeddingComputer`` so the
crop, resize, colour conversion, and normalisation are identical to what the
tracker would have computed inline. Only the storage differs: vectors are
written to the L2 artifact store keyed by detection variant *and* ReID variant,
rather than to a cache keyed by sequence name alone.

Covering every exported detection, rather than only those above a tracker's
threshold, is what makes the level reusable: BoostTrack boosts low-confidence
detections above its cut before embedding them, so which detections a run needs
is not knowable in advance.

Rows are stored in the same order as the detection file, so row *i* of the
embedding matrix corresponds to row *i* of ``det.txt``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOSTTRACK_ROOT = REPO_ROOT / "repos" / "BoostTrack"

sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import (  # noqa: E402
    EMBEDDINGS_FORMAT,
    TrackingArtifacts,
    sha256_file,
)
from mot20_tracking.detections import read_detections  # noqa: E402
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import validate_slug  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, help="detection variant slug")
    parser.add_argument("--reid", required=True, help="reid variant slug")
    parser.add_argument("--split", default="val_half")
    parser.add_argument(
        "--reid-dataset",
        default="mot20",
        help="BoostTrack dataset key selecting the ReID model",
    )
    parser.add_argument(
        "--test-dataset",
        action="store_true",
        help="select the FastReID SBS model instead of the generic OSNet model",
    )
    parser.add_argument("--batch-frames", type=int, default=1)
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_slug(args.detector, "detector")
    validate_slug(args.reid, "reid")

    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    detections_dir = artifacts.detections_dir(args.detector, args.split)
    if not artifacts.manifest_path(detections_dir).exists():
        raise SystemExit(f"detection variant not exported: {detections_dir}")
    level_dir = artifacts.embeddings_dir(args.detector, args.reid, args.split)
    if artifacts.manifest_path(level_dir).exists():
        raise SystemExit(f"embeddings already exported: {level_dir}")

    # BoostTrack resolves weights and vendored packages relative to its own root.
    os.chdir(BOOSTTRACK_ROOT)
    sys.path.insert(0, str(BOOSTTRACK_ROOT))
    import external  # noqa: F401  (installs the vendored-package sys.path entry)
    import cv2
    from tracker.embedding import EmbeddingComputer

    computer = EmbeddingComputer(args.reid_dataset, args.test_dataset, True)
    # Never touch BoostTrack's sequence-keyed on-disk cache: it cannot
    # distinguish ReID models and would silently return another model's vectors.
    computer.cache = {}
    computer.cache_name = "__precompute__"

    sequences = read_split(args.split, repo_root=REPO_ROOT)
    entries = []
    started = time.perf_counter()
    for sequence in sequences:
        detection_file = artifacts.detections_file(args.detector, args.split, sequence.name)
        detections = read_detections(detection_file, sequence.name, sequence.length)
        vectors: list[np.ndarray] = []
        row_frames: list[int] = []
        for frame_id in range(1, sequence.length + 1):
            rows = detections.rows_for(frame_id)
            if rows.shape[0] == 0:
                continue
            image = cv2.imread(str(sequence.frame_path(frame_id)))
            if image is None:
                raise SystemExit(f"unreadable frame: {sequence.frame_path(frame_id)}")
            embeddings = computer.compute_embedding(
                image, rows[:, :4], f"{sequence.name}:{frame_id}"
            )
            embeddings = np.asarray(embeddings, dtype=np.float32).reshape(rows.shape[0], -1)
            vectors.append(embeddings)
            row_frames.extend([frame_id] * rows.shape[0])
            computer.cache.clear()
        matrix = (
            np.concatenate(vectors, axis=0)
            if vectors
            else np.empty((0, 0), dtype=np.float32)
        )
        if matrix.shape[0] != detections.detection_count:
            raise SystemExit(
                f"{sequence.name}: {matrix.shape[0]} embeddings for "
                f"{detections.detection_count} detections"
            )
        if not np.isfinite(matrix).all():
            raise SystemExit(f"{sequence.name}: non-finite embedding values")
        destination = artifacts.embeddings_file(
            args.detector, args.reid, args.split, sequence.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise SystemExit(f"refusing to overwrite embeddings: {destination}")
        np.savez_compressed(
            destination,
            embeddings=matrix,
            row_frame=np.asarray(row_frames, dtype=np.int32),
        )
        norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.zeros(1)
        entries.append(
            {
                "sequence": sequence.name,
                "rows": int(matrix.shape[0]),
                "dimension": int(matrix.shape[1]),
                "mean_norm": round(float(norms.mean()), 4),
                "detections_sha256": sha256_file(detection_file),
                "written_sha256": sha256_file(destination),
            }
        )
        print(
            f"{sequence.name}: {matrix.shape[0]:>7} embeddings, dim {matrix.shape[1]}, "
            f"mean norm {norms.mean():.4f}"
        )

    elapsed = time.perf_counter() - started
    dimensions = {entry["dimension"] for entry in entries}
    if len(dimensions) != 1:
        raise SystemExit(f"inconsistent embedding dimensions across sequences: {dimensions}")

    digest = artifacts.write_manifest(
        level_dir,
        {
            "format": EMBEDDINGS_FORMAT,
            "detector_variant": args.detector,
            "reid_variant": args.reid,
            "split": args.split,
            "reid_dataset_key": args.reid_dataset,
            "test_dataset": bool(args.test_dataset),
            "extractor": "BoostTrack tracker.embedding.EmbeddingComputer(grid_off=True)",
            "row_order": "matches det.txt row order; row_frame gives each row's frame id",
            "dimension": dimensions.pop(),
            "wall_seconds": round(elapsed, 2),
            "sequences": entries,
            "totals": {
                "sequences": len(entries),
                "rows": sum(int(entry["rows"]) for entry in entries),
            },
        },
    )
    print(f"\nmanifest {artifacts.manifest_path(level_dir)} sha256 {digest}")


if __name__ == "__main__":
    main()
