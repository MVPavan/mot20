"""Level-1 detection files: read, write, and validate.

The on-disk format follows the MOT detection convention already used by
``datasets/val_half/<sequence>/det_yoloxx20/`` and
``datasets/MOT20_TEST_DET/``, so existing detections register as ordinary
variants:

``frame,-1,left,top,width,height,score,class,visibility``

Coordinates are top-left width/height in ORIGINAL image pixels, frame IDs are
1-based, and scores are in [0, 1]. Boxes are stored below any tracker
threshold on purpose: BoostTrack boosts low-confidence detections that match
existing tracks before applying its own cut, so a pre-filtered file silently
disables that mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PEDESTRIAN_CLASS = 0


@dataclass(frozen=True)
class SequenceDetections:
    """Detections for one sequence, grouped by frame.

    Attributes:
        sequence: Sequence name, for example ``"MOT20-01"``.
        frames: Mapping of 1-based frame ID to an ``(N, 5)`` float32 array of
            ``[x1, y1, x2, y2, score]`` in original pixels. Frames with no
            detections are present with an ``(0, 5)`` array, so a consumer that
            must be called once per frame cannot silently skip one.
    """

    sequence: str
    frames: dict[int, np.ndarray]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def detection_count(self) -> int:
        return int(sum(array.shape[0] for array in self.frames.values()))

    def rows_for(self, frame_id: int) -> np.ndarray:
        """Return the ``(N, 5)`` array for *frame_id*, empty if absent."""
        return self.frames.get(frame_id, np.empty((0, 5), dtype=np.float32))


def write_detections(
    destination: Path,
    detections: SequenceDetections,
    sequence_length: int,
) -> None:
    """Write a detection file once, requiring every frame to be represented.

    Args:
        destination: Target ``det.txt`` path.
        detections: Frame-indexed detections in original pixels.
        sequence_length: Frame count from ``seqinfo.ini``. Every frame from 1
            to this value must be present, including empty ones.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If frames are missing, duplicated, or out of range.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing detections: {destination}")
    if sequence_length < 1:
        raise ValueError(f"sequence length must be positive: {sequence_length}")
    missing = [f for f in range(1, sequence_length + 1) if f not in detections.frames]
    if missing:
        raise ValueError(
            f"{detections.sequence}: detection export is missing {len(missing)} frame(s), "
            f"first={missing[0]}"
        )
    extra = [f for f in detections.frames if not 1 <= f <= sequence_length]
    if extra:
        raise ValueError(f"{detections.sequence}: frame IDs outside 1..{sequence_length}: {extra[:5]}")

    lines: list[str] = []
    for frame_id in range(1, sequence_length + 1):
        for x1, y1, x2, y2, score in detections.frames[frame_id]:
            lines.append(
                f"{frame_id},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                f"{score:.4f},{PEDESTRIAN_CLASS},1.0"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        if lines:
            stream.write("\n")


def read_detections(source: Path, sequence: str, sequence_length: int) -> SequenceDetections:
    """Read a detection file into frame-indexed xyxy arrays.

    Frames absent from the file are materialised as empty arrays, so the result
    always covers ``1..sequence_length``.
    """
    source = Path(source)
    frames: dict[int, list[list[float]]] = {f: [] for f in range(1, sequence_length + 1)}
    with source.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                raise ValueError(f"{source}:{number}: expected at least 7 comma-separated fields")
            frame_id = int(float(parts[0]))
            left, top, width, height, score = (float(parts[i]) for i in (2, 3, 4, 5, 6))
            if frame_id not in frames:
                raise ValueError(f"{source}:{number}: frame {frame_id} outside 1..{sequence_length}")
            frames[frame_id].append([left, top, left + width, top + height, score])
    return SequenceDetections(
        sequence=sequence,
        frames={
            frame_id: np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            for frame_id, rows in frames.items()
        },
    )


def validate_detections(
    detections: SequenceDetections,
    width: int,
    height: int,
    sequence_length: int,
    tolerance: float = 1.0,
) -> dict[str, float | int]:
    """Check structural validity and return summary statistics.

    Boxes are allowed to exceed the image bounds by *tolerance* pixels before
    being reported, since a detector may legitimately predict a box whose edge
    lands fractionally outside the frame.

    Raises:
        ValueError: On a missing frame, a non-finite value, an out-of-range
            score, a degenerate box, or a box beyond the tolerance.
    """
    if detections.frame_count != sequence_length:
        raise ValueError(
            f"{detections.sequence}: expected {sequence_length} frames, "
            f"found {detections.frame_count}"
        )
    per_frame: list[int] = []
    min_score = 1.0
    max_score = 0.0
    for frame_id in range(1, sequence_length + 1):
        rows = detections.frames[frame_id]
        per_frame.append(rows.shape[0])
        if rows.shape[0] == 0:
            continue
        if not np.isfinite(rows).all():
            raise ValueError(f"{detections.sequence}: frame {frame_id} has non-finite values")
        scores = rows[:, 4]
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"{detections.sequence}: frame {frame_id} has scores outside [0, 1]")
        if ((rows[:, 2] <= rows[:, 0]) | (rows[:, 3] <= rows[:, 1])).any():
            raise ValueError(f"{detections.sequence}: frame {frame_id} has a degenerate box")
        if (
            (rows[:, 0] < -tolerance).any()
            or (rows[:, 1] < -tolerance).any()
            or (rows[:, 2] > width + tolerance).any()
            or (rows[:, 3] > height + tolerance).any()
        ):
            raise ValueError(f"{detections.sequence}: frame {frame_id} has a box outside the image")
        min_score = min(min_score, float(scores.min()))
        max_score = max(max_score, float(scores.max()))
    counts = np.asarray(per_frame)
    return {
        "frames": sequence_length,
        "detections": detections.detection_count,
        "max_per_frame": int(counts.max()),
        "mean_per_frame": round(float(counts.mean()), 3),
        "empty_frames": int((counts == 0).sum()),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
    }
