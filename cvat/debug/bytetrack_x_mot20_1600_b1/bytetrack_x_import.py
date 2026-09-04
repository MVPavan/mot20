"""Convert supplied YOLO detection labels into CVAT shapes."""
from __future__ import annotations

from pathlib import Path
from typing import Any


BOUNDARY_TOLERANCE_PIXELS = 0.01


def read_yolo_shapes(
    labels_dir: Path,
    *,
    sequence_length: int,
    image_width: int,
    image_height: int,
    label_id: int,
    start_frame: int = 1,
    stop_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Read one YOLO label file per MOT frame without inventing score or identity."""
    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        raise ValueError(f"labels directory does not exist: {labels_dir}")
    expected_names = {f"{frame:06d}.txt" for frame in range(1, sequence_length + 1)}
    actual_names = {path.name for path in labels_dir.glob("*.txt")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"label files do not exactly match frames; missing={missing[:3]} extra={extra[:3]}")
    stop_frame = sequence_length if stop_frame is None else stop_frame
    if not 1 <= start_frame <= stop_frame <= sequence_length:
        raise ValueError(f"invalid frame range: {start_frame}-{stop_frame}")

    shapes: list[dict[str, Any]] = []
    for frame in range(start_frame, stop_frame + 1):
        path = labels_dir / f"{frame:06d}.txt"
        for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
            if not raw.strip():
                continue
            columns = raw.split()
            if len(columns) not in (5, 6) or columns[0] != "0":
                raise ValueError(f"unsupported YOLO row at {path}:{line_number}")
            try:
                center_x, center_y, width, height = (float(value) for value in columns[1:5])
                if len(columns) == 6:
                    float(columns[5])
            except ValueError as error:
                raise ValueError(f"invalid YOLO geometry at {path}:{line_number}") from error
            if not 0 <= center_x <= 1 or not 0 <= center_y <= 1 or not 0 < width <= 1 or not 0 < height <= 1:
                raise ValueError(f"invalid normalized YOLO geometry at {path}:{line_number}")
            left = (center_x - width / 2) * image_width
            top = (center_y - height / 2) * image_height
            right = left + width * image_width
            bottom = top + height * image_height
            if (
                left < -BOUNDARY_TOLERANCE_PIXELS
                or top < -BOUNDARY_TOLERANCE_PIXELS
                or right > image_width + BOUNDARY_TOLERANCE_PIXELS
                or bottom > image_height + BOUNDARY_TOLERANCE_PIXELS
            ):
                raise ValueError(f"YOLO geometry falls outside image bounds at {path}:{line_number}")
            left = max(0.0, left)
            top = max(0.0, top)
            right = min(float(image_width), right)
            bottom = min(float(image_height), bottom)
            shapes.append({
                "type": "rectangle",
                "frame": frame - start_frame,
                "label_id": label_id,
                "group": 0,
                "source": "semi-auto",
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0,
                "points": [left, top, right, bottom],
                "attributes": [],
            })
    return shapes


def annotation_payload(shapes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a scoreless, untracked CVAT annotation payload."""
    return {"version": 0, "tags": [], "shapes": shapes, "tracks": []}
