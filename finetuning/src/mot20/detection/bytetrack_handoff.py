"""Validated RF-DETR raw detection export and ByteTrack handoff helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def byte_tracker_inputs(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Convert original-pixel pedestrian predictions to ByteTrack's scale-one input."""
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or scores.shape != (boxes.shape[0],) or class_ids.shape != (boxes.shape[0],):
        raise ValueError("expected boxes [N, 4], scores [N], and class IDs [N]")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("detections must have finite boxes and scores in [0, 1]")
    if not np.equal(class_ids, 0).all():
        raise ValueError("ByteTrack handoff accepts only pedestrian class 0")
    clipped = boxes.copy()
    clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, width)
    clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, height)
    valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    detections = np.column_stack((clipped[valid], scores[valid])).astype(np.float32, copy=False)
    native_size = (height, width)
    return detections, native_size, native_size


def write_raw_detector_export(
    frames: Sequence[dict[str, Any]],
    destination: Path,
    config_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Write a never-overwritten, frame-complete original-pixel detector export."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite raw detector export: {destination}")
    _validate_sha256(config_sha256, "config_sha256")
    _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    seen_frames: set[tuple[str, int]] = set()
    validated_frames = []
    for frame in frames:
        sequence = frame.get("sequence")
        frame_id = frame.get("frame_id")
        width = frame.get("width")
        height = frame.get("height")
        if not isinstance(sequence, str) or not sequence or not isinstance(frame_id, int) or frame_id < 1:
            raise ValueError("raw export frames require a nonempty sequence and 1-based frame_id")
        if (sequence, frame_id) in seen_frames:
            raise ValueError(f"duplicate raw export frame: {sequence}/{frame_id}")
        seen_frames.add((sequence, frame_id))
        detections = _validate_frame_detections(frame.get("detections"), width, height)
        validated_frames.append(
            {"sequence": sequence, "frame_id": frame_id, "width": width, "height": height, "detections": detections}
        )
    payload = {
        "format": "mot20.rfdetr.raw-detector-export.v1",
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "frames": validated_frames,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _validate_frame_detections(detections: Any, width: Any, height: Any) -> list[list[float]]:
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise ValueError("raw export frames require positive native dimensions")
    if not isinstance(detections, list):
        raise ValueError("raw export frame detections must be a list")
    validated: list[list[float]] = []
    for detection in detections:
        if not isinstance(detection, list) or len(detection) != 5 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in detection):
            raise ValueError("raw export detections must be finite [x1, y1, x2, y2, score] rows")
        left, top, right, bottom, score = (float(value) for value in detection)
        if not 0 <= score <= 1 or not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("raw export detection is outside native bounds or has an invalid score")
        validated.append([left, top, right, bottom, score])
    return validated


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")