from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import supervision as sv

from mot20.viewer.contracts import Observation


def observations_to_detections(observations: Sequence[Observation]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.asarray(
            [observation.display_box for observation in observations],
            dtype=np.float32,
        ).reshape((-1, 4)),
        confidence=np.asarray(
            [float("nan") if observation.score is None else observation.score for observation in observations],
            dtype=np.float32,
        ),
        tracker_id=np.asarray(
            [
                -1 if observation.usable_track_id is None else observation.usable_track_id
                for observation in observations
            ],
            dtype=int,
        ),
        data={
            "row_index": np.asarray([observation.row_index for observation in observations], dtype=int),
            "row_hash": np.asarray([observation.row_hash for observation in observations], dtype=str),
        },
    )


def filter_detections_by_tracker_id(
    detections: sv.Detections,
    tracker_id: int,
) -> sv.Detections:
    if detections.tracker_id is None:
        return cast(sv.Detections, detections[np.zeros(len(detections), dtype=bool)])
    return cast(sv.Detections, detections[detections.tracker_id == tracker_id])