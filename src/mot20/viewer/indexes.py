from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mot20.viewer.contracts import Observation, Sequence


@dataclass(frozen=True)
class SequenceIndexes:
    frames: Mapping[int, tuple[Observation, ...]]
    tracks: Mapping[int, tuple[Observation, ...]]


def build_indexes(sequence: Sequence, observations: tuple[Observation, ...]) -> SequenceIndexes:
    frames: dict[int, list[Observation]] = {frame: [] for frame in range(1, sequence.length + 1)}
    tracks: defaultdict[int, list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.sequence != sequence.name:
            raise ValueError(
                f"observation sequence {observation.sequence} does not match index sequence {sequence.name}"
            )
        frames[observation.frame].append(observation)
        if observation.usable_track_id is not None:
            tracks[observation.usable_track_id].append(observation)
    frozen_frames = MappingProxyType({frame: tuple(rows) for frame, rows in frames.items()})
    frozen_tracks = MappingProxyType({track_id: tuple(rows) for track_id, rows in sorted(tracks.items())})
    return SequenceIndexes(frames=frozen_frames, tracks=frozen_tracks)