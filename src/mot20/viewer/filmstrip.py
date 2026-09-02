from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, Query, Request

from mot20.viewer.api import (
    ApiModel,
    ErrorDetail,
    ObservationResponse,
    ViewerApiError,
    _observation_response,
    _require_source,
    require_track_capability,
)
from mot20.viewer.contracts import Observation
from mot20.viewer.loaders import SourceRegistry
from mot20.viewer.tracks import _require_track

MAX_FILMSTRIP_SAMPLES = 64


class FilmstripSampleResponse(ApiModel):
    is_current: bool
    observation: ObservationResponse


class FilmstripResponse(ApiModel):
    source_key: str
    sequence: str
    source_hash: str
    track_id: int
    current_row_index: int
    total_observations: int
    sampled_count: int
    samples: tuple[FilmstripSampleResponse, ...]


filmstrip_router = APIRouter()


@filmstrip_router.get(
    "/api/sequences/{source_key}/tracks/{track_id}/filmstrip",
    response_model=FilmstripResponse,
)
def track_filmstrip(
    request: Request,
    source_key: str,
    track_id: int,
    current_row_index: int = Query(ge=1),
    source_hash: str | None = None,
) -> FilmstripResponse:
    registry: SourceRegistry = request.app.state.registry
    source = _require_source(registry, source_key, source_hash)
    require_track_capability(source)
    observations = _require_track(source, track_id)
    try:
        sampled = sample_filmstrip(
            observations,
            current_row_index=current_row_index,
        )
    except ValueError as error:
        raise ViewerApiError(
            404,
            ErrorDetail(
                code="current_observation_not_found",
                message=(
                    f"row {current_row_index} is not an observation in track {track_id} "
                    f"for source {source.config.key!r}"
                ),
                source_key=source.config.key,
            ),
        ) from error
    return FilmstripResponse(
        source_key=source.config.key,
        sequence=source.sequence.name,
        source_hash=source.source_hash,
        track_id=track_id,
        current_row_index=current_row_index,
        total_observations=len(observations),
        sampled_count=len(sampled),
        samples=tuple(
            FilmstripSampleResponse(
                is_current=observation.row_index == current_row_index,
                observation=_observation_response(observation),
            )
            for observation in sampled
        ),
    )


def sample_filmstrip(
    observations: Sequence[Observation],
    *,
    current_row_index: int,
) -> tuple[Observation, ...]:
    ordered = tuple(sorted(observations, key=lambda observation: (observation.frame, observation.row_index)))
    current_position = next(
        (
            position
            for position, observation in enumerate(ordered)
            if observation.row_index == current_row_index
        ),
        None,
    )
    if current_position is None:
        raise ValueError(f"row {current_row_index} is not an observation in the track")
    if len(ordered) <= MAX_FILMSTRIP_SAMPLES:
        return ordered

    pinned_positions = {0, current_position, len(ordered) - 1}
    earlier_positions = tuple(range(1, current_position))
    later_positions = tuple(range(current_position + 1, len(ordered) - 1))
    remaining_slots = MAX_FILMSTRIP_SAMPLES - len(pinned_positions)
    earlier_count, later_count = _side_allocations(
        len(earlier_positions),
        len(later_positions),
        remaining_slots,
    )
    selected_positions = pinned_positions | set(
        _evenly_spaced_positions(earlier_positions, earlier_count)
    ) | set(_evenly_spaced_positions(later_positions, later_count))
    return tuple(ordered[position] for position in sorted(selected_positions))


def _side_allocations(earlier: int, later: int, slots: int) -> tuple[int, int]:
    earlier_count = int(earlier > 0 and slots > 0)
    later_count = int(later > 0 and slots > earlier_count)
    extras = slots - earlier_count - later_count
    earlier_capacity = earlier - earlier_count
    later_capacity = later - later_count
    total_capacity = earlier_capacity + later_capacity
    if extras <= 0 or total_capacity <= 0:
        return earlier_count, later_count

    earlier_extra = min(earlier_capacity, extras * earlier_capacity // total_capacity)
    later_extra = min(later_capacity, extras - earlier_extra)
    unallocated = extras - earlier_extra - later_extra
    earlier_extra += min(earlier_capacity - earlier_extra, unallocated)
    unallocated = extras - earlier_extra - later_extra
    later_extra += min(later_capacity - later_extra, unallocated)
    return earlier_count + earlier_extra, later_count + later_extra


def _evenly_spaced_positions(positions: tuple[int, ...], count: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    if count >= len(positions):
        return positions
    if count == 1:
        return (positions[(len(positions) - 1) // 2],)
    return tuple(
        positions[slot * (len(positions) - 1) // (count - 1)]
        for slot in range(count)
    )