from __future__ import annotations

from fastapi import APIRouter

from mot20.viewer.api import ApiModel

COLOR_CONTRACT_VERSION = "fnv1a32-hsv-integer-v1"
COLOR_KEY_ENCODING = "UTF-8(sequence + U+001F + decimal track ID)"
_FNV_OFFSET_BASIS = 2_166_136_261
_FNV_PRIME = 16_777_619
_CHROMA = 172
_MINIMUM_CHANNEL = 58


class TrackColor(ApiModel):
    sequence: str
    track_id: int
    hue: int
    rgb: tuple[int, int, int]
    hex: str


class TrackColorContractResponse(ApiModel):
    version: str
    key_encoding: str
    fnv_offset_basis: int
    fnv_prime: int
    chroma: int
    minimum_channel: int
    vectors: tuple[TrackColor, ...]


def track_color(sequence: str, track_id: int) -> TrackColor:
    hash_value = _FNV_OFFSET_BASIS
    for byte in f"{sequence}\x1f{track_id}".encode():
        hash_value ^= byte
        hash_value = hash_value * _FNV_PRIME & 0xFFFFFFFF
    hue = hash_value % 360
    distance = 60 - abs((hue % 120) - 60)
    intermediate = (_CHROMA * distance + 30) // 60
    sector = hue // 60
    channels = (
        (_CHROMA, intermediate, 0),
        (intermediate, _CHROMA, 0),
        (0, _CHROMA, intermediate),
        (0, intermediate, _CHROMA),
        (intermediate, 0, _CHROMA),
        (_CHROMA, 0, intermediate),
    )[sector]
    rgb = (
        channels[0] + _MINIMUM_CHANNEL,
        channels[1] + _MINIMUM_CHANNEL,
        channels[2] + _MINIMUM_CHANNEL,
    )
    return TrackColor(
        sequence=sequence,
        track_id=track_id,
        hue=hue,
        rgb=rgb,
        hex=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
    )


COLOR_GOLDEN_VECTORS = (
    track_color("MOT20-01", 1),
    track_color("MOT20-01", 8),
    track_color("MOT20-06", 8),
)

color_router = APIRouter()


@color_router.get(
    "/api/contracts/track-colors",
    response_model=TrackColorContractResponse,
)
def track_color_contract() -> TrackColorContractResponse:
    return TrackColorContractResponse(
        version=COLOR_CONTRACT_VERSION,
        key_encoding=COLOR_KEY_ENCODING,
        fnv_offset_basis=_FNV_OFFSET_BASIS,
        fnv_prime=_FNV_PRIME,
        chroma=_CHROMA,
        minimum_channel=_MINIMUM_CHANNEL,
        vectors=COLOR_GOLDEN_VECTORS,
    )