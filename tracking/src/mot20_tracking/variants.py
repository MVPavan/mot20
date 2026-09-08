"""Variant identifiers for the tracking experiment grid.

A tracking result is identified by a triple of variant slugs: which detector
produced the boxes, which ReID model embedded them, and which tracker
configuration consumed both. Every artifact path and every cache key is derived
from these slugs, so they are validated strictly rather than accepted as free
text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
"""Lowercase alphanumeric words joined by single hyphens."""

_SEPARATOR = "__"

# Reserved because it is the combination separator; a slug containing it would
# make a combination identifier ambiguous to parse.
_FORBIDDEN = (_SEPARATOR, "/", "\\", " ")


def validate_slug(value: str, kind: str) -> str:
    """Return *value* if it is a well-formed variant slug, else raise.

    Args:
        value: Candidate slug, for example ``"rfdetr2xl-e5-t005"``.
        kind: Human-readable role used in the error message, for example
            ``"detector"``.

    Raises:
        ValueError: If the slug is empty, malformed, or contains a reserved
            separator.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} variant must be a non-empty string")
    for token in _FORBIDDEN:
        if token in value:
            raise ValueError(f"{kind} variant must not contain {token!r}: {value!r}")
    if not SLUG_PATTERN.match(value):
        raise ValueError(
            f"{kind} variant must be lowercase alphanumeric words joined by hyphens: {value!r}"
        )
    return value


@dataclass(frozen=True)
class Combination:
    """A fully specified detector, ReID, and tracker combination.

    ``reid`` is the literal slug ``"none"`` for tracker configurations that run
    without appearance embeddings, so that a ReID-free run is an ordinary cell
    of the grid rather than a missing value.
    """

    detector: str
    reid: str
    tracker: str

    def __post_init__(self) -> None:
        validate_slug(self.detector, "detector")
        validate_slug(self.reid, "reid")
        validate_slug(self.tracker, "tracker")

    @property
    def uses_reid(self) -> bool:
        return self.reid != "none"

    @property
    def detection_reid_id(self) -> str:
        """Identifier for the embedding level, which depends on both inputs."""
        return f"{self.detector}{_SEPARATOR}{self.reid}"

    def __str__(self) -> str:
        return f"{self.detector}{_SEPARATOR}{self.reid}{_SEPARATOR}{self.tracker}"

    @classmethod
    def parse(cls, value: str) -> "Combination":
        """Parse ``"<detector>__<reid>__<tracker>"`` back into a combination."""
        parts = value.split(_SEPARATOR)
        if len(parts) != 3:
            raise ValueError(f"combination must have three {_SEPARATOR}-separated parts: {value!r}")
        return cls(detector=parts[0], reid=parts[1], tracker=parts[2])


def detection_reid_id(detector: str, reid: str) -> str:
    """Embedding-level identifier for a detector and ReID pair."""
    validate_slug(detector, "detector")
    validate_slug(reid, "reid")
    return f"{detector}{_SEPARATOR}{reid}"
