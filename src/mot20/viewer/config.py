from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, cast

from mot20.viewer.contracts import Diagnostic

type AdapterKind = Literal["mot_gt_9", "mot_result_10"]


class ConfigError(ValueError):
    """Raised when viewer configuration is malformed or unsafe."""


@dataclass(frozen=True)
class Provenance:
    producer: str | None = None
    detector: str | None = None
    checkpoint: str | None = None
    tracker: str | None = None
    post_processing: str | None = None
    adaptation_iterations: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SourceConfig:
    key: str
    sequence: str
    seqinfo: str
    images: str
    annotations: str
    adapter: AdapterKind
    provenance: Provenance


@dataclass(frozen=True)
class ViewerConfig:
    sources: tuple[SourceConfig, ...]


@dataclass(frozen=True)
class ResolvedSourcePaths:
    seqinfo: Path
    images: Path
    annotations: Path


def load_config(config_path: Path) -> ViewerConfig:
    try:
        with Path(config_path).open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read viewer configuration: {config_path}") from error
    if set(document) != {"sources"} or not isinstance(document["sources"], list):
        raise ConfigError("viewer configuration must contain only a sources array")
    sources = tuple(_parse_source(value, index) for index, value in enumerate(document["sources"], start=1))
    keys = [source.key for source in sources]
    if len(keys) != len(set(keys)):
        raise ConfigError("source keys must be unique")
    return ViewerConfig(sources=sources)


def provenance_diagnostics(config: ViewerConfig) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    for source in config.sources:
        missing = tuple(field.name for field in fields(Provenance) if getattr(source.provenance, field.name) is None)
        if missing:
            diagnostics.append(
                Diagnostic(
                    code="missing_provenance",
                    message=f"source {source.key} has incomplete provenance",
                    source_key=source.key,
                    fields=missing,
                )
            )
    return tuple(diagnostics)


def resolve_source_paths(source: SourceConfig, repository_root: Path) -> ResolvedSourcePaths:
    root = Path(repository_root).resolve(strict=True)
    resolved: dict[str, Path] = {}
    for name in ("seqinfo", "images", "annotations"):
        configured = Path(getattr(source, name))
        if configured.is_absolute():
            raise ConfigError(f"source {source.key} {name} path escapes repository root")
        candidate = (root / configured).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ConfigError(f"source {source.key} {name} path escapes repository root")
        resolved[name] = candidate
    return ResolvedSourcePaths(**resolved)


def _parse_source(value: object, index: int) -> SourceConfig:
    required = {"key", "sequence", "seqinfo", "images", "annotations", "adapter"}
    allowed = required | {"provenance"}
    if not isinstance(value, dict) or set(value) - allowed or not required <= set(value):
        raise ConfigError(f"source {index} must contain exactly the required source fields and optional provenance")
    strings = {name: _require_string(value[name], f"source {index} {name}") for name in required}
    adapter = strings["adapter"]
    if adapter not in ("mot_gt_9", "mot_result_10"):
        raise ConfigError(f"source {index} has unsupported adapter: {adapter}")
    provenance = _parse_provenance(value.get("provenance"), index)
    return SourceConfig(
        key=strings["key"],
        sequence=strings["sequence"],
        seqinfo=strings["seqinfo"],
        images=strings["images"],
        annotations=strings["annotations"],
        adapter=cast(AdapterKind, adapter),
        provenance=provenance,
    )


def _parse_provenance(value: object, source_index: int) -> Provenance:
    if value is None:
        return Provenance()
    allowed = {field.name for field in fields(Provenance)}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ConfigError(f"source {source_index} provenance contains unsupported fields")
    adaptation_iterations = value.get("adaptation_iterations")
    if adaptation_iterations is not None and (
        type(adaptation_iterations) is not int or adaptation_iterations < 0
    ):
        raise ConfigError(f"source {source_index} provenance adaptation_iterations must be a non-negative integer")
    return Provenance(
        producer=_optional_string(value, "producer", source_index),
        detector=_optional_string(value, "detector", source_index),
        checkpoint=_optional_string(value, "checkpoint", source_index),
        tracker=_optional_string(value, "tracker", source_index),
        post_processing=_optional_string(value, "post_processing", source_index),
        adaptation_iterations=adaptation_iterations,
        notes=_optional_string(value, "notes", source_index),
    )


def _optional_string(values: dict[object, object], name: str, source_index: int) -> str | None:
    value = values.get(name)
    if value is None:
        return None
    return _require_string(value, f"source {source_index} provenance {name}")


def _require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{description} must be a non-empty string")
    return value