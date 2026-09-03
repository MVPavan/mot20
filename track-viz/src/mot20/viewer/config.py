from __future__ import annotations

import configparser
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
    paths_are_explicit: bool = False


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


def config_from_paths(images: Path, annotations: Path) -> ViewerConfig:
    try:
        image_path = Path(images).expanduser().resolve(strict=True)
        annotation_path = Path(annotations).expanduser().resolve(strict=True)
    except OSError as error:
        raise ConfigError("direct source paths must exist") from error
    if not image_path.is_dir():
        raise ConfigError(f"images path is not a directory: {image_path}")
    if not annotation_path.is_file():
        raise ConfigError(f"annotations path is not a file: {annotation_path}")

    seqinfo_path = image_path.parent / "seqinfo.ini"
    sequence = _read_sequence_name(seqinfo_path)
    adapter = _infer_adapter(annotation_path)
    key = f"{sequence}-{annotation_path.stem}".lower().replace("_", "-")
    source = SourceConfig(
        key=key,
        sequence=sequence,
        seqinfo=str(seqinfo_path),
        images=str(image_path),
        annotations=str(annotation_path),
        adapter=adapter,
        provenance=Provenance(),
        paths_are_explicit=True,
    )
    return ViewerConfig(sources=(source,))


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
        if source.paths_are_explicit:
            if not configured.is_absolute():
                raise ConfigError(f"source {source.key} explicit {name} path must be absolute")
            resolved[name] = configured.resolve(strict=False)
            continue
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


def _read_sequence_name(seqinfo_path: Path) -> str:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with seqinfo_path.open(encoding="utf-8") as stream:
            parser.read_file(stream)
        return _require_string(parser["Sequence"]["name"], "sequence name")
    except (OSError, KeyError, configparser.Error) as error:
        raise ConfigError(f"cannot infer sequence from {seqinfo_path}") from error


def _infer_adapter(annotation_path: Path) -> AdapterKind:
    try:
        first_row = annotation_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeDecodeError, IndexError) as error:
        raise ConfigError(f"cannot infer annotation format from {annotation_path}") from error
    field_count = len(first_row.split(","))
    if field_count == 9:
        return "mot_gt_9"
    if field_count == 10:
        return "mot_result_10"
    raise ConfigError(
        f"annotations must contain MOT ground-truth rows with 9 fields "
        f"or prediction rows with 10 fields; found {field_count}"
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