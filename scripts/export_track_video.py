from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mot20.viewer.config import load_config
from mot20.viewer.exports import (
    OFFLINE_DEFAULT_FRAME_CAP,
    OFFLINE_HARD_FRAME_CAP,
    ExportArtifactCollisionError,
    ExportParameters,
    ExportValidationError,
    write_track_video,
)
from mot20.viewer.loaders import load_registry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.max_frames <= OFFLINE_HARD_FRAME_CAP:
        parser.error(
            f"--max-frames must be between 1 and {OFFLINE_HARD_FRAME_CAP}"
        )
    root = Path(arguments.repository_root).resolve(strict=True)
    config = load_config(root / "configs" / "viewer.toml")
    registry = load_registry(config, root)
    source = next(
        (candidate for candidate in registry.sources if candidate.config.key == arguments.source_key),
        None,
    )
    if source is None:
        parser.error(f"source {arguments.source_key!r} is not available")
    track = source.indexes.tracks.get(arguments.track_id)
    if track is None:
        parser.error(
            f"track {arguments.track_id} is not usable in source {arguments.source_key!r}"
        )
    if arguments.source_hash is not None and arguments.source_hash != source.source_hash:
        parser.error("--source-hash does not match the current annotation result")
    start_frame = arguments.start_frame or min(observation.frame for observation in track)
    end_frame = arguments.end_frame or max(observation.frame for observation in track)
    parameters = ExportParameters(
        track_id=arguments.track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        context_count=arguments.context_count,
        trace_length=arguments.trace_length,
    )
    try:
        artifact = write_track_video(
            source=source,
            repository_root=root,
            parameters=parameters,
            kind="offline_track_video",
            frame_limit=arguments.max_frames,
        )
    except (ExportValidationError, ExportArtifactCollisionError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "export_id": artifact.export_id,
                "status": artifact.status,
                "artifact_directory": artifact.artifact_directory.relative_to(root).as_posix(),
                "video_path": artifact.video_path.relative_to(root).as_posix(),
                "metadata_path": artifact.metadata_path.relative_to(root).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one bounded, read-only per-track MOT20 video"
    )
    parser.add_argument("source_key")
    parser.add_argument("track_id", type=int)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--source-hash")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int, default=OFFLINE_DEFAULT_FRAME_CAP)
    parser.add_argument("--context-count", type=int, default=3)
    parser.add_argument("--trace-length", type=int, default=30)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())