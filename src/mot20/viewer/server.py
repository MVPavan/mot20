from __future__ import annotations

import argparse
import ipaddress
import logging
from collections.abc import Sequence
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from mot20.viewer.api import create_app
from mot20.viewer.colors import color_router
from mot20.viewer.config import ViewerConfig, config_from_paths, load_config
from mot20.viewer.context import context_router
from mot20.viewer.crops import crop_router
from mot20.viewer.events import event_router
from mot20.viewer.exports import export_router
from mot20.viewer.filmstrip import filmstrip_router
from mot20.viewer.loaders import load_registry
from mot20.viewer.tracks import track_router

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INTERNAL_ROUTERS = (
    track_router,
    color_router,
    filmstrip_router,
    crop_router,
    context_router,
    event_router,
    export_router,
)


def build_app(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    images: Path | None = None,
    annotations: Path | None = None,
    config_path: Path | None = None,
    development_origin: str | None = None,
    application_origin: str = "http://127.0.0.1:8000",
    trusted_hosts: tuple[str, ...] = ("127.0.0.1", "localhost"),
) -> FastAPI:
    root = Path(repository_root).resolve(strict=True)
    if images is not None and annotations is not None:
        config = config_from_paths(images, annotations)
    elif config_path is not None:
        selected_config = config_path
        if not selected_config.is_absolute():
            selected_config = root / selected_config
        config = load_config(selected_config)
    else:
        config = ViewerConfig(sources=())
    registry = load_registry(config, root)
    return create_app(
        registry=registry,
        repository_root=root,
        trusted_hosts=trusted_hosts,
        development_origin=development_origin,
        application_origin=application_origin,
        extension_routers=INTERNAL_ROUTERS,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    if not _is_loopback_host(arguments.host) and not arguments.allow_non_loopback:
        parser.error("a non-loopback --host requires --allow-non-loopback")
    if not _is_loopback_host(arguments.host):
        LOGGER.warning("starting viewer on non-loopback host %s", arguments.host)
    if (arguments.images is None) != (arguments.annotations is None):
        parser.error("--images and --annotations must be provided together")
    if arguments.images is not None and arguments.config is not None:
        parser.error("--config cannot be combined with --images and --annotations")
    trusted_hosts = tuple(arguments.trusted_host or ("127.0.0.1", "localhost"))
    application_origin = (
        arguments.app_origin
        or arguments.dev_origin
        or f"http://{arguments.host}:{arguments.port}"
    )
    app = build_app(
        images=arguments.images,
        annotations=arguments.annotations,
        config_path=arguments.config,
        development_origin=arguments.dev_origin,
        application_origin=application_origin,
        trusted_hosts=trusted_hosts,
    )
    uvicorn.run(app, host=arguments.host, port=arguments.port)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local MOT20 viewer")
    parser.add_argument("--images", type=Path, help="folder containing numbered JPEG frames")
    parser.add_argument("--annotations", type=Path, help="MOT predictions or ground-truth file")
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-non-loopback", action="store_true")
    parser.add_argument("--trusted-host", action="append")
    parser.add_argument("--dev-origin")
    parser.add_argument("--app-origin")
    return parser


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


if __name__ == "__main__":
    main()