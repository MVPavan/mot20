from __future__ import annotations

import base64
import io
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from test_tracks import build_track_registry

from mot20.viewer.api import create_app
from mot20.viewer.crops import _render_crop, crop_router


class CropApiTest(unittest.TestCase):
    def test_crop_uses_exact_row_identity_returns_geometry_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(crop_router,),
                )
            )
            url = (
                "/api/sequences/fixture-gt/observations/1/crop"
                f"?source_hash={source_hash}&padding=2&max_size=64"
            )

            first = client.get(url)
            second = client.get(url)

            cache_files = tuple((root / "track-viz" / "artifacts" / "cache").rglob("*.jpg"))
            cached_bytes = cache_files[0].read_bytes()

        self.assertEqual(first.status_code, 200)
        payload = first.json()
        self.assertEqual(payload["row_index"], 1)
        self.assertEqual(payload["source_hash"], source_hash)
        self.assertEqual(
            payload["raw_geometry"],
            {"x": -1.0, "y": -2.0, "width": 4.0, "height": 5.0},
        )
        self.assertEqual(
            payload["display_geometry"],
            {"x1": 0.0, "y1": 0.0, "x2": 3.0, "y2": 3.0},
        )
        self.assertEqual(
            payload["crop_geometry"],
            {
                "x1": 0,
                "y1": 0,
                "x2": 5,
                "y2": 5,
                "width": 5,
                "height": 5,
                "padding": 2,
                "max_size": 64,
                "output_width": 5,
                "output_height": 5,
            },
        )
        image_bytes = base64.b64decode(payload["image_base64"], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (5, 5))
        self.assertEqual(second.json()["image_base64"], payload["image_base64"])
        self.assertEqual(len(cache_files), 1)
        self.assertEqual(cache_files[0].stem, payload["cache_key"])
        self.assertEqual(cached_bytes, image_bytes)

    def test_crop_rejects_stale_hash_missing_rows_and_out_of_bounds_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(crop_router,),
                )
            )

            stale = client.get(
                "/api/sequences/fixture-gt/observations/1/crop?source_hash=stale"
            )
            missing = client.get(
                f"/api/sequences/fixture-gt/observations/999/crop?source_hash={source_hash}"
            )
            invalid = tuple(
                client.get(
                    "/api/sequences/fixture-gt/observations/1/crop"
                    f"?source_hash={source_hash}&{parameter}"
                )
                for parameter in ("padding=-1", "padding=129", "max_size=31", "max_size=1025")
            )

        self.assertEqual(stale.status_code, 412)
        self.assertEqual(stale.json()["error"]["code"], "stale_source_hash")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "observation_not_found")
        self.assertTrue(all(response.status_code == 422 for response in invalid))

    def test_crop_cache_key_changes_when_source_jpeg_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(crop_router,),
                )
            )
            url = f"/api/sequences/fixture-gt/observations/1/crop?source_hash={source_hash}"

            first = client.get(url).json()
            Image.new("RGB", (16, 12), color=(200, 100, 50)).save(
                root / "fixture" / "MOT20-01" / "img1" / "000001.jpg",
                format="JPEG",
            )
            second = client.get(url).json()
            cache_files = tuple((root / "track-viz" / "artifacts" / "cache").rglob("*.jpg"))

        self.assertNotEqual(first["source_image_hash"], second["source_image_hash"])
        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(len(cache_files), 2)

    def test_crop_cache_rejects_symlink_escape_without_writing_outside(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            root = workspace / "repository"
            root.mkdir()
            registry = build_track_registry(root)
            outside = workspace / "outside"
            outside.mkdir()
            viewer_root = root / "track-viz"
            viewer_root.mkdir()
            (viewer_root / "artifacts").symlink_to(outside, target_is_directory=True)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(crop_router,),
                )
            )

            response = client.get(
                f"/api/sequences/fixture-gt/observations/1/crop?source_hash={source_hash}"
            )
            outside_contents = tuple(outside.iterdir())

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "unsafe_cache_root")
        self.assertEqual(outside_contents, ())

    def test_crop_cache_rejects_symlinked_existing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(crop_router,),
                )
            )
            url = f"/api/sequences/fixture-gt/observations/1/crop?source_hash={source_hash}"
            first = client.get(url)
            cache_file = next((root / "track-viz" / "artifacts" / "cache").rglob("*.jpg"))
            cache_file.unlink()
            outside = root / "outside.jpg"
            outside.write_bytes(b"outside cache content")
            cache_file.symlink_to(outside)

            response = client.get(url)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "unsafe_cache_path")

    def test_concurrent_duplicate_crops_create_one_immutable_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(crop_router,),
                )
            )
            url = f"/api/sequences/fixture-gt/observations/1/crop?source_hash={source_hash}"

            with ThreadPoolExecutor(max_workers=8) as executor:
                responses = tuple(executor.map(lambda _request: client.get(url), range(8)))
            cache_files = tuple((root / "track-viz" / "artifacts" / "cache").rglob("*.jpg"))

        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(len({response.json()["image_base64"] for response in responses}), 1)
        self.assertEqual(sum(response.json()["cache_status"] == "created" for response in responses), 1)
        self.assertEqual(len(cache_files), 1)

    def test_held_crop_render_does_not_delay_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source_hash = registry.sources[0].source_hash
            render_started = threading.Event()
            release_render = threading.Event()

            def held_render(*args, **kwargs):  # type: ignore[no-untyped-def]
                render_started.set()
                if not release_render.wait(timeout=1):
                    raise TimeoutError("test did not release crop render")
                return _render_crop(*args, **kwargs)

            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(crop_router,),
                )
            )
            with (
                patch("mot20.viewer.crops._render_crop", side_effect=held_render),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                crop_future = executor.submit(
                    client.get,
                    "/api/sequences/fixture-gt/observations/1/crop"
                    f"?source_hash={source_hash}",
                )
                self.assertTrue(render_started.wait(timeout=1))
                metadata_future = executor.submit(client.get, "/api/sequences/fixture-gt")
                metadata = metadata_future.result(timeout=1)
                release_render.set()
                crop = crop_future.result(timeout=1)

        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(crop.status_code, 200)


if __name__ == "__main__":
    unittest.main()