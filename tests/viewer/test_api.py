from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter
from fastapi.testclient import TestClient
from PIL import Image

from mot20.viewer.api import create_app, require_track_capability
from mot20.viewer.config import AdapterKind, Provenance, SourceConfig, ViewerConfig
from mot20.viewer.loaders import SourceRegistry, load_registry


def build_registry(
    root: Path,
    *,
    key: str = "fixture-gt",
    sequence: str = "MOT20-01",
    adapter: AdapterKind = "mot_gt_9",
    rows: str = "1,7,-1,-2,4,5,1,1,0.8\n2,7,1,1,3,4,1,7,0.5\n",
) -> SourceRegistry:
    sequence_root = root / "fixture" / "MOT20-01"
    image_root = sequence_root / "img1"
    image_root.mkdir(parents=True)
    (sequence_root / "seqinfo.ini").write_text(
        f"""[Sequence]
name={sequence}
imDir=img1
frameRate=25
seqLength=3
imWidth=8
imHeight=6
imExt=.jpg
    """,
        encoding="utf-8",
    )
    for frame in range(1, 4):
        Image.new("RGB", (8, 6), color=(frame, 0, 0)).save(
            image_root / f"{frame:06d}.jpg",
            format="JPEG",
        )
    (sequence_root / "gt.txt").write_text(rows, encoding="utf-8")
    source = SourceConfig(
        key=key,
        sequence=sequence,
        seqinfo="fixture/MOT20-01/seqinfo.ini",
        images="fixture/MOT20-01/img1",
        annotations="fixture/MOT20-01/gt.txt",
        adapter=adapter,
        provenance=Provenance(producer="synthetic"),
    )
    return load_registry(ViewerConfig((source,)), root)


class ViewerApiTest(unittest.TestCase):
    def test_health_and_sequence_list_expose_typed_metadata_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(root)
            client = TestClient(create_app(registry=registry, repository_root=root))

            health = client.get("/api/health")
            response = client.get("/api/sequences")

        self.assertEqual(
            health.json(),
            {"status": "ok", "source_count": 1, "unavailable_count": 0},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["sources"]), 1)
        source = payload["sources"][0]
        self.assertEqual(source["source_key"], "fixture-gt")
        self.assertEqual(source["sequence"], "MOT20-01")
        self.assertEqual(source["frame_numbering"], "one_based")
        self.assertEqual(source["frame_count"], 3)
        self.assertEqual(source["width"], 8)
        self.assertEqual(source["height"], 6)
        self.assertEqual(source["frame_rate"], 25)
        self.assertEqual(source["adapter"], "mot_gt_9")
        self.assertEqual(source["source_class"], "ground_truth")
        self.assertEqual(source["policy_classification"], "ground_truth_training_source")
        self.assertEqual(source["source_row_count"], 2)
        self.assertEqual(source["observation_count"], 1)
        self.assertEqual(source["capability"]["id_status"], "tracked")
        self.assertTrue(source["capability"]["track_features"])
        self.assertEqual(source["provenance"]["producer"], "synthetic")
        self.assertEqual(source["diagnostics"][0]["code"], "missing_provenance")
        self.assertEqual(payload["unavailable"], [])
        serialized = response.text
        self.assertNotIn("seqinfo", serialized)
        self.assertNotIn("annotations", serialized)
        self.assertNotIn(str(root), serialized)

    def test_source_detail_and_observations_use_hash_precondition_and_normalized_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(root)
            source_hash = registry.sources[0].source_hash
            client = TestClient(create_app(registry=registry, repository_root=root))

            detail = client.get(f"/api/sequences/fixture-gt?source_hash={source_hash}")
            response = client.get(
                f"/api/sequences/fixture-gt/frames/1/observations?source_hash={source_hash}"
            )

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["source_hash"], source_hash)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_key"], "fixture-gt")
        self.assertEqual(payload["sequence"], "MOT20-01")
        self.assertEqual(payload["frame"], 1)
        self.assertEqual(payload["frame_numbering"], "one_based")
        self.assertEqual(payload["source_hash"], source_hash)
        self.assertEqual(len(payload["observations"]), 1)
        observation = payload["observations"][0]
        self.assertEqual(observation["row_index"], 1)
        self.assertEqual(len(observation["row_hash"]), 64)
        self.assertEqual(
            observation["raw_geometry"],
            {"x": -1.0, "y": -2.0, "width": 4.0, "height": 5.0},
        )
        self.assertEqual(
            observation["display_geometry"],
            {"x1": 0.0, "y1": 0.0, "x2": 3.0, "y2": 3.0},
        )
        self.assertIsNone(observation["score"])
        self.assertEqual(
            observation["ground_truth"],
            {"mark": 1, "class_id": 1, "visibility": 0.8},
        )
        self.assertEqual(observation["score_semantics"], "not_defined")
        self.assertEqual(observation["ground_truth_semantics"], "mot_mark_class_visibility")
        self.assertIsNone(observation["opaque_result_fields"])
        self.assertNotIn("raw_row", response.text)
        self.assertNotIn("raw_fields", response.text)

    def test_frame_route_serves_exact_enumerated_jpeg_with_strong_etag_and_304(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(root)
            source_hash = registry.sources[0].source_hash
            expected = (root / "fixture" / "MOT20-01" / "img1" / "000002.jpg").read_bytes()
            expected_etag = f'"{hashlib.sha256(expected).hexdigest()}"'
            client = TestClient(create_app(registry=registry, repository_root=root))

            response = client.get(
                f"/api/sequences/fixture-gt/frames/2?source_hash={source_hash}"
            )
            conditional = client.get(
                f"/api/sequences/fixture-gt/frames/2?source_hash={source_hash}",
                headers={"If-None-Match": expected_etag},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, expected)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["etag"], expected_etag)
        self.assertEqual(response.headers["cache-control"], "private, max-age=31536000, immutable")
        self.assertEqual(conditional.status_code, 304)
        self.assertEqual(conditional.content, b"")
        self.assertEqual(conditional.headers["etag"], expected_etag)

    def test_empty_registry_and_source_errors_are_typed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            empty_client = TestClient(
                create_app(
                    registry=load_registry(ViewerConfig(()), root),
                    repository_root=root,
                )
            )
            empty = empty_client.get("/api/sequences")

            registry = build_registry(root)
            client = TestClient(create_app(registry=registry, repository_root=root))
            unknown = client.get("/api/sequences/not-configured")
            out_of_range = client.get("/api/sequences/fixture-gt/frames/0")
            stale_responses = (
                client.get("/api/sequences/fixture-gt?source_hash=stale"),
                client.get(
                    "/api/sequences/fixture-gt/frames/1/observations?source_hash=stale"
                ),
                client.get("/api/sequences/fixture-gt/frames/1?source_hash=stale"),
            )

        self.assertEqual(
            empty.json(),
            {"sources": [], "unavailable": [], "diagnostics": []},
        )
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["error"]["code"], "unknown_source")
        self.assertEqual(out_of_range.status_code, 404)
        self.assertEqual(out_of_range.json()["error"]["code"], "frame_out_of_range")
        for response in stale_responses:
            with self.subTest(path=response.request.url.path):
                self.assertEqual(response.status_code, 412)
                self.assertEqual(response.json()["error"]["code"], "stale_source_hash")

    def test_held_frame_read_does_not_delay_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(root)
            read_started = threading.Event()
            release_read = threading.Event()

            def held_reader(path: Path) -> bytes:
                read_started.set()
                if not release_read.wait(timeout=1):
                    raise TimeoutError("test did not release frame read")
                return path.read_bytes()

            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    frame_reader=held_reader,
                )
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                frame_response = executor.submit(
                    client.get,
                    "/api/sequences/fixture-gt/frames/1",
                )
                self.assertTrue(read_started.wait(timeout=1))
                metadata_response = executor.submit(client.get, "/api/sequences/fixture-gt")
                metadata = metadata_response.result(timeout=1)
                release_read.set()
                frame = frame_response.result(timeout=1)

        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(frame.status_code, 200)

    def test_host_cors_and_extension_router_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(root)
            router = APIRouter()

            @router.get("/extension-check")
            async def extension_check() -> dict[str, bool]:
                return {"registered": True}

            development_client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    trusted_hosts=("viewer.local",),
                    development_origin="http://127.0.0.1:5173",
                    extension_routers=(router,),
                ),
                base_url="http://viewer.local",
            )
            allowed_host = development_client.get("/api/health")
            rejected_host = development_client.get(
                "/api/health",
                headers={"Host": "attacker.invalid"},
            )
            allowed_cors = development_client.options(
                "/api/health",
                headers={
                    "Origin": "http://127.0.0.1:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
            rejected_cors = development_client.options(
                "/api/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
            extension = development_client.get("/extension-check")

            production_client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    trusted_hosts=("viewer.local",),
                ),
                base_url="http://viewer.local",
            )
            production_cors = production_client.get(
                "/api/health",
                headers={"Origin": "http://127.0.0.1:5173"},
            )

        self.assertEqual(allowed_host.status_code, 200)
        self.assertEqual(rejected_host.status_code, 400)
        self.assertEqual(allowed_cors.status_code, 200)
        self.assertEqual(
            allowed_cors.headers["access-control-allow-origin"],
            "http://127.0.0.1:5173",
        )
        self.assertEqual(rejected_cors.status_code, 400)
        self.assertNotIn("access-control-allow-origin", rejected_cors.headers)
        self.assertEqual(extension.json(), {"registered": True})
        self.assertNotIn("access-control-allow-origin", production_cors.headers)

    def test_track_capability_guard_returns_typed_error_for_sentinel_only_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(root)
            unsupported = registry.sources[0]
            unsupported = unsupported.__class__(
                config=unsupported.config,
                sequence=unsupported.sequence,
                source_hash=unsupported.source_hash,
                source_rows=unsupported.source_rows,
                observations=unsupported.observations,
                capability=unsupported.capability.__class__(
                    id_status="sentinel_only",
                    track_features=False,
                    usable_track_ids=(),
                    diagnostics=unsupported.capability.diagnostics,
                ),
                indexes=unsupported.indexes,
            )
            router = APIRouter()

            @router.get("/capability-check")
            async def capability_check() -> dict[str, bool]:
                require_track_capability(unsupported)
                return {"supported": True}

            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(router,),
                )
            )
            response = client.get("/capability-check")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "unsupported_track_capability")

    def test_frontend_mount_uses_fixed_build_location_only_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(root)
            api_only_client = TestClient(create_app(registry=registry, repository_root=root))
            api_only = api_only_client.get("/viewer/sequence")

            distribution = root / "web" / "dist"
            distribution.mkdir(parents=True)
            (distribution / "index.html").write_text("<main>viewer</main>", encoding="utf-8")
            (distribution / "viewer.js").write_text("export const ready = true;", encoding="utf-8")
            static_client = TestClient(create_app(registry=registry, repository_root=root))
            index = static_client.get("/viewer/sequence")
            asset = static_client.get("/viewer.js")
            unknown_api = static_client.get("/api/not-a-route")

        self.assertEqual(api_only.status_code, 404)
        self.assertEqual(index.text, "<main>viewer</main>")
        self.assertEqual(asset.text, "export const ready = true;")
        self.assertEqual(unknown_api.status_code, 404)
        self.assertNotIn("<main>viewer</main>", unknown_api.text)

    def test_tracker_result_metadata_and_observations_preserve_defined_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_registry(
                root,
                key="fixture-result",
                sequence="MOT20-06",
                adapter="mot_result_10",
                rows="1,-1,1,2,3,4,0.75,-1,4.5,9\n",
            )
            client = TestClient(create_app(registry=registry, repository_root=root))

            detail = client.get("/api/sequences/fixture-result")
            response = client.get("/api/sequences/fixture-result/frames/1/observations")

        self.assertEqual(detail.json()["source_class"], "tracker_result")
        self.assertEqual(
            detail.json()["policy_classification"],
            "local_test_adapted_development_material",
        )
        self.assertEqual(detail.json()["capability"]["id_status"], "sentinel_only")
        observation = response.json()["observations"][0]
        self.assertEqual(observation["score"], 0.75)
        self.assertEqual(observation["score_semantics"], "tracker_score")
        self.assertIsNone(observation["ground_truth"])
        self.assertEqual(observation["ground_truth_semantics"], "not_defined")
        self.assertEqual(observation["opaque_result_fields"], [-1.0, 4.5, 9.0])


if __name__ == "__main__":
    unittest.main()