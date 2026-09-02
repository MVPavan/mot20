from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from test_tracks import build_track_registry

from mot20.viewer.api import create_app
from mot20.viewer.context import context_router
from mot20.viewer.loaders import SourceRegistry


class ContextApiTest(unittest.TestCase):
    def test_context_ranking_is_normalized_stable_and_capped_at_eight(self) -> None:
        rows = (
            "3,1,1,1,2,4,1,1,0.9",
            *(f"3,{track_id},3,1,2,4,1,1,0.9" for track_id in range(2, 12)),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root, rows=rows)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(context_router,),
                )
            )

            response = client.get(
                "/api/sequences/fixture-gt/tracks/1/context"
                f"?source_hash={source_hash}&frame=3&window_radius=0&count=8"
            )
            over_cap = client.get(
                "/api/sequences/fixture-gt/tracks/1/context?frame=3&count=9"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_hash"], source_hash)
        self.assertEqual(payload["window"], {"center_frame": 3, "start_frame": 3, "end_frame": 3, "radius": 0})
        self.assertEqual(payload["requested_count"], 8)
        self.assertEqual(payload["hard_cap"], 8)
        self.assertEqual(payload["total_competitors"], 10)
        self.assertEqual([item["track_id"] for item in payload["competitors"]], list(range(2, 10)))
        first_evidence = payload["competitors"][0]["evidence"][0]
        self.assertEqual(first_evidence["edge_distance_pixels"], 0.0)
        self.assertEqual(first_evidence["focal_box_height"], 4.0)
        self.assertEqual(first_evidence["normalized_edge_proximity"], 0.0)
        self.assertEqual(first_evidence["iou"], 0.0)
        self.assertEqual(over_cap.status_code, 422)

    def test_context_window_handles_track_gaps_and_never_crosses_sequences(self) -> None:
        first_rows = (
            "1,1,1,1,2,4,1,1,0.9",
            "1,2,2,1,2,4,1,1,0.9",
            "3,1,5,1,2,4,1,1,0.9",
            "3,3,6,1,2,4,1,1,0.9",
            "5,1,9,1,2,4,1,1,0.9",
            "5,4,10,1,2,4,1,1,0.9",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = build_track_registry(root, fixture_name="first", rows=first_rows)
            second = build_track_registry(
                root,
                key="fixture-second",
                sequence="MOT20-02",
                fixture_name="second",
                rows=(
                    "1,1,1,1,2,4,1,1,0.9",
                    "1,99,2,1,2,4,1,1,0.9",
                ),
            )
            registry = SourceRegistry(
                sources=(first.sources[0], second.sources[0]),
                unavailable=(),
                diagnostics=first.diagnostics + second.diagnostics,
            )
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(context_router,),
                )
            )

            boundary = client.get(
                "/api/sequences/fixture-gt/tracks/1/context?frame=1&window_radius=1"
            )
            gap = client.get(
                "/api/sequences/fixture-gt/tracks/1/context?frame=2&window_radius=0"
            )
            repeated = client.get(
                "/api/sequences/fixture-gt/tracks/1/context?frame=1&window_radius=1"
            )
            stale = client.get(
                "/api/sequences/fixture-gt/tracks/1/context?frame=1&source_hash=stale"
            )

        self.assertEqual(boundary.json()["window"]["start_frame"], 1)
        self.assertEqual([item["track_id"] for item in boundary.json()["competitors"]], [2])
        self.assertEqual(gap.json()["total_competitors"], 0)
        self.assertEqual(gap.json()["competitors"], [])
        self.assertEqual(repeated.json(), boundary.json())
        self.assertEqual(stale.status_code, 412)
        self.assertEqual(stale.json()["error"]["code"], "stale_source_hash")


if __name__ == "__main__":
    unittest.main()