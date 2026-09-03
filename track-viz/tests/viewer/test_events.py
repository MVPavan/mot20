from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from test_tracks import build_track_registry

from mot20.viewer.api import create_app
from mot20.viewer.config import AdapterKind
from mot20.viewer.context import context_router
from mot20.viewer.events import event_router
from mot20.viewer.loaders import SourceRegistry


class EventApiTest(unittest.TestCase):
    def test_events_default_off_and_exact_thresholds_are_inclusive(self) -> None:
        rows = (
            "1,1,1,1,2,4,0.7,-1,-1,-1",
            "2,1,3,1,2,4,0.5,-1,-1,-1",
            "3,1,3,0,2,6,0.4,-1,-1,-1",
            "2,2,6,1,2,4,0.9,-1,-1,-1",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root, adapter="mot_result_10", rows=rows)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(event_router,),
                )
            )

            defaults = client.get(
                f"/api/sequences/fixture-gt/tracks/1/events?source_hash={source_hash}"
            )
            repeated_defaults = client.get(
                f"/api/sequences/fixture-gt/tracks/1/events?source_hash={source_hash}"
            )
            enabled = client.get(
                "/api/sequences/fixture-gt/tracks/1/events"
                "?enable_displacement=true&displacement_threshold=0.5"
                "&enable_scale_change=true&scale_change_threshold=0.5"
                "&enable_close_interaction=true&close_interaction_threshold=0.25"
                "&confidence_threshold=0.5"
            )
            below_motion_above_close = client.get(
                "/api/sequences/fixture-gt/tracks/1/events"
                "?enable_displacement=true&displacement_threshold=0.499999"
                "&enable_scale_change=true&scale_change_threshold=0.499999"
                "&enable_close_interaction=true&close_interaction_threshold=0.250001"
                "&confidence_threshold=0.499999"
            )
            above_motion_below_close = client.get(
                "/api/sequences/fixture-gt/tracks/1/events"
                "?enable_displacement=true&displacement_threshold=0.500001"
                "&enable_scale_change=true&scale_change_threshold=0.500001"
                "&enable_close_interaction=true&close_interaction_threshold=0.249999"
                "&confidence_threshold=0.500001"
            )

        self.assertEqual(defaults.status_code, 200)
        default_payload = defaults.json()
        self.assertEqual(default_payload["displacement_events"], [])
        self.assertEqual(default_payload["scale_change_events"], [])
        self.assertEqual(default_payload["close_interaction_events"], [])
        self.assertEqual(default_payload["confidence"]["status"], "meaningful")
        self.assertEqual(default_payload["confidence"]["threshold"], 0.5)
        self.assertEqual(repeated_defaults.json(), default_payload)
        self.assertEqual(
            [item["row_index"] for item in default_payload["low_confidence_observations"]],
            [2, 3],
        )

        self.assertEqual(enabled.status_code, 200)
        payload = enabled.json()
        self.assertEqual(payload["source_hash"], source_hash)
        self.assertEqual([event["to_frame"] for event in payload["displacement_events"]], [2])
        self.assertEqual(payload["displacement_events"][0]["normalized_displacement"], 0.5)
        self.assertEqual([event["to_frame"] for event in payload["scale_change_events"]], [3])
        self.assertEqual(payload["scale_change_events"][0]["normalized_scale_change"], 0.5)
        self.assertEqual([event["competitor_track_id"] for event in payload["close_interaction_events"]], [2])
        self.assertEqual(payload["close_interaction_events"][0]["normalized_edge_proximity"], 0.25)
        self.assertEqual(len(below_motion_above_close.json()["displacement_events"]), 1)
        self.assertEqual(len(below_motion_above_close.json()["scale_change_events"]), 1)
        self.assertEqual(len(below_motion_above_close.json()["close_interaction_events"]), 1)
        self.assertEqual(
            [item["row_index"] for item in below_motion_above_close.json()["low_confidence_observations"]],
            [3],
        )
        self.assertEqual(above_motion_below_close.json()["displacement_events"], [])
        self.assertEqual(above_motion_below_close.json()["scale_change_events"], [])
        self.assertEqual(above_motion_below_close.json()["close_interaction_events"], [])
        self.assertEqual(
            [item["row_index"] for item in above_motion_below_close.json()["low_confidence_observations"]],
            [2, 3],
        )

    def test_confidence_capability_variants_are_conservative_and_explicit(self) -> None:
        variants: tuple[tuple[str, AdapterKind, tuple[str, ...], str], ...] = (
            (
                "absent",
                "mot_gt_9",
                ("1,1,1,1,2,4,1,1,0.2", "2,1,1,1,2,4,1,1,0.9"),
                "confidence_not_defined",
            ),
            (
                "constant",
                "mot_result_10",
                ("1,1,1,1,2,4,0.8,-1,-1,-1", "2,1,1,1,2,4,0.8,-1,-1,-1"),
                "constant_confidence",
            ),
            (
                "sentinel",
                "mot_result_10",
                ("1,1,1,1,2,4,-1,-1,-1,-1", "2,1,1,1,2,4,-1,-1,-1,-1"),
                "sentinel_confidence",
            ),
        )
        for status, adapter, rows, diagnostic_code in variants:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                registry = build_track_registry(root, adapter=adapter, rows=rows)
                client = TestClient(
                    create_app(
                        registry=registry,
                        repository_root=root,
                        extension_routers=(event_router,),
                    )
                )

                response = client.get("/api/sequences/fixture-gt/tracks/1/events")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["confidence"]["status"], status)
            self.assertFalse(payload["confidence"]["meaningful"])
            self.assertEqual(payload["confidence"]["diagnostic"]["code"], diagnostic_code)
            self.assertEqual(payload["low_confidence_observations"], [])

    def test_displacement_across_a_gap_is_normalized_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(
                root,
                adapter="mot_result_10",
                rows=(
                    "1,1,1,1,2,4,0.8,-1,-1,-1",
                    "3,1,5,1,2,4,0.7,-1,-1,-1",
                ),
            )
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(event_router,),
                )
            )

            response = client.get(
                "/api/sequences/fixture-gt/tracks/1/events"
                "?enable_displacement=true&displacement_threshold=0.5"
            )

        event = response.json()["displacement_events"][0]
        self.assertEqual(event["frame_delta"], 2)
        self.assertEqual(event["center_displacement_pixels"], 4.0)
        self.assertEqual(event["normalized_displacement"], 0.5)

    def test_raw_event_families_preserve_contiguous_flagged_records_at_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(
                root,
                adapter="mot_result_10",
                rows=(
                    "1,1,0,0,2,4,0.9,-1,-1,-1",
                    "2,1,4,0,2,8,0.8,-1,-1,-1",
                    "3,1,8,0,2,16,0.7,-1,-1,-1",
                    "1,2,1,0,2,4,0.9,-1,-1,-1",
                    "2,2,5,0,2,8,0.9,-1,-1,-1",
                    "3,2,9,0,2,16,0.9,-1,-1,-1",
                ),
            )
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(event_router,),
                )
            )
            response = client.get(
                "/api/sequences/fixture-gt/tracks/1/events"
                "?enable_displacement=true&displacement_threshold=0.5"
                "&enable_scale_change=true&scale_change_threshold=0.5"
                "&enable_close_interaction=true&close_interaction_threshold=0"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            [(item["from_frame"], item["to_frame"], item["frame_delta"])
            for item in payload["displacement_events"]],
            [(1, 2, 1), (2, 3, 1)],
        )
        self.assertEqual(
            [(item["from_frame"], item["to_frame"], item["frame_delta"])
            for item in payload["scale_change_events"]],
            [(1, 2, 1), (2, 3, 1)],
        )
        self.assertEqual(
            [(item["frame"], item["competitor_track_id"], item["normalized_edge_proximity"])
            for item in payload["close_interaction_events"]],
            [(1, 2, 0.0), (2, 2, 0.0), (3, 2, 0.0)],
        )

    def test_events_are_hash_scoped_and_sentinel_only_tracks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tracked = build_track_registry(
                root,
                fixture_name="tracked",
                adapter="mot_result_10",
                rows=("1,1,1,1,2,4,0.8,-1,-1,-1",),
            )
            sentinel = build_track_registry(
                root,
                key="fixture-result",
                sequence="MOT20-06",
                fixture_name="sentinel",
                adapter="mot_result_10",
                rows=("1,-1,1,1,2,4,0.8,-1,-1,-1",),
            )
            registry = SourceRegistry(
                sources=(tracked.sources[0], sentinel.sources[0]),
                unavailable=(),
                diagnostics=tracked.diagnostics + sentinel.diagnostics,
            )
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(context_router, event_router),
                )
            )

            stale = client.get(
                "/api/sequences/fixture-gt/tracks/1/events?source_hash=stale"
            )
            unsupported = client.get(
                "/api/sequences/fixture-result/tracks/1/events"
            )
            unsupported_context = client.get(
                "/api/sequences/fixture-result/tracks/1/context?frame=1"
            )

        self.assertEqual(stale.status_code, 412)
        self.assertEqual(stale.json()["error"]["code"], "stale_source_hash")
        self.assertEqual(unsupported.status_code, 409)
        self.assertEqual(unsupported.json()["error"]["code"], "unsupported_track_capability")
        self.assertEqual(unsupported_context.status_code, 409)
        self.assertEqual(
            unsupported_context.json()["error"]["code"],
            "unsupported_track_capability",
        )


if __name__ == "__main__":
    unittest.main()
