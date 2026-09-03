from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from mot20.viewer.api import create_app
from mot20.viewer.config import AdapterKind, Provenance, SourceConfig, ViewerConfig
from mot20.viewer.filmstrip import filmstrip_router
from mot20.viewer.loaders import SourceRegistry, load_registry
from mot20.viewer.tracks import track_router


def build_track_registry(
    root: Path,
    *,
    key: str = "fixture-gt",
    sequence: str = "MOT20-01",
    fixture_name: str = "fixture",
    adapter: AdapterKind = "mot_gt_9",
    rows: tuple[str, ...] | None = None,
) -> SourceRegistry:
    sequence_root = root / fixture_name / "MOT20-01"
    image_root = sequence_root / "img1"
    image_root.mkdir(parents=True)
    (sequence_root / "seqinfo.ini").write_text(
        f"""[Sequence]
name={sequence}
imDir=img1
frameRate=25
seqLength=6
imWidth=16
imHeight=12
imExt=.jpg
""",
        encoding="utf-8",
    )
    for frame in range(1, 7):
        Image.new("RGB", (16, 12), color=(frame * 10, 20, 30)).save(
            image_root / f"{frame:06d}.jpg",
            format="JPEG",
        )
    source_rows = rows or (
        "1,7,-1,-2,4,5,1,1,0.9",
        "1,8,2,2,5,6,1,1,0.8",
        "2,7,2,1,4,5,1,1,0.9",
        "3,8,4,2,5,6,1,1,0.8",
        "3,7,4,1,4,5,1,1,0.9",
        "6,8,8,2,5,6,1,1,0.8",
    )
    (sequence_root / "gt.txt").write_text(
        "\n".join(source_rows) + "\n",
        encoding="utf-8",
    )
    source = SourceConfig(
        key=key,
        sequence=sequence,
        seqinfo=f"{fixture_name}/MOT20-01/seqinfo.ini",
        images=f"{fixture_name}/MOT20-01/img1",
        annotations=f"{fixture_name}/MOT20-01/gt.txt",
        adapter=adapter,
        provenance=Provenance(producer="synthetic"),
    )
    return load_registry(ViewerConfig((source,)), root)


class TrackApiTest(unittest.TestCase):
    def test_track_lookup_reports_exact_sequence_local_evidence_and_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(track_router,),
                )
            )

            response = client.get(
                "/api/sequences/fixture-gt/tracks/8?current_row_index=4"
            )
            continuous = client.get("/api/sequences/fixture-gt/tracks/7")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_key"], "fixture-gt")
        self.assertEqual(payload["sequence"], "MOT20-01")
        self.assertEqual(payload["track_id"], 8)
        self.assertEqual(payload["observation_frames"], [1, 3, 6])
        self.assertEqual(
            payload["gaps"],
            [
                {"start_frame": 2, "end_frame": 2, "length": 1},
                {"start_frame": 4, "end_frame": 5, "length": 2},
            ],
        )
        self.assertEqual(payload["first_observation"]["row_index"], 2)
        self.assertEqual(payload["last_observation"]["row_index"], 6)
        self.assertEqual(payload["previous_observation"]["frame"], 1)
        self.assertEqual(payload["next_observation"]["frame"], 6)
        self.assertEqual(continuous.json()["observation_frames"], [1, 2, 3])
        self.assertEqual(continuous.json()["gaps"], [])

    def test_exact_track_search_missing_ids_boundaries_and_sequence_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = build_track_registry(root, fixture_name="first")
            second = build_track_registry(
                root,
                key="fixture-second",
                sequence="MOT20-02",
                fixture_name="second",
                rows=("2,8,1,1,4,5,1,1,0.9",),
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
                    extension_routers=(track_router,),
                )
            )

            exact = client.get("/api/sequences/fixture-gt/tracks?track_id=8")
            non_match = client.get("/api/sequences/fixture-gt/tracks?track_id=80")
            partial = client.get("/api/sequences/fixture-gt/tracks?track_id=8x")
            first_boundary = client.get(
                "/api/sequences/fixture-gt/tracks/8?current_row_index=2"
            )
            last_boundary = client.get(
                "/api/sequences/fixture-gt/tracks/8?current_row_index=6"
            )
            second_sequence = client.get("/api/sequences/fixture-second/tracks/8")

        self.assertEqual(exact.status_code, 200)
        self.assertEqual(exact.json()["track_id"], 8)
        self.assertEqual(non_match.status_code, 404)
        self.assertEqual(non_match.json()["error"]["code"], "track_not_found")
        self.assertEqual(partial.status_code, 422)
        self.assertIsNone(first_boundary.json()["previous_observation"])
        self.assertEqual(first_boundary.json()["next_observation"]["row_index"], 4)
        self.assertEqual(last_boundary.json()["previous_observation"]["row_index"], 4)
        self.assertIsNone(last_boundary.json()["next_observation"])
        self.assertEqual(second_sequence.json()["sequence"], "MOT20-02")
        self.assertEqual(second_sequence.json()["observation_frames"], [2])

    def test_sentinel_only_source_rejects_all_track_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(
                root,
                key="fixture-result",
                sequence="MOT20-06",
                adapter="mot_result_10",
                rows=("1,-1,1,1,4,5,0.8,-1,-1,-1",),
            )
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(track_router, filmstrip_router),
                )
            )

            responses = (
                client.get("/api/sequences/fixture-result/tracks/1"),
                client.get("/api/sequences/fixture-result/tracks?track_id=1"),
                client.get(
                    "/api/sequences/fixture-result/tracks/1/filmstrip?current_row_index=1"
                ),
            )

        for response in responses:
            with self.subTest(path=response.request.url.path):
                self.assertEqual(response.status_code, 409)
                self.assertEqual(
                    response.json()["error"]["code"],
                    "unsupported_track_capability",
                )

    def test_filmstrip_requires_current_row_to_belong_to_exact_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    extension_routers=(filmstrip_router,),
                )
            )

            response = client.get(
                "/api/sequences/fixture-gt/tracks/8/filmstrip?current_row_index=4"
            )
            wrong_track = client.get(
                "/api/sequences/fixture-gt/tracks/8/filmstrip?current_row_index=1"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_observations"], 3)
        self.assertEqual(payload["sampled_count"], 3)
        self.assertEqual(
            [sample["observation"]["row_index"] for sample in payload["samples"]],
            [2, 4, 6],
        )
        self.assertEqual(
            [sample["is_current"] for sample in payload["samples"]],
            [False, True, False],
        )
        self.assertEqual(wrong_track.status_code, 404)
        self.assertEqual(
            wrong_track.json()["error"]["code"],
            "current_observation_not_found",
        )


if __name__ == "__main__":
    unittest.main()