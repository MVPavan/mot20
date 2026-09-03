from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import supervision as sv
from fastapi.testclient import TestClient
from PIL import Image

from mot20.viewer.api import create_app
from mot20.viewer.colors import track_color
from mot20.viewer.config import AdapterKind, Provenance, SourceConfig, ViewerConfig
from mot20.viewer.contracts import parse_gt_row
from mot20.viewer.exports import (
    ExportArtifactCollisionError,
    ExportParameters,
    ExportRenderer,
    export_router,
    write_track_video,
)
from mot20.viewer.loaders import SourceRegistry, load_registry
from scripts.export_track_video import main as export_track_main

APPLICATION_ORIGIN = "http://127.0.0.1:5173"


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


class ExportApiTest(unittest.TestCase):
    def test_focused_export_rejects_unsafe_preconditions_with_typed_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source_hash = registry.sources[0].source_hash
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    application_origin=APPLICATION_ORIGIN,
                    extension_routers=(export_router,),
                )
            )
            request = {
                "source_hash": source_hash,
                "track_id": 7,
                "start_frame": 1,
                "end_frame": 3,
            }

            missing_hash = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json={**request, "source_hash": None},
            )
            stale_hash = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json={**request, "source_hash": "stale"},
            )
            wrong_origin = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": "http://attacker.invalid"},
                json=request,
            )
            over_cap = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json={**request, "end_frame": 301},
            )
            out_of_range = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json={**request, "start_frame": 0},
            )

            sentinel_registry = build_track_registry(
                root,
                key="fixture-result",
                sequence="MOT20-06",
                fixture_name="sentinel",
                adapter="mot_result_10",
                rows=("1,-1,1,1,4,5,0.8,-1,-1,-1",),
            )
            sentinel_client = TestClient(
                create_app(
                    registry=sentinel_registry,
                    repository_root=root,
                    application_origin=APPLICATION_ORIGIN,
                    extension_routers=(export_router,),
                )
            )
            unsupported = sentinel_client.post(
                "/api/sequences/fixture-result/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json={**request, "source_hash": sentinel_registry.sources[0].source_hash},
            )

        expected = (
            (missing_hash, 428, "missing_source_hash"),
            (stale_hash, 412, "stale_source_hash"),
            (wrong_origin, 403, "invalid_export_origin"),
            (over_cap, 413, "interactive_export_frame_cap_exceeded"),
            (out_of_range, 422, "invalid_export_frame_range"),
            (unsupported, 409, "unsupported_track_capability"),
        )
        for response, status_code, code in expected:
            with self.subTest(code=code):
                self.assertEqual(response.status_code, status_code)
                self.assertEqual(response.json()["error"]["code"], code)

    def test_focused_export_is_atomic_hash_keyed_reproducible_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source = registry.sources[0]
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    application_origin=APPLICATION_ORIGIN,
                    extension_routers=(export_router,),
                )
            )
            request = {
                "source_hash": source.source_hash,
                "track_id": 7,
                "start_frame": 1,
                "end_frame": 3,
                "context_count": 1,
                "trace_length": 2,
            }

            first = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json=request,
            )
            second = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json=request,
            )

            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json()["status"], "created")
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["status"], "existing")
            self.assertEqual(first.json()["export_id"], second.json()["export_id"])
            artifact_directory = root / first.json()["artifact_directory"]
            video_path = root / first.json()["video_path"]
            metadata_path = root / first.json()["metadata_path"]
            self.assertEqual(artifact_directory.parent, root / "track-viz/artifacts/exports")
            self.assertTrue(video_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertFalse(any(path.name.startswith(".tmp-") for path in artifact_directory.parent.iterdir()))

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["export_id"], first.json()["export_id"])
            self.assertEqual(metadata["kind"], "focused_clip")
            self.assertEqual(metadata["source"]["annotation_result_sha256"], source.source_hash)
            self.assertEqual([item["frame"] for item in metadata["source"]["frames"]], [1, 2, 3])
            self.assertTrue(all(len(item["image_sha256"]) == 64 for item in metadata["source"]["frames"]))
            self.assertTrue(metadata["source"]["frames"][0]["observation_row_hashes"])
            self.assertEqual(metadata["selection"]["track_id"], 7)
            self.assertEqual(metadata["render"]["geometry_basis"], "display_clamped_xyxy")
            self.assertEqual(metadata["render"]["focal_color"]["hex"], track_color("MOT20-01", 7).hex)
            self.assertEqual(metadata["render"]["annotators"]["trace_color_lookup"], "track")
            self.assertEqual(metadata["incoming_provenance"]["producer"], "synthetic")
            self.assertEqual(metadata["policy_classification"], "ground_truth_training_source")
            self.assertEqual(len(metadata["output"]["sha256"]), 64)

            video_info = sv.VideoInfo.from_video_path(str(video_path))
            self.assertEqual(video_info.total_frames, 3)
            self.assertEqual(video_info.resolution_wh, (16, 12))
            self.assertEqual(video_info.fps, 25)

    def test_annotation_change_after_startup_is_rejected_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_track_registry(root)
            source = registry.sources[0]
            client = TestClient(
                create_app(
                    registry=registry,
                    repository_root=root,
                    application_origin=APPLICATION_ORIGIN,
                    extension_routers=(export_router,),
                )
            )
            annotation_path = root / source.config.annotations
            annotation_path.write_text(
                annotation_path.read_text(encoding="utf-8") + "6,9,1,1,2,2,1,1,0.9\n",
                encoding="utf-8",
            )

            response = client.post(
                "/api/sequences/fixture-gt/exports",
                headers={"Origin": APPLICATION_ORIGIN},
                json={
                    "source_hash": source.source_hash,
                    "track_id": 7,
                    "start_frame": 1,
                    "end_frame": 3,
                },
            )

            self.assertEqual(response.status_code, 412)
            self.assertEqual(response.json()["error"]["code"], "source_result_changed")
            self.assertFalse((root / "track-viz/artifacts/exports").exists())

    def test_test_sequence_export_metadata_retains_adapted_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = build_track_registry(
                root,
                key="fixture-result",
                sequence="MOT20-06",
                adapter="mot_result_10",
                rows=(
                    "1,7,1,1,4,5,0.8,-1,-1,-1",
                    "2,7,2,1,4,5,0.8,-1,-1,-1",
                ),
            ).sources[0]

            artifact = write_track_video(
                source=source,
                repository_root=root,
                parameters=ExportParameters(track_id=7, start_frame=1, end_frame=2),
                kind="offline_track_video",
                frame_limit=2,
            )

            metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["policy_classification"],
                "local_test_adapted_development_material",
            )

    def test_corrupt_collision_is_typed_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = build_track_registry(root).sources[0]
            parameters = ExportParameters(track_id=7, start_frame=1, end_frame=2)
            artifact = write_track_video(
                source=source,
                repository_root=root,
                parameters=parameters,
                kind="focused_clip",
                frame_limit=300,
            )
            original_video = artifact.video_path.read_bytes()
            artifact.metadata_path.unlink()

            with self.assertRaises(ExportArtifactCollisionError):
                write_track_video(
                    source=source,
                    repository_root=root,
                    parameters=parameters,
                    kind="focused_clip",
                    frame_limit=300,
                )

            self.assertEqual(artifact.video_path.read_bytes(), original_video)
            self.assertFalse(artifact.metadata_path.exists())

    def test_partial_render_failure_leaves_no_final_or_temporary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = build_track_registry(root).sources[0]
            successful_reads = 0

            def failing_reader(path: Path) -> bytes:
                nonlocal successful_reads
                successful_reads += 1
                if successful_reads > 3:
                    raise OSError("synthetic frame failure")
                return path.read_bytes()

            with self.assertRaises(OSError):
                write_track_video(
                    source=source,
                    repository_root=root,
                    parameters=ExportParameters(track_id=7, start_frame=1, end_frame=3),
                    kind="focused_clip",
                    frame_limit=300,
                    frame_reader=failing_reader,
                )

            export_root = root / "track-viz/artifacts/exports"
            self.assertTrue(export_root.is_dir())
            self.assertEqual(tuple(export_root.iterdir()), ())


class ExportRenderParityTest(unittest.TestCase):
    def test_focal_color_geometry_and_bounded_track_trace_match_browser_contract(self) -> None:
        observation = parse_gt_row(
            "1,7,10,20,30,25,1,1,0.9",
            source_key="fixture-gt",
            sequence="MOT20-01",
            row_index=1,
            source_hash="source-hash",
            image_width=100,
            image_height=60,
            sequence_length=2,
        )
        context_observation = parse_gt_row(
            "1,8,50,20,40,25,1,1,0.9",
            source_key="fixture-gt",
            sequence="MOT20-01",
            row_index=2,
            source_hash="source-hash",
            image_width=100,
            image_height=60,
            sequence_length=2,
        )
        renderer = ExportRenderer(
            sequence="MOT20-01",
            focal_track_id=7,
            context_track_ids=(8,),
            trace_length=17,
        )

        rendered = renderer.annotate(
            np.zeros((60, 100, 3), dtype=np.uint8),
            (observation, context_observation),
        )

        expected_bgr = np.asarray(track_color("MOT20-01", 7).rgb[::-1], dtype=np.uint8)
        context_bgr = np.asarray(track_color("MOT20-01", 8).rgb[::-1], dtype=np.uint8)
        np.testing.assert_array_equal(rendered[30, 10], expected_bgr)
        np.testing.assert_array_equal(rendered[30, 40], expected_bgr)
        np.testing.assert_array_equal(rendered[30, 7], np.zeros(3, dtype=np.uint8))
        np.testing.assert_array_equal(rendered[20, 50], context_bgr)
        np.testing.assert_array_equal(rendered[20, 70], np.zeros(3, dtype=np.uint8))
        self.assertEqual(renderer.trace_annotator.color_lookup, sv.ColorLookup.TRACK)
        self.assertEqual(renderer.trace_annotator.trace.max_size, 17)


class OfflineExportCliTest(unittest.TestCase):
    @patch("scripts.export_track_video.write_track_video")
    @patch("scripts.export_track_video.load_registry")
    @patch("scripts.export_track_video.load_config")
    def test_offline_export_allows_a_configured_limit_above_interactive_cap(
        self,
        config_loader: Mock,
        registry_loader: Mock,
        video_writer: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = build_track_registry(root).sources[0]
            source = replace(
                source,
                sequence=replace(source.sequence, length=500),
            )
            registry_loader.return_value = replace(
                build_track_registry(root, fixture_name="second"),
                sources=(source,),
            )
            artifact_directory = root / "track-viz/artifacts/exports/export-id"
            video_writer.return_value = Mock(
                export_id="export-id",
                status="created",
                artifact_directory=artifact_directory,
                video_path=artifact_directory / "track.mp4",
                metadata_path=artifact_directory / "metadata.json",
            )

            exit_code = export_track_main(
                [
                    "fixture-gt",
                    "7",
                    "--repository-root",
                    str(root),
                    "--start-frame",
                    "1",
                    "--end-frame",
                    "301",
                    "--max-frames",
                    "301",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(video_writer.call_args.kwargs["kind"], "offline_track_video")
        self.assertEqual(video_writer.call_args.kwargs["frame_limit"], 301)
        self.assertEqual(video_writer.call_args.kwargs["parameters"].end_frame, 301)
        config_loader.assert_called_once()

    @patch("scripts.export_track_video.write_track_video")
    def test_offline_export_rejects_a_limit_above_the_hard_cap(self, video_writer: Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory, self.assertRaises(SystemExit):
            export_track_main(
                [
                    "fixture-gt",
                    "7",
                    "--repository-root",
                    temporary_directory,
                    "--max-frames",
                    "100001",
                ]
            )
        video_writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()