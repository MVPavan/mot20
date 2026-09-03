from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mot20.viewer.config import Provenance, SourceConfig, ViewerConfig
from mot20.viewer.loaders import SourceError, load_registry, load_source


class SourceLoaderTest(unittest.TestCase):
    def test_ground_truth_source_preserves_rows_and_derives_tracked_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence_root = root / "fixture" / "MOT20-01"
            image_root = sequence_root / "img1"
            image_root.mkdir(parents=True)
            (sequence_root / "seqinfo.ini").write_text(
                """[Sequence]
name=MOT20-01
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
                Image.new("RGB", (8, 6), color=(frame, 0, 0)).save(image_root / f"{frame:06d}.jpg", format="JPEG")
            annotation_bytes = b"1,7,0,0,3,4,1,1,0.8\n2,7,1,1,3,4,1,7,0.5\n"
            (sequence_root / "gt.txt").write_bytes(annotation_bytes)
            source = SourceConfig(
                key="fixture-gt",
                sequence="MOT20-01",
                seqinfo="fixture/MOT20-01/seqinfo.ini",
                images="fixture/MOT20-01/img1",
                annotations="fixture/MOT20-01/gt.txt",
                adapter="mot_gt_9",
                provenance=Provenance(producer="synthetic"),
            )

            loaded = load_source(source, root)

        self.assertEqual(loaded.sequence.image_names, ("000001.jpg", "000002.jpg", "000003.jpg"))
        self.assertEqual(loaded.source_hash, hashlib.sha256(annotation_bytes).hexdigest())
        self.assertEqual(len(loaded.source_rows), 2)
        self.assertEqual([row.row_index for row in loaded.source_rows], [1, 2])
        self.assertEqual([row.row_index for row in loaded.observations], [1])
        self.assertEqual(loaded.capability.id_status, "tracked")
        self.assertTrue(loaded.capability.track_features)
        self.assertEqual([row.row_index for row in loaded.indexes.tracks[7]], [1])
        self.assertEqual(loaded.indexes.frames[3], ())

    def test_registry_distinguishes_absent_invalid_and_empty_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing = SourceConfig(
                key="missing",
                sequence="MOT20-01",
                seqinfo="missing/seqinfo.ini",
                images="missing/img1",
                annotations="missing/gt.txt",
                adapter="mot_gt_9",
                provenance=Provenance(),
            )
            unavailable_registry = load_registry(ViewerConfig((missing,)), root)
            empty_registry = load_registry(ViewerConfig(()), root)

            invalid_root = root / "invalid"
            (invalid_root / "img1").mkdir(parents=True)
            (invalid_root / "seqinfo.ini").write_text("not seqinfo", encoding="utf-8")
            (invalid_root / "rows.txt").write_text("1,1,0,0,1,1,1,1,1\n", encoding="utf-8")
            invalid = SourceConfig(
                key="invalid",
                sequence="MOT20-01",
                seqinfo="invalid/seqinfo.ini",
                images="invalid/img1",
                annotations="invalid/rows.txt",
                adapter="mot_gt_9",
                provenance=Provenance(),
            )

            with self.assertRaises(SourceError):
                load_registry(ViewerConfig((invalid,)), root)

        self.assertEqual(unavailable_registry.sources, ())
        self.assertEqual(unavailable_registry.unavailable[0].config.key, "missing")
        self.assertEqual(unavailable_registry.unavailable[0].diagnostic.code, "source_unavailable")
        self.assertEqual(empty_registry.sources, ())
        self.assertEqual(empty_registry.unavailable, ())

    def test_jpeg_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            root = workspace / "repository"
            sequence_root = root / "fixture"
            image_root = sequence_root / "img1"
            image_root.mkdir(parents=True)
            (sequence_root / "seqinfo.ini").write_text(
                """[Sequence]
name=MOT20-01
imDir=img1
frameRate=25
seqLength=1
imWidth=8
imHeight=6
imExt=.jpg
""",
                encoding="utf-8",
            )
            (sequence_root / "rows.txt").write_text("1,1,0,0,3,4,1,1,0.8\n", encoding="utf-8")
            outside_image = workspace / "outside.jpg"
            Image.new("RGB", (8, 6)).save(outside_image, format="JPEG")
            (image_root / "000001.jpg").symlink_to(outside_image)
            source = SourceConfig(
                key="escaped-image",
                sequence="MOT20-01",
                seqinfo="fixture/seqinfo.ini",
                images="fixture/img1",
                annotations="fixture/rows.txt",
                adapter="mot_gt_9",
                provenance=Provenance(),
            )

            with self.assertRaisesRegex(SourceError, "escapes configured image directory"):
                load_source(source, root)

    def test_jpeg_names_counts_and_dimensions_must_match_seqinfo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence_root = root / "fixture"
            image_root = sequence_root / "img1"
            image_root.mkdir(parents=True)
            (sequence_root / "seqinfo.ini").write_text(
                """[Sequence]
name=MOT20-01
imDir=img1
frameRate=25
seqLength=1
imWidth=8
imHeight=6
imExt=.jpg
""",
                encoding="utf-8",
            )
            (sequence_root / "rows.txt").write_text("1,1,0,0,3,4,1,1,0.8\n", encoding="utf-8")
            source = SourceConfig(
                key="invalid-image",
                sequence="MOT20-01",
                seqinfo="fixture/seqinfo.ini",
                images="fixture/img1",
                annotations="fixture/rows.txt",
                adapter="mot_gt_9",
                provenance=Provenance(),
            )

            Image.new("RGB", (8, 6)).save(image_root / "000002.jpg", format="JPEG")
            with self.assertRaisesRegex(SourceError, "image count or names"):
                load_source(source, root)

            (image_root / "000002.jpg").unlink()
            Image.new("RGB", (9, 6)).save(image_root / "000001.jpg", format="JPEG")
            with self.assertRaisesRegex(SourceError, "dimensions or format"):
                load_source(source, root)


if __name__ == "__main__":
    unittest.main()