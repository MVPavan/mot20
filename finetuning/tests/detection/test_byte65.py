from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from mot20.detection.byte65 import materialize_byte65_yolo_dataset


class Byte65MaterializationTest(unittest.TestCase):
    def test_uses_post_modification_labels_and_clips_boundary_boxes(self) -> None:
        root = Path(tempfile.mkdtemp())
        archive_path = root / "byte65.zip"
        dataset_path = root / "byte65nms_68seq"
        with zipfile.ZipFile(archive_path, "w") as archive:
            image = io.BytesIO()
            Image.new("RGB", (100, 80)).save(image, format="JPEG")
            archive.writestr("byte65-modified-images-yolo/images/MOT20-06_000001.jpg", image.getvalue())
            archive.writestr(
                "byte65-modified-images-yolo/post_modification_annotations/MOT20-06_000001.txt",
                "0 0.5 0.5 0.2 0.25\n0 0.5 0.95 0.2 0.2\n",
            )

        audit = materialize_byte65_yolo_dataset(archive_path, dataset_path, "byte65nms_68seq")

        self.assertEqual(audit["image_count"], 1)
        self.assertEqual(audit["annotation_count"], 2)
        self.assertEqual(audit["clipped_annotation_count"], 1)
        self.assertTrue((dataset_path / "images" / "MOT20-06" / "000001.jpg").is_file())
        self.assertEqual(
            (dataset_path / "labels" / "MOT20-06" / "000001.txt").read_text(encoding="utf-8").splitlines(),
            ["0 0.5 0.5 0.2 0.25", "0 0.5 0.925 0.2 0.15"],
        )
        manifest = json.loads((dataset_path / "annotations.coco.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["images"][0]["file_name"], "images/MOT20-06/000001.jpg")
        self.assertEqual(manifest["annotations"][1]["bbox"], [40.0, 68.0, 20.0, 12.0])
        self.assertEqual(manifest["metadata"]["dataset_name"], "byte65nms_68seq")
        self.assertEqual(json.loads((dataset_path / "audit.json").read_text(encoding="utf-8"))["annotation_count"], 2)

    def test_refuses_to_overwrite_an_existing_dataset(self) -> None:
        root = Path(tempfile.mkdtemp())
        archive_path = root / "byte65.zip"
        archive_path.touch()
        dataset_path = root / "byte65nms_68seq"
        dataset_path.mkdir()

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            materialize_byte65_yolo_dataset(archive_path, dataset_path, "byte65nms_68seq")


if __name__ == "__main__":
    unittest.main()