from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from mot20.detection.crowdhuman import extract_crowdhuman_sources


class CrowdHumanExtractionTest(unittest.TestCase):
    def test_extracts_images_and_writes_a_verified_receipt(self) -> None:
        root = Path(tempfile.mkdtemp())
        train_archive = _write_archive(root / "train.zip", {"Images/train.jpg": "red"})
        val_archive = _write_archive(root / "val.zip", {"Images/val.jpg": "blue"})
        train_odgt = _write_odgt(root / "train.odgt", "train")
        val_odgt = _write_odgt(root / "val.odgt", "val")
        destination = root / "extracted"

        receipt = extract_crowdhuman_sources(
            [train_archive],
            val_archive,
            train_odgt,
            val_odgt,
            destination,
            expected_train_images=1,
            expected_val_images=1,
        )

        self.assertTrue((destination / "train" / "Images" / "train.jpg").is_file())
        self.assertTrue((destination / "val" / "Images" / "val.jpg").is_file())
        self.assertEqual(receipt["image_counts"], {"train": 1, "val": 1})
        self.assertEqual(json.loads((destination / "extraction.json").read_text(encoding="utf-8")), receipt)

    def test_refuses_an_existing_destination(self) -> None:
        root = Path(tempfile.mkdtemp())
        archive = _write_archive(root / "images.zip", {"Images/train.jpg": "red"})
        odgt = _write_odgt(root / "train.odgt", "train")
        destination = root / "extracted"
        destination.mkdir()

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            extract_crowdhuman_sources([archive], archive, odgt, odgt, destination, 1, 1)


def _write_archive(path: Path, images: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for member, color in images.items():
            payload = io.BytesIO()
            Image.new("RGB", (10, 10), color).save(payload, format="JPEG")
            archive.writestr(member, payload.getvalue())
    return path


def _write_odgt(path: Path, image_id: str) -> Path:
    path.write_text(json.dumps({"ID": image_id, "gtboxes": []}) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()