from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mot20.detection.dataset_audit import audit_rfdetr_coco_dataset


class RfDetrCocoDatasetAuditTest(unittest.TestCase):
    def test_accepts_images_under_linked_source_root(self) -> None:
        root = Path(tempfile.mkdtemp())
        source_image = _write_image(root / "sources" / "MOT20-01" / "img1" / "000001.jpg", "red")
        linked_root = root / "train" / "mot20_train"
        linked_root.parent.mkdir(parents=True)
        linked_root.symlink_to(source_image.parents[2], target_is_directory=True)
        _write_manifest(root / "train", [linked_root / "MOT20-01" / "img1" / "000001.jpg"], [(1, 0)])
        valid_image = _write_image(root / "valid" / "mot20_val" / "MOT20-01" / "img1" / "000002.jpg", "blue")
        _write_manifest(root / "valid", [valid_image], [(1, 0)])

        audit = audit_rfdetr_coco_dataset(root)

        self.assertEqual(audit["split_counts"]["train"]["images"], 1)

    def test_reports_counts_density_and_safe_query_floor(self) -> None:
        root = Path(tempfile.mkdtemp())
        train_image = _write_image(root / "train" / "mot20_train" / "MOT20-01" / "img1" / "000001.jpg", "red")
        valid_image = _write_image(root / "valid" / "mot20_val" / "MOT20-01" / "img1" / "000002.jpg", "blue")
        _write_manifest(root / "train", [train_image], [(1, 0), (1, 0), (1, 1)])
        _write_manifest(root / "valid", [valid_image], [(1, 0)])

        audit = audit_rfdetr_coco_dataset(root, group_detr=13)

        self.assertEqual(audit["split_counts"], {"train": {"images": 1, "annotations": 3}, "valid": {"images": 1, "annotations": 1}})
        self.assertEqual(audit["density"]["train"]["positive"]["maximum"], 2)
        self.assertEqual(audit["density"]["train"]["ignored"]["maximum"], 1)
        self.assertEqual(audit["density"]["valid"]["positive"]["maximum"], 1)
        self.assertEqual(audit["query_capacity"]["minimum_num_queries"], 13)
        self.assertEqual(audit["cross_split_duplicate_images"], [])

    def test_rejects_cross_split_duplicate_image_bytes(self) -> None:
        root = Path(tempfile.mkdtemp())
        train_image = _write_image(root / "train" / "mot20_train" / "MOT20-01" / "img1" / "000001.jpg", "red")
        valid_image = root / "valid" / "mot20_val" / "MOT20-01" / "img1" / "000002.jpg"
        valid_image.parent.mkdir(parents=True)
        shutil.copyfile(train_image, valid_image)
        _write_manifest(root / "train", [train_image], [(1, 0)])
        _write_manifest(root / "valid", [valid_image], [(1, 0)])

        with self.assertRaisesRegex(ValueError, "cross-split duplicate image bytes"):
            audit_rfdetr_coco_dataset(root)

    def test_rejects_mot20_frame_in_both_splits(self) -> None:
        root = Path(tempfile.mkdtemp())
        train_image = _write_image(root / "train" / "mot20_train" / "MOT20-01" / "img1" / "000001.jpg", "red")
        valid_image = _write_image(root / "valid" / "mot20_val" / "MOT20-01" / "img1" / "000001.jpg", "blue")
        _write_manifest(root / "train", [train_image], [(1, 0)], source_frame_id=1)
        _write_manifest(root / "valid", [valid_image], [(1, 0)], source_frame_id=1)

        with self.assertRaisesRegex(ValueError, "MOT20 temporal overlap"):
            audit_rfdetr_coco_dataset(root)


def _write_image(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color).save(path)
    return path


def _write_manifest(
    split_root: Path,
    image_paths: list[Path],
    annotations: list[tuple[int, int]],
    source_frame_id: int | None = None,
) -> None:
    split = split_root.name
    images = [
        {
            "id": index,
            "file_name": image_path.relative_to(split_root).as_posix(),
            "width": 100,
            "height": 80,
            "source_dataset": "MOT20",
            "source_sequence": "MOT20-01",
            "source_frame_id": source_frame_id if source_frame_id is not None else index + (0 if split == "train" else 1),
        }
        for index, image_path in enumerate(image_paths, start=1)
    ]
    annotation_records = [
        {
            "id": index,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": iscrowd,
        }
        for index, (image_id, iscrowd) in enumerate(annotations, start=1)
    ]
    (split_root / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotation_records, "categories": [{"id": 1, "name": "pedestrian"}]}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()