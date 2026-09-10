from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mot20.detection.probe_dataset import assemble_dense_probe_dataset


class DenseProbeDatasetTest(unittest.TestCase):
    def test_selects_dense_images_preserves_an_ignored_example_and_links_roots(self) -> None:
        root = Path(tempfile.mkdtemp())
        source = root / "source"
        _write_split(source, "train", "mot20_train", [3, 1, 2], [0, 1, 0], ["MOT20", "MOT20", "Byte65"])
        _write_split(source, "valid", "mot20_val", [1, 2], [0, 0])
        destination = root / "probe"

        assemble_dense_probe_dataset(source, destination, train_images=2, valid_images=1, required_train_source="Byte65")

        train = _read(destination / "train" / "_annotations.coco.json")
        self.assertEqual([image["id"] for image in train["images"]], [2, 3])
        self.assertTrue(any(annotation["iscrowd"] for annotation in train["annotations"]))
        self.assertTrue(any(image["source_dataset"] == "Byte65" for image in train["images"]))
        self.assertTrue((destination / "train" / "mot20_train").is_symlink())
        self.assertTrue((destination / "train" / "mot20_train" / "MOT20-01" / "img1" / "000001.jpg").is_file())


def _write_split(
    root: Path,
    split: str,
    prefix: str,
    positive_counts: list[int],
    ignored_counts: list[int],
    sources: list[str] | None = None,
) -> None:
    image_root = root / split / prefix / "MOT20-01" / "img1"
    image_root.mkdir(parents=True)
    images = []
    annotations = []
    annotation_id = 1
    sources = sources or ["MOT20"] * len(positive_counts)
    for image_id, (positive, ignored, source) in enumerate(zip(positive_counts, ignored_counts, sources, strict=True), start=1):
        image_name = f"{image_id:06d}.jpg"
        Image.new("RGB", (100, 80), "red").save(image_root / image_name)
        images.append(
            {
                "id": image_id,
                "file_name": f"{prefix}/MOT20-01/img1/{image_name}",
                "width": 100,
                "height": 80,
                "source_dataset": source,
                "source_sequence": "MOT20-01",
                "source_frame_id": image_id + (10 if split == "valid" else 0),
            }
        )
        for iscrowd, count in ((0, positive), (1, ignored)):
            for _ in range(count):
                annotations.append(
                    {"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": iscrowd}
                )
                annotation_id += 1
    split_root = root / split
    (split_root / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "pedestrian"}]}),
        encoding="utf-8",
    )


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()