import configparser
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mot20.detection.coco_conversion import (
    assemble_rfdetr_coco_dataset,
    convert_crowdhuman_split,
    convert_mot20_split,
    merge_bytetrack_mot20_crowdhuman,
    merge_byte65_test_adapted_overlay,
    write_coco_manifest,
)


class Mot20CocoConversionTest(unittest.TestCase):
    def test_rejects_an_image_whose_dimensions_disagree_with_seqinfo(self) -> None:
        dataset_root = Path(tempfile.mkdtemp()) / "MOT20" / "train"
        sequence_root = dataset_root / "MOT20-01"
        image_root = sequence_root / "img1"
        image_root.mkdir(parents=True)
        Image.new("RGB", (99, 80)).save(image_root / "000001.jpg")
        _write_seqinfo(sequence_root, length=1, width=100, height=80)
        (sequence_root / "gt").mkdir()
        (sequence_root / "gt" / "gt.txt").write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "MOT20 image dimensions do not match"):
            convert_mot20_split(dataset_root, "train_half")

    def test_train_half_matches_bytetrack_boundaries_and_preserves_ignored_people(self) -> None:
        dataset_root = Path(tempfile.mkdtemp()) / "MOT20" / "train"
        sequence_root = dataset_root / "MOT20-01"
        image_root = sequence_root / "img1"
        image_root.mkdir(parents=True)
        for frame in range(1, 5):
            Image.new("RGB", (100, 80)).save(image_root / f"{frame:06d}.jpg")
        _write_seqinfo(sequence_root, length=4, width=100, height=80)
        (sequence_root / "gt").mkdir()
        (sequence_root / "gt" / "gt.txt").write_text(
            "1,1,10,10,20,20,1,1,1\n"
            "1,2,50,10,20,20,0,7,1\n"
            "2,3,10,10,20,20,0,6,1\n"
            "3,4,-5,10,20,20,1,1,1\n"
            "3,5,200,10,20,20,1,1,1\n"
            "4,4,10,10,20,20,1,1,1\n",
            encoding="utf-8",
        )

        manifest = convert_mot20_split(dataset_root, "train_half")

        self.assertEqual([image["frame_id"] for image in manifest["images"]], [1, 2, 3])
        self.assertEqual([annotation["iscrowd"] for annotation in manifest["annotations"]], [0, 1, 0])
        self.assertEqual([annotation["category_id"] for annotation in manifest["annotations"]], [1, 1, 1])
        self.assertEqual(manifest["annotations"][1]["source_class_id"], 7)
        self.assertEqual(manifest["metadata"]["conversion_revision"], "mot20-rfdetr-coco-v2")
        self.assertEqual(
            manifest["metadata"]["box_accounting"],
            {
                "clipped_positive": 1,
                "excluded": 1,
                "rejected_invalid_box": 1,
                "retained_ignored": 1,
                "retained_positive": 1,
            },
        )


class CrowdHumanCocoConversionTest(unittest.TestCase):
    def test_uses_fbox_and_preserves_bytetrack_ignore_annotations(self) -> None:
        root = Path(tempfile.mkdtemp())
        image_root = root / "Images"
        image_root.mkdir()
        Image.new("RGB", (100, 80)).save(image_root / "sample.jpg")
        annotation_path = root / "annotation_train.odgt"
        annotation_path.write_text(
            '{"ID": "sample", "gtboxes": ['
            '{"tag": "person", "fbox": [10, 10, 20, 20], "vbox": [11, 11, 10, 10], "extra": {"ignore": 0}}, '
            '{"tag": "mask", "fbox": [-5, 40, 20, 20], "vbox": [0, 40, 15, 20], "extra": {"ignore": 1}}'
            ']}\n',
            encoding="utf-8",
        )

        manifest = convert_crowdhuman_split(annotation_path, image_root, "train")

        self.assertEqual(manifest["images"][0]["file_name"], "sample.jpg")
        self.assertEqual(manifest["annotations"][0]["bbox"], [10.0, 10.0, 20.0, 20.0])
        self.assertEqual(manifest["annotations"][0]["bbox_vis"], [11, 11, 10, 10])
        self.assertEqual([annotation["iscrowd"] for annotation in manifest["annotations"]], [0, 1])
        self.assertEqual(manifest["annotations"][1]["bbox"], [0.0, 40.0, 15.0, 20.0])
        self.assertEqual(
            manifest["metadata"]["box_accounting"],
            {"clipped_ignored": 1, "retained_positive": 1},
        )


class ByteTrackMixerTest(unittest.TestCase):
    def test_merges_mot20_and_both_crowdhuman_splits_with_unique_ids(self) -> None:
        mot20 = _manifest("MOT20", "train_half", "MOT20-01/img1/000001.jpg")
        crowdhuman_train = _manifest("CrowdHuman", "train", "train.jpg")
        crowdhuman_val = _manifest("CrowdHuman", "val", "val.jpg")

        mixed = merge_bytetrack_mot20_crowdhuman(mot20, crowdhuman_train, crowdhuman_val)

        self.assertEqual(
            [image["file_name"] for image in mixed["images"]],
            ["mot20_train/MOT20-01/img1/000001.jpg", "crowdhuman_train/train.jpg", "crowdhuman_val/val.jpg"],
        )
        self.assertEqual([image["id"] for image in mixed["images"]], [1, 2, 3])
        self.assertEqual([annotation["image_id"] for annotation in mixed["annotations"]], [1, 2, 3])
        self.assertEqual(mixed["metadata"]["sources"], ["mot20_train_half", "crowdhuman_train", "crowdhuman_val"])

    def test_rejects_a_manifest_with_an_unexpected_source_or_split(self) -> None:
        mot20 = _manifest("MOT20", "val_half", "MOT20-01/img1/000001.jpg")
        crowdhuman_train = _manifest("CrowdHuman", "train", "train.jpg")
        crowdhuman_val = _manifest("CrowdHuman", "val", "val.jpg")

        with self.assertRaisesRegex(ValueError, "expected MOT20 train_half"):
            merge_bytetrack_mot20_crowdhuman(mot20, crowdhuman_train, crowdhuman_val)


class RfDetrDatasetAssemblyTest(unittest.TestCase):
    def test_assembles_an_explicit_byte65_train_root(self) -> None:
        root = Path(tempfile.mkdtemp())
        mot20_root = root / "mot20"
        crowdhuman_train_root = root / "crowdhuman_train"
        crowdhuman_val_root = root / "crowdhuman_val"
        byte65_root = root / "byte65"
        (mot20_root / "MOT20-01" / "img1").mkdir(parents=True)
        crowdhuman_train_root.mkdir()
        crowdhuman_val_root.mkdir()
        (byte65_root / "images" / "MOT20-06").mkdir(parents=True)
        Image.new("RGB", (100, 80)).save(mot20_root / "MOT20-01" / "img1" / "000001.jpg")
        Image.new("RGB", (100, 80)).save(crowdhuman_train_root / "train.jpg")
        Image.new("RGB", (100, 80)).save(crowdhuman_val_root / "val.jpg")
        Image.new("RGB", (100, 80)).save(byte65_root / "images" / "MOT20-06" / "000001.jpg")
        clean_train = merge_bytetrack_mot20_crowdhuman(
            _manifest("MOT20", "train_half", "MOT20-01/img1/000001.jpg"),
            _manifest("CrowdHuman", "train", "train.jpg"),
            _manifest("CrowdHuman", "val", "val.jpg"),
        )
        byte65 = _manifest("Byte65", "test_adapted_overlay", "images/MOT20-06/000001.jpg")
        byte65["metadata"]["human_audit"] = "exhaustive"
        dataset_root = root / "rfdetr"

        assemble_rfdetr_coco_dataset(
            dataset_root,
            merge_byte65_test_adapted_overlay(clean_train, byte65),
            _manifest("MOT20", "val_half", "MOT20-01/img1/000001.jpg"),
            mot20_root,
            crowdhuman_train_root,
            crowdhuman_val_root,
            extra_train_image_roots={"byte65": byte65_root},
        )

        self.assertTrue((dataset_root / "train" / "byte65").is_symlink())
        self.assertTrue((dataset_root / "train" / "byte65" / "images" / "MOT20-06" / "000001.jpg").is_file())

    def test_assembles_resolvable_train_and_validation_image_roots(self) -> None:
        root = Path(tempfile.mkdtemp())
        mot20_root = root / "mot20"
        crowdhuman_train_root = root / "crowdhuman_train"
        crowdhuman_val_root = root / "crowdhuman_val"
        (mot20_root / "MOT20-01" / "img1").mkdir(parents=True)
        crowdhuman_train_root.mkdir()
        crowdhuman_val_root.mkdir()
        Image.new("RGB", (100, 80)).save(mot20_root / "MOT20-01" / "img1" / "000001.jpg")
        Image.new("RGB", (100, 80)).save(crowdhuman_train_root / "train.jpg")
        Image.new("RGB", (100, 80)).save(crowdhuman_val_root / "val.jpg")
        train_manifest = merge_bytetrack_mot20_crowdhuman(
            _manifest("MOT20", "train_half", "MOT20-01/img1/000001.jpg"),
            _manifest("CrowdHuman", "train", "train.jpg"),
            _manifest("CrowdHuman", "val", "val.jpg"),
        )
        val_manifest = _manifest("MOT20", "val_half", "MOT20-01/img1/000001.jpg")
        dataset_root = root / "rfdetr"

        assemble_rfdetr_coco_dataset(
            dataset_root,
            train_manifest,
            val_manifest,
            mot20_root,
            crowdhuman_train_root,
            crowdhuman_val_root,
        )

        train_annotations = json.loads((dataset_root / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
        valid_annotations = json.loads((dataset_root / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
        self.assertTrue((dataset_root / "train" / "mot20_train").is_symlink())
        self.assertTrue((dataset_root / "train" / "crowdhuman_train").is_symlink())
        self.assertTrue((dataset_root / "train" / "crowdhuman_val").is_symlink())
        self.assertTrue((dataset_root / "valid" / "mot20_val").is_symlink())
        self.assertTrue((dataset_root / "train" / train_annotations["images"][0]["file_name"]).is_file())
        self.assertEqual(valid_annotations["images"][0]["file_name"], "mot20_val/MOT20-01/img1/000001.jpg")
        self.assertTrue((dataset_root / "valid" / valid_annotations["images"][0]["file_name"]).is_file())


class Byte65OverlayMixerTest(unittest.TestCase):
    def test_adds_human_audited_byte65_with_a_clear_test_adapted_manifest_identity(self) -> None:
        clean_train = merge_bytetrack_mot20_crowdhuman(
            _manifest("MOT20", "train_half", "MOT20-01/img1/000001.jpg"),
            _manifest("CrowdHuman", "train", "train.jpg"),
            _manifest("CrowdHuman", "val", "val.jpg"),
        )
        byte65 = _manifest("Byte65", "test_adapted_overlay", "images/MOT20-06/000001.jpg")
        byte65["metadata"]["human_audit"] = "exhaustive"

        mixed = merge_byte65_test_adapted_overlay(clean_train, byte65)

        self.assertEqual(mixed["images"][-1]["file_name"], "byte65/images/MOT20-06/000001.jpg")
        self.assertEqual(mixed["metadata"]["classification"], "local_test_adapted")
        self.assertTrue(mixed["metadata"]["includes_mot20_test_derived_labels"])
        self.assertEqual(mixed["metadata"]["sources"][-1], "byte65_human_audited")


class CocoManifestWriteTest(unittest.TestCase):
    def test_writes_deterministic_manifest_and_refuses_overwrite(self) -> None:
        destination = Path(tempfile.mkdtemp()) / "manifest.json"

        digest = write_coco_manifest(_manifest("MOT20", "train_half", "MOT20-01/img1/000001.jpg"), destination)

        self.assertEqual(len(digest), 64)
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["metadata"]["source"], "MOT20")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            write_coco_manifest(_manifest("MOT20", "train_half", "MOT20-01/img1/000001.jpg"), destination)


def _manifest(source: str, split: str, file_name: str) -> dict[str, object]:
    return {
        "images": [{"id": 1, "file_name": file_name, "width": 100, "height": 80}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "pedestrian"}],
        "metadata": {"source": source, "split": split},
    }


def _write_seqinfo(sequence_root: Path, length: int, width: int, height: int) -> None:
    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": sequence_root.name,
        "imDir": "img1",
        "seqLength": str(length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }
    with (sequence_root / "seqinfo.ini").open("w") as stream:
        config.write(stream)


if __name__ == "__main__":
    unittest.main()