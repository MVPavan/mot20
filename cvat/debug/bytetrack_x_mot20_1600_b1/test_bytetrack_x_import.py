from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("bytetrack_x_import.py")
SPEC = importlib.util.spec_from_file_location("bytetrack_x_import", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ByteTrackXImportTest(unittest.TestCase):
    def test_normalized_yolo_rows_become_zero_based_semi_auto_shapes(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "000001.txt").write_text("0 0.5 0.5 0.25 0.5 0.9\n")
        (root / "000002.txt").write_text("0 0.25 0.25 0.1 0.2\n")

        shapes = MODULE.read_yolo_shapes(root, sequence_length=2, image_width=1920, image_height=734, label_id=17)

        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0]["frame"], 0)
        self.assertEqual(shapes[0]["label_id"], 17)
        self.assertEqual(shapes[0]["source"], "semi-auto")
        self.assertEqual(shapes[0]["points"], [720.0, 183.5, 1200.0, 550.5])
        self.assertEqual(shapes[1]["frame"], 1)

    def test_subpixel_boundary_rounding_is_clamped_but_material_overflow_is_rejected(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "000001.txt").write_text("0 0.5 0.00000025 0.25 0.000001\n")

        shapes = MODULE.read_yolo_shapes(root, sequence_length=1, image_width=1920, image_height=734, label_id=17)

        self.assertEqual(shapes[0]["points"][1], 0.0)
        (root / "000001.txt").write_text("0 0.5 0.1 0.25 0.5\n")
        with self.assertRaisesRegex(ValueError, "outside image bounds"):
            MODULE.read_yolo_shapes(root, sequence_length=1, image_width=1920, image_height=734, label_id=17)

    def test_requested_frame_range_becomes_task_local(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "000001.txt").write_text("0 0.5 0.5 0.25 0.5\n")
        (root / "000002.txt").write_text("0 0.5 0.5 0.25 0.5\n")
        (root / "000003.txt").write_text("0 0.5 0.5 0.25 0.5\n")

        shapes = MODULE.read_yolo_shapes(
            root,
            sequence_length=3,
            image_width=1920,
            image_height=734,
            label_id=17,
            start_frame=2,
            stop_frame=3,
        )

        self.assertEqual([shape["frame"] for shape in shapes], [0, 1])


if __name__ == "__main__":
    unittest.main()
