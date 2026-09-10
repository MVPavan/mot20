from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mot20.detection.bytetrack_handoff import byte_tracker_inputs, write_raw_detector_export


class ByteTrackHandoffTest(unittest.TestCase):
    def test_clamps_original_pixel_detections_and_preserves_identity_scale(self) -> None:
        detections, img_info, img_size = byte_tracker_inputs(
            boxes=np.array([[-2, 3, 120, 80], [30, 30, 30, 31]], dtype=np.float32),
            scores=np.array([0.9, 0.4], dtype=np.float32),
            class_ids=np.array([0, 0], dtype=np.int64),
            width=100,
            height=70,
        )

        np.testing.assert_array_equal(detections, np.array([[0, 3, 100, 70, 0.9]], dtype=np.float32))
        self.assertEqual(img_info, (70, 100))
        self.assertEqual(img_size, (70, 100))

    def test_rejects_non_pedestrian_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "class 0"):
            byte_tracker_inputs(
                np.array([[1, 1, 2, 2]], dtype=np.float32),
                np.array([0.5], dtype=np.float32),
                np.array([1]),
                10,
                10,
            )

    def test_writes_an_immutable_export_including_empty_frames(self) -> None:
        destination = Path(tempfile.mkdtemp()) / "raw.json"
        write_raw_detector_export(
            [
                {"sequence": "MOT20-01", "frame_id": 1, "width": 100, "height": 70, "detections": [[1, 2, 3, 4, 0.9]]},
                {"sequence": "MOT20-01", "frame_id": 2, "width": 100, "height": 70, "detections": []},
            ],
            destination,
            config_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
        )

        export = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(export["frames"][1]["detections"], [])
        self.assertEqual(export["checkpoint_sha256"], "b" * 64)
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            write_raw_detector_export([], destination, "a" * 64, "b" * 64)


if __name__ == "__main__":
    unittest.main()