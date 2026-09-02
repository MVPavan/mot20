from __future__ import annotations

import unittest

from mot20.viewer.contracts import parse_result_row
from mot20.viewer.supervision_adapter import (
    filter_detections_by_tracker_id,
    observations_to_detections,
)


class SupervisionAdapterTest(unittest.TestCase):
    def test_normalized_observations_convert_and_filter_by_tracker_id(self) -> None:
        observations = tuple(
            parse_result_row(
                row,
                source_key="fixture-result",
                sequence="MOT20-06",
                row_index=row_index,
                source_hash="source-hash",
                image_width=100,
                image_height=80,
                sequence_length=1,
            )
            for row_index, row in enumerate(
                (
                    "1,7,10,20,30,40,0.8,-1,-1,-1",
                    "1,8,20,10,10,15,0.6,-1,-1,-1",
                ),
                start=1,
            )
        )

        detections = observations_to_detections(observations)
        selected = filter_detections_by_tracker_id(detections, 8)

        self.assertEqual(detections.xyxy.tolist(), [[10.0, 20.0, 40.0, 60.0], [20.0, 10.0, 30.0, 25.0]])
        assert detections.tracker_id is not None
        self.assertEqual(detections.tracker_id.tolist(), [7, 8])
        self.assertEqual(selected.xyxy.tolist(), [[20.0, 10.0, 30.0, 25.0]])
        assert selected.tracker_id is not None
        self.assertEqual(selected.tracker_id.tolist(), [8])
        self.assertEqual(list(selected.data["row_index"]), [2])
        self.assertEqual(list(selected.data["row_hash"]), [observations[1].row_hash])


if __name__ == "__main__":
    unittest.main()