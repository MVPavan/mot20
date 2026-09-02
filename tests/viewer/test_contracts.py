from __future__ import annotations

import hashlib
import unittest
from dataclasses import FrozenInstanceError

from mot20.viewer.contracts import ContractError, parse_gt_row, parse_result_row


class GroundTruthContractTest(unittest.TestCase):
    def test_partial_box_preserves_raw_evidence_and_clamps_display_box(self) -> None:
        observation = parse_gt_row(
            "1,7,-5,8,20,30,1,1,0.75",
            source_key="mot20-01-gt",
            sequence="MOT20-01",
            row_index=1,
            source_hash="source-hash",
            image_width=100,
            image_height=80,
            sequence_length=2,
        )

        self.assertEqual(observation.raw_row, "1,7,-5,8,20,30,1,1,0.75")
        self.assertEqual(observation.row_hash, hashlib.sha256(observation.raw_row.encode("utf-8")).hexdigest())
        self.assertEqual(observation.raw_fields, ("1", "7", "-5", "8", "20", "30", "1", "1", "0.75"))
        self.assertEqual(observation.raw_xywh, (-5.0, 8.0, 20.0, 30.0))
        self.assertEqual(observation.display_box, (0.0, 8.0, 15.0, 38.0))
        self.assertEqual(observation.usable_track_id, 7)
        self.assertTrue(observation.reviewable)
        with self.assertRaises(FrozenInstanceError):
            observation.__setattr__("frame", 2)

    def test_invalid_rows_are_rejected_strictly(self) -> None:
        invalid_rows = (
            "1,7,0,0,10,10,1,1",
            "1,7,0,0,nan,10,1,1,0.5",
            "1,7,0,0,0,10,1,1,0.5",
            "3,7,0,0,10,10,1,1,0.5",
            "1,7,100,0,10,10,1,1,0.5",
        )
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ContractError):
                parse_gt_row(
                    row,
                    source_key="fixture",
                    sequence="MOT20-01",
                    row_index=1,
                    source_hash="source-hash",
                    image_width=100,
                    image_height=80,
                    sequence_length=2,
                )

    def test_duplicate_rows_keep_distinct_source_row_identity(self) -> None:
        raw_row = "1,7,0,0,10,10,1,1,0.75"
        observations = tuple(
            parse_gt_row(
                raw_row,
                source_key="fixture",
                sequence="MOT20-01",
                row_index=row_index,
                source_hash="source-hash",
                image_width=100,
                image_height=80,
                sequence_length=2,
            )
            for row_index in (1, 2)
        )

        self.assertEqual(observations[0].row_hash, observations[1].row_hash)
        self.assertEqual(tuple(observation.row_index for observation in observations), (1, 2))


class TrackerResultContractTest(unittest.TestCase):
    def test_sentinel_identity_and_opaque_fields_are_preserved(self) -> None:
        observation = parse_result_row(
            "2,-1,10,20,30,40,0.82,-1,-1,-1",
            source_key="mot20-06-joco",
            sequence="MOT20-06",
            row_index=3,
            source_hash="source-hash",
            image_width=100,
            image_height=80,
            sequence_length=2,
        )

        self.assertEqual(observation.raw_track_id, -1)
        self.assertIsNone(observation.usable_track_id)
        self.assertEqual(observation.score, 0.82)
        self.assertEqual(observation.opaque_result_fields, (-1.0, -1.0, -1.0))
        self.assertTrue(observation.reviewable)

    def test_tracker_rows_require_exactly_ten_fields(self) -> None:
        with self.assertRaisesRegex(ContractError, "exactly 10 fields"):
            parse_result_row(
                "2,-1,10,20,30,40,0.82,-1,-1",
                source_key="mot20-06-joco",
                sequence="MOT20-06",
                row_index=3,
                source_hash="source-hash",
                image_width=100,
                image_height=80,
                sequence_length=2,
            )


if __name__ == "__main__":
    unittest.main()