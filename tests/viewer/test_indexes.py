from __future__ import annotations

import unittest
from collections.abc import MutableMapping
from typing import cast

from mot20.viewer.contracts import Observation, Sequence, parse_result_row
from mot20.viewer.indexes import build_indexes


class SequenceIndexTest(unittest.TestCase):
    def test_empty_frames_and_duplicate_sentinel_rows_remain_explicit(self) -> None:
        sequence = Sequence(
            name="MOT20-06",
            length=3,
            width=100,
            height=80,
            frame_rate=25,
            image_names=("000001.jpg", "000002.jpg", "000003.jpg"),
        )
        row = "1,-1,10,20,30,40,0.8,-1,-1,-1"
        observations = tuple(
            parse_result_row(
                row,
                source_key="fixture",
                sequence=sequence.name,
                row_index=row_index,
                source_hash="source-hash",
                image_width=sequence.width,
                image_height=sequence.height,
                sequence_length=sequence.length,
            )
            for row_index in (1, 2)
        )

        indexes = build_indexes(sequence, observations)

        self.assertEqual(tuple(indexes.frames), (1, 2, 3))
        self.assertEqual([row.row_index for row in indexes.frames[1]], [1, 2])
        self.assertEqual(indexes.frames[2], ())
        self.assertEqual(indexes.frames[3], ())
        self.assertEqual(dict(indexes.tracks), {})
        with self.assertRaises(TypeError):
            mutable_frames = cast(MutableMapping[int, tuple[Observation, ...]], indexes.frames)
            mutable_frames[1] = ()

    def test_sequence_identity_isolated_even_when_track_ids_match(self) -> None:
        first_sequence = Sequence("MOT20-01", 1, 100, 80, 25, ("000001.jpg",))
        second_sequence = Sequence("MOT20-02", 1, 100, 80, 25, ("000001.jpg",))
        first_observation = parse_result_row(
            "1,4,10,20,30,40,0.8,-1,-1,-1",
            source_key="first",
            sequence=first_sequence.name,
            row_index=1,
            source_hash="first-hash",
            image_width=100,
            image_height=80,
            sequence_length=1,
        )
        second_observation = parse_result_row(
            "1,4,10,20,30,40,0.8,-1,-1,-1",
            source_key="second",
            sequence=second_sequence.name,
            row_index=1,
            source_hash="second-hash",
            image_width=100,
            image_height=80,
            sequence_length=1,
        )

        first_indexes = build_indexes(first_sequence, (first_observation,))
        second_indexes = build_indexes(second_sequence, (second_observation,))

        self.assertEqual(first_indexes.tracks[4][0].sequence, "MOT20-01")
        self.assertEqual(second_indexes.tracks[4][0].sequence, "MOT20-02")
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_indexes(first_sequence, (second_observation,))


if __name__ == "__main__":
    unittest.main()