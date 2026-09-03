from __future__ import annotations

import unittest

from mot20.viewer.contracts import parse_gt_row
from mot20.viewer.filmstrip import MAX_FILMSTRIP_SAMPLES, sample_filmstrip


def build_observations(count: int):  # type: ignore[no-untyped-def]
    return tuple(
        parse_gt_row(
            f"{frame},12,1,1,4,5,1,1,0.9",
            source_key="fixture-gt",
            sequence="MOT20-01",
            row_index=frame,
            source_hash="source-hash",
            image_width=16,
            image_height=12,
            sequence_length=count,
        )
        for frame in range(1, count + 1)
    )


class FilmstripSamplerTest(unittest.TestCase):
    def test_sampler_caps_and_keeps_current_endpoints_and_both_sides(self) -> None:
        observations = build_observations(100)

        first = sample_filmstrip(observations, current_row_index=50)
        second = sample_filmstrip(observations, current_row_index=50)
        row_indexes = tuple(observation.row_index for observation in first)

        self.assertEqual(MAX_FILMSTRIP_SAMPLES, 64)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(row_indexes[0], 1)
        self.assertEqual(row_indexes[-1], 100)
        self.assertIn(50, row_indexes)
        self.assertTrue(any(1 < row_index < 50 for row_index in row_indexes))
        self.assertTrue(any(50 < row_index < 100 for row_index in row_indexes))
        self.assertEqual(row_indexes, tuple(sorted(row_indexes)))


if __name__ == "__main__":
    unittest.main()