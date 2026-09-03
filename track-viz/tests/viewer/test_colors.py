from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mot20.viewer.api import create_app
from mot20.viewer.colors import COLOR_GOLDEN_VECTORS, color_router, track_color
from mot20.viewer.config import ViewerConfig
from mot20.viewer.loaders import load_registry


class TrackColorTest(unittest.TestCase):
    def test_color_contract_has_stable_language_neutral_golden_vectors(self) -> None:
        expected = (
            ("MOT20-01", 1, 131, (58, 230, 90), "#3ae65a"),
            ("MOT20-01", 8, 48, (230, 196, 58), "#e6c43a"),
            ("MOT20-06", 8, 329, (230, 58, 147), "#e63a93"),
        )

        self.assertEqual(
            tuple(
                (vector.sequence, vector.track_id, vector.hue, vector.rgb, vector.hex)
                for vector in COLOR_GOLDEN_VECTORS
            ),
            expected,
        )
        self.assertEqual(track_color("MOT20-01", 8), COLOR_GOLDEN_VECTORS[1])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            client = TestClient(
                create_app(
                    registry=load_registry(ViewerConfig(()), root),
                    repository_root=root,
                    extension_routers=(color_router,),
                )
            )
            response = client.get("/api/contracts/track-colors")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["version"], "fnv1a32-hsv-integer-v1")
        self.assertEqual(payload["key_encoding"], "UTF-8(sequence + U+001F + decimal track ID)")
        self.assertEqual(payload["vectors"][1]["hex"], "#e6c43a")


if __name__ == "__main__":
    unittest.main()