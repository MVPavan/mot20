from __future__ import annotations

import gc
import unittest
from pathlib import Path

import pytest

from mot20.viewer.config import load_config
from mot20.viewer.loaders import load_source


@pytest.mark.local_data
class LocalMot20SmokeTest(unittest.TestCase):
    def test_configured_sources_preserve_identity_contracts(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        config = load_config(repository_root / "track-viz" / "configs" / "viewer.toml")
        expected = {
            "mot20-01-gt": ("tracked", 9, 26_647),
            "mot20-06-joco": ("sentinel_only", 10, 136_267),
            "mot20-08-joco": ("sentinel_only", 10, 92_213),
        }

        self.assertEqual({source.key for source in config.sources}, set(expected))
        for source in config.sources:
            with self.subTest(source=source.key):
                loaded = load_source(source, repository_root)
                expected_status, expected_fields, expected_rows = expected[source.key]
                self.assertEqual(loaded.capability.id_status, expected_status)
                self.assertEqual(len(loaded.source_rows), expected_rows)
                self.assertTrue(all(len(row.raw_fields) == expected_fields for row in loaded.source_rows))
                self.assertTrue(all(row.source_hash == loaded.source_hash for row in loaded.source_rows))
                if expected_status == "tracked":
                    self.assertTrue(loaded.capability.track_features)
                    self.assertTrue(loaded.indexes.tracks)
                    self.assertTrue(all(row.mark == 1 and row.class_id == 1 for row in loaded.observations))
                else:
                    self.assertFalse(loaded.capability.track_features)
                    self.assertEqual(dict(loaded.indexes.tracks), {})
                    self.assertTrue(all(row.raw_track_id == -1 for row in loaded.source_rows))
                    self.assertTrue(all(row.usable_track_id is None for row in loaded.source_rows))
                    self.assertTrue(all(row.opaque_result_fields is not None for row in loaded.source_rows))
                del loaded
                gc.collect()


if __name__ == "__main__":
    unittest.main()