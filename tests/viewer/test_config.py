from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from mot20.viewer.config import (
    ConfigError,
    Provenance,
    SourceConfig,
    config_from_paths,
    load_config,
    provenance_diagnostics,
    resolve_source_paths,
)


class ViewerConfigTest(unittest.TestCase):
    def test_direct_paths_infer_sequence_and_annotation_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sequence_root = Path(temporary_directory) / "MOT20-01"
            images = sequence_root / "img1"
            images.mkdir(parents=True)
            (sequence_root / "seqinfo.ini").write_text(
                "[Sequence]\nname=MOT20-01\n",
                encoding="utf-8",
            )
            ground_truth = sequence_root / "gt.txt"
            ground_truth.write_text("1,7,0,0,3,4,1,1,0.8\n", encoding="utf-8")
            predictions = sequence_root / "predictions.txt"
            predictions.write_text("1,7,0,0,3,4,0.9,-1,-1,-1\n", encoding="utf-8")

            ground_truth_config = config_from_paths(images, ground_truth)
            prediction_config = config_from_paths(images, predictions)

        self.assertEqual(ground_truth_config.sources[0].sequence, "MOT20-01")
        self.assertEqual(ground_truth_config.sources[0].adapter, "mot_gt_9")
        self.assertEqual(prediction_config.sources[0].adapter, "mot_result_10")
        self.assertEqual(Path(prediction_config.sources[0].images), images.resolve())
        self.assertEqual(Path(prediction_config.sources[0].annotations), predictions.resolve())

    def test_valid_config_is_immutable_and_diagnoses_missing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "viewer.toml"
            config_path.write_text(
                """
[[sources]]
key = "fixture"
sequence = "MOT20-01"
seqinfo = "data/MOT20-01/seqinfo.ini"
images = "data/MOT20-01/img1"
annotations = "data/MOT20-01/gt.txt"
adapter = "mot_gt_9"
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.sources[0].key, "fixture")
        diagnostics = provenance_diagnostics(config)
        self.assertEqual(diagnostics[0].code, "missing_provenance")
        self.assertIn("producer", diagnostics[0].fields)
        with self.assertRaises(FrozenInstanceError):
            config.__setattr__("sources", ())

    def test_complete_optional_provenance_is_preserved_without_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "viewer.toml"
            config_path.write_text(
                """
[[sources]]
key = "fixture"
sequence = "MOT20-01"
seqinfo = "data/MOT20-01/seqinfo.ini"
images = "data/MOT20-01/img1"
annotations = "data/MOT20-01/gt.txt"
adapter = "mot_gt_9"

[sources.provenance]
producer = "fixture producer"
detector = "fixture detector"
checkpoint = "fixture checkpoint"
tracker = "fixture tracker"
post_processing = "none"
adaptation_iterations = 3
notes = "synthetic only"
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.sources[0].provenance.adaptation_iterations, 3)
        self.assertEqual(provenance_diagnostics(config), ())

    def test_traversal_and_symlink_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            root = workspace / "repository"
            root.mkdir()
            outside = workspace / "outside"
            outside.mkdir()
            (root / "escape").symlink_to(outside, target_is_directory=True)

            for annotations in ("../outside/rows.txt", "escape/rows.txt"):
                source = SourceConfig(
                    key="fixture",
                    sequence="MOT20-01",
                    seqinfo="data/seqinfo.ini",
                    images="data/img1",
                    annotations=annotations,
                    adapter="mot_gt_9",
                    provenance=Provenance(),
                )
                with self.subTest(annotations=annotations), self.assertRaisesRegex(
                    ConfigError, "escapes repository root"
                ):
                    resolve_source_paths(source, root)


if __name__ == "__main__":
    unittest.main()