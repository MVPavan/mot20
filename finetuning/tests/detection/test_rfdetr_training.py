from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mot20.detection.rfdetr_training import (
    DEFAULT_LONG_SIDE_CAP,
    apply_long_side_cap,
    load_training_config,
    validate_training_config,
)


class RfDetrTrainingConfigTest(unittest.TestCase):
    def test_rejects_a_config_that_is_not_approved_for_training(self) -> None:
        config = _write_config(status="blocked_pending_real_capacity_probe", num_queries=52, num_select=52, eval_max_dets=52)

        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_training_config(load_training_config(config), _audit(maximum=51))

    def test_rejects_query_capacity_that_cannot_cover_observed_labels(self) -> None:
        config = _write_config(status="approved", num_queries=52, num_select=52, eval_max_dets=52)

        with self.assertRaisesRegex(ValueError, "greater than the observed maximum"):
            validate_training_config(load_training_config(config), _audit(maximum=52))

    def test_accepts_an_approved_capacity_aligned_to_group_detr(self) -> None:
        config = _write_config(status="approved", num_queries=65, num_select=65, eval_max_dets=65)

        loaded = load_training_config(config)
        validate_training_config(loaded, _audit(maximum=52))

        self.assertEqual(loaded["model"]["name"], "RFDETR2XLarge")

    def test_accepts_an_approved_config_before_its_dataset_audit(self) -> None:
        config = _write_config(status="approved", num_queries=65, num_select=65, eval_max_dets=65)

        validate_training_config(load_training_config(config))

    def test_accepts_rfdetr_auto_batch_mode(self) -> None:
        config = _write_config(status="approved", num_queries=65, num_select=65, eval_max_dets=65)
        config.write_text(config.read_text(encoding="utf-8").replace("batch_size = 1", 'batch_size = "auto"'), encoding="utf-8")

        validate_training_config(load_training_config(config), _audit(maximum=52))

    def test_rejects_a_run_classification_that_differs_from_the_dataset_audit(self) -> None:
        config = _write_config(status="approved", num_queries=65, num_select=65, eval_max_dets=65)

        with self.assertRaisesRegex(ValueError, "classification does not match"):
            validate_training_config(
                load_training_config(config),
                {**_audit(maximum=52), "classification": "local_test_adapted"},
            )


def _write_config(status: str, num_queries: int, num_select: int, eval_max_dets: int) -> Path:
    path = Path(tempfile.mkdtemp()) / "training.toml"
    path.write_text(
        f"""[run]
status = \"{status}\"
classification = "clean_held_out_validation"

[model]
name = \"RFDETR2XLarge\"
num_classes = 1
resolution = 880

[capacity]
group_detr = 13
num_queries = {num_queries}
num_select = {num_select}
eval_max_dets = {eval_max_dets}

[training]
batch_size = 1
grad_accum_steps = 1
epochs = 1
amp_dtype = \"bf16\"
""",
        encoding="utf-8",
    )
    return path


class LongSideCapTest(unittest.TestCase):
    """The cap decides the effective input geometry, so training and detection
    export must agree on it. Exporting a checkpoint at the wrong cap silently
    produces detections from geometry the model never saw."""

    def setUp(self) -> None:
        try:
            from rfdetr.datasets import coco as rfdetr_coco
        except ImportError:  # pragma: no cover - only the training env has rfdetr
            self.skipTest("rfdetr is not installed in this environment")
        self.coco = rfdetr_coco
        self.original = rfdetr_coco._COCO_MAX_SIZE
        self.addCleanup(setattr, rfdetr_coco, "_COCO_MAX_SIZE", self.original)

    def test_omitting_max_size_reports_the_library_default_without_mutating_it(self) -> None:
        self.assertEqual(apply_long_side_cap(None), DEFAULT_LONG_SIDE_CAP)
        self.assertEqual(self.coco._COCO_MAX_SIZE, self.original)

    def test_the_recorded_default_still_matches_the_installed_library(self) -> None:
        self.assertEqual(self.original, DEFAULT_LONG_SIDE_CAP)

    def test_raising_the_cap_updates_the_module_global_the_transform_reads(self) -> None:
        self.assertEqual(apply_long_side_cap(1600), 1600)
        self.assertEqual(self.coco._COCO_MAX_SIZE, 1600)

    def test_rejects_caps_that_are_not_positive_multiples_of_the_window_stride(self) -> None:
        # True is an int subclass, and a bare isinstance check would accept it.
        for bad in (0, -40, 1601, 1.0, True, "1600"):
            with self.subTest(max_size=bad), self.assertRaises(ValueError):
                apply_long_side_cap(bad)
            self.assertEqual(self.coco._COCO_MAX_SIZE, self.original)


def _audit(maximum: int) -> dict[str, object]:
    return {"query_capacity": {"maximum_loss_participating_labels": maximum}}


if __name__ == "__main__":
    unittest.main()