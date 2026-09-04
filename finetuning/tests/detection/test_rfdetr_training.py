from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mot20.detection.rfdetr_training import load_training_config, validate_training_config


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


def _audit(maximum: int) -> dict[str, object]:
    return {"query_capacity": {"maximum_loss_participating_labels": maximum}}


if __name__ == "__main__":
    unittest.main()