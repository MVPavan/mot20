from __future__ import annotations

import importlib.util
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "train_rfdetr_2xl.py"
_SPEC = importlib.util.spec_from_file_location("train_rfdetr_2xl", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCHER)


class RfDetrLauncherTest(unittest.TestCase):
    def test_seed_reproduces_launcher_owned_initialization(self) -> None:
        with patch.object(_LAUNCHER.torch.cuda, "is_available", return_value=False):
            _LAUNCHER._seed_everything(42)
            first = (random.random(), _LAUNCHER.torch.rand(1).item())
            _LAUNCHER._seed_everything(42)
            second = (random.random(), _LAUNCHER.torch.rand(1).item())

        self.assertEqual(first, second)

    def test_ddp_child_reuses_parent_created_initialization_artifacts(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        initialization = run_dir / "rfdetr-2xl-q390-initialization.pth"
        initialization.write_bytes(b"trusted test checkpoint")
        saved_provenance = {"expanded_checkpoint_path": str(initialization), "classification": "local_test_adapted"}
        (run_dir / "run-provenance.json").write_text(json.dumps(saved_provenance), encoding="utf-8")

        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            result_dir, result_checkpoint, result_provenance = _LAUNCHER._prepare_run_artifacts(
                {},
                Path("unused-checkpoint.pth"),
                run_dir,
                {"classification": "clean_held_out_validation"},
            )

        self.assertEqual(result_dir, run_dir)
        self.assertEqual(result_checkpoint, initialization)
        self.assertEqual(result_provenance, saved_provenance)

    def test_prepare_run_is_rejected_inside_an_external_ddp_rank(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            with self.assertRaisesRegex(ValueError, "before external DDP ranks"):
                _LAUNCHER._validate_prepare_run_parent()

    def test_multi_gpu_ddp_requires_an_external_rank_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "then launch this script with torchrun"):
                _LAUNCHER._validate_external_ddp_launch({"strategy": "ddp", "devices": 8})

    def test_external_ddp_rank_is_accepted(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            _LAUNCHER._validate_external_ddp_launch({"strategy": "ddp", "devices": 8})


if __name__ == "__main__":
    unittest.main()