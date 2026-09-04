#!/usr/bin/env python3
"""Validate and launch an approved, immutable RF-DETR 2XL detector run."""

from __future__ import annotations

import argparse
import hashlib
import json
import gc
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from mot20.detection.checkpoint_loading import expand_query_parameters
from mot20.detection.coco_conversion import write_coco_manifest
from mot20.detection.dataset_audit import audit_rfdetr_coco_dataset
from mot20.detection.rfdetr_integration import use_ignore_aware_rfdetr
from mot20.detection.rfdetr_training import load_training_config, validate_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_training_config(config_path)
    validate_training_config(config)
    dataset_root = args.dataset_root.resolve()
    audit = audit_rfdetr_coco_dataset(dataset_root, group_detr=config["capacity"]["group_detr"])
    validate_training_config(config, audit)
    seed = config["training"].get("seed")
    if seed is not None:
        _seed_everything(seed)
    checkpoint = (config_path.parents[1] / config["model"]["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"RF-DETR checkpoint is not a file: {checkpoint}")
    provenance = _provenance(config_path, dataset_root, checkpoint, audit)
    if args.dry_run:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return
    if args.prepare_run:
        _validate_prepare_run_parent()
        _, expanded_checkpoint, prepared_provenance = _prepare_run_artifacts(
            config,
            checkpoint,
            args.run_dir.resolve(),
            provenance,
        )
        print(json.dumps({**prepared_provenance, "expanded_checkpoint_path": str(expanded_checkpoint)}, indent=2, sort_keys=True))
        return
    _validate_external_ddp_launch(config["training"])
    run_dir, expanded_checkpoint, provenance = _prepare_run_artifacts(config, checkpoint, args.run_dir.resolve(), provenance)

    from rfdetr import RFDETR2XLarge

    model_config = config["model"]
    capacity = config["capacity"]
    training = config["training"]
    model = RFDETR2XLarge(
        pretrain_weights=str(expanded_checkpoint),
        num_classes=model_config["num_classes"],
        resolution=model_config["resolution"],
        amp=model_config["amp"],
        gradient_checkpointing=model_config["gradient_checkpointing"],
        num_queries=capacity["num_queries"],
        num_select=capacity["num_select"],
        group_detr=capacity["group_detr"],
        device="cuda",
    )
    train_started_at = time.perf_counter()
    with use_ignore_aware_rfdetr(config["run"]["ignored_iou_threshold"]):
        model.train(
            dataset_dir=str(dataset_root),
            output_dir=str(run_dir),
            dataset_file="roboflow",
            device="cuda",
            resolution=model_config["resolution"],
            class_names=["pedestrian"],
            eval_max_dets=capacity["eval_max_dets"],
            notes=provenance,
            **training,
        )
    if os.environ.get("LOCAL_RANK", "0") == "0":
        write_coco_manifest(
            {
                "format": "mot20.rfdetr.launcher-result.v1",
                "classification": config["run"]["classification"],
                "train_call_wall_seconds": time.perf_counter() - train_started_at,
            },
            run_dir / "launcher-result.json",
        )


def _provenance(config_path: Path, dataset_root: Path, checkpoint: Path, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "mot20.rfdetr.run-provenance.v1",
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "dataset_root": str(dataset_root),
        "train_manifest_sha256": _sha256_file(dataset_root / "train" / "_annotations.coco.json"),
        "valid_manifest_sha256": _sha256_file(dataset_root / "valid" / "_annotations.coco.json"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "audit": audit,
    }


def _materialize_expanded_checkpoint(config: dict[str, Any], checkpoint: Path, run_dir: Path) -> Path:
    from rfdetr.config import TrainConfig
    from rfdetr.training import RFDETRModelModule
    from rfdetr_plus.models.detection import RFDETR2XLargeConfig

    capacity = config["capacity"]
    target_queries = capacity["num_queries"]
    output_path = run_dir / "rfdetr-2xl-q390-initialization.pth"
    model_config = RFDETR2XLargeConfig(
        pretrain_weights=str(checkpoint),
        num_classes=1,
        resolution=config["model"]["resolution"],
        num_queries=target_queries,
        num_select=capacity["num_select"],
        group_detr=capacity["group_detr"],
        device="cuda",
    )
    object.__setattr__(model_config, "pretrain_weights", None)
    seed_train_config = TrainConfig(dataset_dir=".", output_dir=str(run_dir), dataset_file="roboflow")
    initialized_state = RFDETRModelModule(model_config, seed_train_config).model.state_dict()
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_state = checkpoint_data.get("model")
    rows = checkpoint_state.get("query_feat.weight") if isinstance(checkpoint_state, dict) else None
    if not isinstance(rows, torch.Tensor) or rows.shape[0] % capacity["group_detr"]:
        raise ValueError("trusted RF-DETR checkpoint has an invalid query table")
    checkpoint_queries = rows.shape[0] // capacity["group_detr"]
    expanded_queries = expand_query_parameters(
        checkpoint_state,
        initialized_state,
        checkpoint_queries,
        target_queries,
        capacity["group_detr"],
    )
    torch.save({**checkpoint_data, "model": {**checkpoint_state, **expanded_queries}}, output_path)
    del initialized_state, checkpoint_data, checkpoint_state
    gc.collect()
    return output_path


def _prepare_run_artifacts(
    config: dict[str, Any],
    checkpoint: Path,
    run_dir: Path,
    provenance: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    run_dir = Path(run_dir)
    expanded_checkpoint = run_dir / "rfdetr-2xl-q390-initialization.pth"
    provenance_path = run_dir / "run-provenance.json"
    if "LOCAL_RANK" in os.environ:
        if not expanded_checkpoint.is_file() or not provenance_path.is_file():
            raise ValueError("DDP child cannot find parent-created run initialization artifacts")
        with provenance_path.open(encoding="utf-8") as stream:
            saved_provenance = json.load(stream)
        if not isinstance(saved_provenance, dict):
            raise ValueError("DDP child found an invalid run provenance record")
        return run_dir, expanded_checkpoint, saved_provenance
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing RF-DETR run: {run_dir}")
    run_dir.mkdir(parents=True)
    expanded_checkpoint = _materialize_expanded_checkpoint(config, checkpoint, run_dir)
    provenance["expanded_checkpoint_path"] = str(expanded_checkpoint)
    provenance["expanded_checkpoint_sha256"] = _sha256_file(expanded_checkpoint)
    write_coco_manifest(provenance, provenance_path)
    return run_dir, expanded_checkpoint, provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    """Seed launcher-owned initialization before expanding the query checkpoint."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_prepare_run_parent() -> None:
    if "LOCAL_RANK" in os.environ:
        raise ValueError("--prepare-run must execute once before external DDP ranks start")


def _validate_external_ddp_launch(training: dict[str, Any]) -> None:
    if training.get("strategy") == "ddp" and training.get("devices", 1) > 1 and "LOCAL_RANK" not in os.environ:
        raise ValueError(
            "multi-GPU DDP runs must first use --prepare-run, then launch this script with torchrun"
        )


if __name__ == "__main__":
    main()