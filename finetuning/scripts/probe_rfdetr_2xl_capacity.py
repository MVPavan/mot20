#!/usr/bin/env python3
"""Measure the approved RF-DETR 2XL capacity envelope under real DDP work."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel

from mot20.detection.capacity_probe import pad_batch_to_envelope
from mot20.detection.checkpoint_loading import CLASSIFICATION_HEAD_MARKERS, prepare_expanded_checkpoint_state
from mot20.detection.dataset_audit import audit_rfdetr_coco_dataset
from mot20.detection.rfdetr_integration import use_ignore_aware_rfdetr
from mot20.detection.rfdetr_training import load_training_config, validate_training_config


EXPECTED_ENVELOPE = (1360, 1360)
EXPECTED_GEOMETRY = {
    "resolution": 1120,
    "multi_scale": True,
    "do_random_resize_via_padding": True,
    "square_resize_div_64": False,
    "scale_jitter": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_steps < 1:
        raise ValueError("--train-steps must be positive")
    rank, world_size, device = _initialize_distributed()
    try:
        config_path = args.config.resolve()
        dataset_root = args.dataset_root.resolve()
        config = load_training_config(config_path)
        validate_training_config(config)
        _validate_probe_contract(config, world_size)
        audit = audit_rfdetr_coco_dataset(dataset_root, group_detr=config["capacity"]["group_detr"])
        validate_training_config(config, audit)
        checkpoint = (config_path.parents[1] / config["model"]["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"RF-DETR checkpoint is not a file: {checkpoint}")
        output = args.output.resolve()
        _prepare_output_directory(output, rank)
        _seed_everything(int(config["training"]["seed"]))
        result = _run_probe(config, dataset_root, checkpoint, args.train_steps, device)
        result["provenance"] = _provenance(config_path, dataset_root, checkpoint, audit)
        result["rank"] = rank
        result["world_size"] = world_size
        _write_json(output / f"rank-{rank:02d}.json", result)
        distributed.barrier()
        if rank == 0:
            _write_json(output / "summary.json", _summarize(output, world_size))
    finally:
        distributed.destroy_process_group()


def _initialize_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("capacity probe requires CUDA")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch the capacity probe with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def _validate_probe_contract(config: dict[str, Any], world_size: int) -> None:
    training = config["training"]
    geometry = {"resolution": config["model"]["resolution"], **{key: training.get(key) for key in EXPECTED_GEOMETRY if key != "resolution"}}
    if geometry != EXPECTED_GEOMETRY:
        raise ValueError(f"capacity probe requires the audited geometry {EXPECTED_GEOMETRY}, got {geometry}")
    if training.get("devices") != world_size or training.get("strategy") != "ddp" or not training.get("sync_bn"):
        raise ValueError("capacity probe requires the configured eight-device DDP SyncBatchNorm contract")
    if training.get("batch_size") != 8 or training.get("grad_accum_steps") != 1:
        raise ValueError("capacity probe requires per-rank batch_size=8 and grad_accum_steps=1")
    if config["capacity"].get("num_queries") != 390:
        raise ValueError("capacity probe requires the audited 390-query contract")


def _prepare_output_directory(output: Path, rank: int) -> None:
    if rank == 0:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing capacity-probe output: {output}")
        output.mkdir(parents=True)
    distributed.barrier()


def _run_probe(
    config: dict[str, Any],
    dataset_root: Path,
    checkpoint: Path,
    train_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule
    from rfdetr.training.module_model import get_param_dict
    from rfdetr_plus.models.detection import RFDETR2XLargeConfig

    model_settings = config["model"]
    capacity = config["capacity"]
    training = config["training"]
    train_config = TrainConfig(
        dataset_dir=str(dataset_root),
        output_dir=".",
        dataset_file="roboflow",
        **{key: value for key, value in training.items() if key not in {"devices", "strategy", "sync_bn"}},
    )
    model_config = RFDETR2XLargeConfig(
        pretrain_weights=str(checkpoint),
        num_classes=model_settings["num_classes"],
        resolution=model_settings["resolution"],
        amp=model_settings["amp"],
        gradient_checkpointing=model_settings["gradient_checkpointing"],
        num_queries=capacity["num_queries"],
        num_select=capacity["num_select"],
        group_detr=capacity["group_detr"],
        device="cuda",
    )
    object.__setattr__(model_config, "pretrain_weights", None)
    with use_ignore_aware_rfdetr(config["run"]["ignored_iou_threshold"]):
        data_module = RFDETRDataModule(model_config, train_config)
        data_module.setup("fit")
        train_batch, train_sources = _make_dense_batch(data_module, training["batch_size"], require_ignored=True)
        valid_batch, valid_sources = _make_dense_batch(data_module, training["batch_size"], require_ignored=False, split="valid")
        train_samples, train_targets = _move_batch_to_envelope(train_batch, device)
        valid_samples, valid_targets = _move_batch_to_envelope(valid_batch, device)
        module = RFDETRModelModule(model_config, train_config).to(device)
        checkpoint_queries = _load_expanded_pretrained_weights(
            module.model,
            checkpoint,
            capacity["num_queries"],
            capacity["group_detr"],
            model_config.positional_encoding_size,
        )
        module.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module.model)
        distributed_model = DistributedDataParallel(
            module.model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
        optimizer = _build_optimizer(model_config, train_config, distributed_model.module, get_param_dict, _namespace_from_configs)
        ema_model = AveragedModel(distributed_model.module, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        train_steps_result = _train_steps(
            distributed_model,
            module.criterion,
            optimizer,
            ema_model,
            train_samples,
            train_targets,
            train_steps,
            device,
        )
        validation = _validation_step(module.postprocess, ema_model, valid_samples, valid_targets, device)
    return {
        "format": "mot20.rfdetr.capacity-probe.v1",
        "checkpoint_num_queries": checkpoint_queries,
        "envelope": list(EXPECTED_ENVELOPE),
        "train_sources": train_sources,
        "valid_sources": valid_sources,
        "train": train_steps_result,
        "validation": validation,
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_bytes_after_probe": torch.cuda.mem_get_info(device)[0],
            "total_bytes": torch.cuda.mem_get_info(device)[1],
        },
    }


def _make_dense_batch(data_module: Any, batch_size: int, require_ignored: bool, split: str = "train") -> tuple[tuple[Any, Any], list[dict[str, Any]]]:
    dataset = data_module._dataset_train if split == "train" else data_module._dataset_val
    if dataset is None or not hasattr(dataset, "coco") or not hasattr(dataset, "ids"):
        raise RuntimeError(f"RF-DETR {split} dataset was not initialized as COCO")
    annotations_by_image = dataset.coco.imgToAnns
    image_ids = dataset.ids
    positive_count = lambda image_id: sum(not int(annotation.get("iscrowd", 0)) for annotation in annotations_by_image[image_id])
    ignored_count = lambda image_id: sum(int(annotation.get("iscrowd", 0)) for annotation in annotations_by_image[image_id])
    dense_image_id = max(image_ids, key=positive_count)
    ignored_image_id = max(image_ids, key=ignored_count)
    if require_ignored and not ignored_count(ignored_image_id):
        raise RuntimeError("capacity probe requires an ignored-region training image")
    selected_image_ids = [dense_image_id] * batch_size
    if require_ignored and not ignored_count(dense_image_id):
        selected_image_ids[-1] = ignored_image_id
    indices = {image_id: index for index, image_id in enumerate(image_ids)}
    batch = data_module._collate_fn([dataset[indices[image_id]] for image_id in selected_image_ids])
    sources = [
        {
            "image_id": image_id,
            "file_name": dataset.coco.imgs[image_id]["file_name"],
            "positive_boxes": positive_count(image_id),
            "ignored_boxes": ignored_count(image_id),
        }
        for image_id in selected_image_ids
    ]
    return batch, sources


def _move_batch_to_envelope(batch: tuple[Any, Any], device: torch.device) -> tuple[Any, list[dict[str, Any]]]:
    samples, targets = batch
    padded = pad_batch_to_envelope(samples, *EXPECTED_ENVELOPE).to(device)
    moved_targets = [
        {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
        for target in targets
    ]
    return padded, moved_targets


def _build_optimizer(model_config: Any, train_config: Any, model: torch.nn.Module, get_param_dict: Any, namespace_factory: Any) -> torch.optim.Optimizer:
    parameter_groups = [
        parameter_group
        for parameter_group in get_param_dict(namespace_factory(model_config, train_config), model)
        if parameter_group["params"].requires_grad
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        fused=torch.cuda.is_bf16_supported(),
        **train_config.optimizer_kwargs,
    )


def _train_steps(
    model: DistributedDataParallel,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model: AveragedModel,
    samples: Any,
    targets: list[dict[str, Any]],
    step_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.train()
    results: list[dict[str, Any]] = []
    for step in range(step_count):
        torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(samples, targets)
            losses = criterion(outputs, targets)
            loss = sum(losses[key] * criterion.weight_dict[key] for key in losses if key in criterion.weight_dict)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite capacity-probe loss on step {step}: {loss.item()}")
        loss.backward()
        gradients_finite = all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        if not gradients_finite:
            raise RuntimeError(f"non-finite capacity-probe gradients on step {step}")
        optimizer.step()
        ema_model.update_parameters(model.module)
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        results.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradients_finite": gradients_finite,
                "wall_seconds": time.perf_counter() - started_at,
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "input_shape": list(samples.tensors.shape),
                "positive_boxes": sum(int(target["boxes"].shape[0]) for target in targets),
                "ignored_boxes": sum(int(target["ignored_boxes"].shape[0]) for target in targets),
            }
        )
    return results


def _validation_step(postprocess: Any, model: AveragedModel, samples: Any, targets: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(samples)
        results = postprocess(outputs, torch.stack([target["orig_size"] for target in targets]))
    torch.cuda.synchronize(device)
    expected_detections = outputs["pred_logits"].shape[1]
    detection_counts = [int(result["scores"].shape[0]) for result in results]
    if len(results) != len(targets) or any(count != expected_detections for count in detection_counts):
        raise RuntimeError("capacity-probe validation did not retain the configured top-query output")
    return {
        "wall_seconds": time.perf_counter() - started_at,
        "input_shape": list(samples.tensors.shape),
        "detections_per_image": detection_counts,
        "expected_detections": expected_detections,
    }


def _load_expanded_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    target_queries: int,
    group_detr: int,
    positional_encoding_size: int,
) -> int:
    from rfdetr.models.weights import interpolate_position_embeddings

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_state = checkpoint.get("model")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("trusted RF-DETR checkpoint has no model state dictionary")
    rows = checkpoint_state.get("query_feat.weight")
    if not isinstance(rows, torch.Tensor) or rows.shape[0] % group_detr:
        raise ValueError("trusted RF-DETR checkpoint has an invalid query table")
    checkpoint_queries = rows.shape[0] // group_detr
    interpolate_position_embeddings(checkpoint_state, positional_encoding_size)
    prepared = prepare_expanded_checkpoint_state(
        checkpoint_state,
        model.state_dict(),
        checkpoint_queries,
        target_queries,
        group_detr,
    )
    incompatible = model.load_state_dict(prepared, strict=False)
    missing = [
        name
        for name in incompatible.missing_keys
        if name != "_kp_active_mask" and not any(marker in name for marker in CLASSIFICATION_HEAD_MARKERS)
    ]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"unapproved checkpoint load result: missing={missing}, unexpected={incompatible.unexpected_keys}")
    return checkpoint_queries


def _provenance(config_path: Path, dataset_root: Path, checkpoint: Path, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "dataset_root": str(dataset_root),
        "train_manifest_sha256": _sha256_file(dataset_root / "train" / "_annotations.coco.json"),
        "valid_manifest_sha256": _sha256_file(dataset_root / "valid" / "_annotations.coco.json"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "rfdetr_version": importlib.metadata.version("rfdetr"),
        "rfdetr_plus_version": importlib.metadata.version("rfdetr-plus"),
        "torch_version": torch.__version__,
        "audit": audit,
    }


def _summarize(output: Path, world_size: int) -> dict[str, Any]:
    results = [json.loads((output / f"rank-{rank:02d}.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    peaks = [result["memory"]["peak_reserved_bytes"] for result in results]
    free = [result["memory"]["free_bytes_after_probe"] for result in results]
    return {
        "format": "mot20.rfdetr.capacity-probe-summary.v1",
        "world_size": world_size,
        "all_train_losses_finite": all(step["gradients_finite"] for result in results for step in result["train"]),
        "max_peak_reserved_bytes": max(peaks),
        "min_free_bytes_after_probe": min(free),
        "rank_files": [f"rank-{rank:02d}.json" for rank in range(world_size)],
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()