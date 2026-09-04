#!/usr/bin/env python3
"""Verify RF-DETR loader geometry and target preservation without constructing a model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, SequentialSampler

from mot20.detection.dataset_audit import audit_rfdetr_coco_dataset
from mot20.detection.rfdetr_integration import use_ignore_aware_rfdetr
from mot20.detection.rfdetr_training import load_training_config, validate_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite loader verification output: {args.output}")
    config = load_training_config(args.config.resolve())
    dataset_root = args.dataset_root.resolve()
    audit = audit_rfdetr_coco_dataset(dataset_root, group_detr=config["capacity"]["group_detr"])
    validate_training_config(config, audit)
    result = _verify_loader(config, dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def _verify_loader(config: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    from rfdetr.config import TrainConfig
    from rfdetr.training import RFDETRDataModule
    from rfdetr_plus.models.detection import RFDETR2XLargeConfig

    model = config["model"]
    capacity = config["capacity"]
    training = config["training"]
    seed = training.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
    model_config = RFDETR2XLargeConfig(
        pretrain_weights=model["checkpoint"],
        num_classes=model["num_classes"],
        resolution=model["resolution"],
        num_queries=capacity["num_queries"],
        num_select=capacity["num_select"],
        group_detr=capacity["group_detr"],
        amp=model["amp"],
        gradient_checkpointing=model["gradient_checkpointing"],
        device="cpu",
    )
    train_config = TrainConfig(
        dataset_dir=str(dataset_root),
        output_dir=str(dataset_root),
        dataset_file="roboflow",
        **training,
    )
    block_size = model_config.patch_size * model_config.num_windows
    if training.get("square_resize_div_64", True):
        raise ValueError("loader geometry verification requires square_resize_div_64=false")
    if training.get("scale_jitter", True):
        raise ValueError("loader geometry verification requires scale_jitter=false to verify every image without crops")

    with use_ignore_aware_rfdetr(config["run"]["ignored_iou_threshold"]):
        data_module = RFDETRDataModule(model_config, train_config)
        data_module.setup("fit")
        train_dataset = _require_dataset(data_module._dataset_train, "train")
        valid_dataset = _require_dataset(data_module._dataset_val, "valid")
        return {
            "format": "mot20.rfdetr.loader-geometry.v1",
            "model_constructed": False,
            "classification": config["run"]["classification"],
            "resolution": model["resolution"],
            "multi_scale": training["multi_scale"],
            "square_resize_div_64": training["square_resize_div_64"],
            "scale_jitter": training["scale_jitter"],
            "long_side_cap": 1333,
            "collator_block_size": block_size,
            "train": _verify_split(
                train_dataset,
                data_module._collate_fn,
                training["batch_size"],
                block_size,
                _scales(model["resolution"], training["multi_scale"], model_config, train_config),
            ),
            "valid": _verify_split(
                valid_dataset,
                data_module._collate_fn,
                training["batch_size"],
                block_size,
                [model["resolution"]],
            ),
        }


def _require_dataset(dataset: Any, split: str) -> Any:
    if dataset is None:
        raise RuntimeError(f"RF-DETR did not construct the {split} dataset")
    return dataset


def _scales(resolution: int, multi_scale: bool, model_config: Any, train_config: Any) -> list[int]:
    if not multi_scale:
        return [resolution]
    from rfdetr.datasets.coco import compute_multi_scale_scales

    return compute_multi_scale_scales(
        resolution,
        train_config.expanded_scales,
        model_config.patch_size,
        model_config.num_windows,
    )


def _verify_split(dataset: Any, collate_fn: Any, batch_size: int, block_size: int, scales: list[int]) -> dict[str, Any]:
    from rfdetr.datasets._torchvision import RandomResize

    transformed_sizes: Counter[tuple[int, int]] = Counter()
    unique_selected_scales: Counter[int] = Counter()
    cap_ambiguous_images = 0
    max_long_side = 0
    max_positive_boxes = 0
    max_ignored_boxes = 0
    for index in range(len(dataset)):
        _, target = dataset[index]
        _verify_target(target)
        height, width = (int(value) for value in target["size"])
        original_height, original_width = (int(value) for value in target["orig_size"])
        expected_scales = [
            scale
            for scale in scales
            if RandomResize._get_size(original_height, original_width, scale, 1333) == (height, width)
        ]
        if not expected_scales:
            raise ValueError(
                f"transformed image {index} has unexpected size {(width, height)} for original "
                f"{(original_width, original_height)} and scales {scales}"
            )
        if len(expected_scales) == 1:
            unique_selected_scales.update(expected_scales)
        else:
            cap_ambiguous_images += 1
        image_id = int(target["image_id"].item())
        annotations = dataset.coco.imgToAnns[image_id]
        expected_positive = sum(int(annotation.get("iscrowd", 0)) == 0 for annotation in annotations)
        expected_ignored = sum(int(annotation.get("iscrowd", 0)) == 1 for annotation in annotations)
        if target["boxes"].shape[0] != expected_positive or target["ignored_boxes"].shape[0] != expected_ignored:
            raise ValueError(f"transformed image {index} did not preserve ordinary and ignored target counts")
        transformed_sizes[(width, height)] += 1
        max_long_side = max(max_long_side, width, height)
        max_positive_boxes = max(max_positive_boxes, int(target["boxes"].shape[0]))
        max_ignored_boxes = max(max_ignored_boxes, int(target["ignored_boxes"].shape[0]))

    batch_count = 0
    max_padded_width = 0
    max_padded_height = 0
    loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), collate_fn=collate_fn)
    for samples, targets in loader:
        _, _, padded_height, padded_width = samples.tensors.shape
        if padded_height % block_size or padded_width % block_size:
            raise ValueError(f"collated shape is not aligned to {block_size}: {(padded_width, padded_height)}")
        expected_padded_height = _round_up(max(int(target["size"][0]) for target in targets), block_size)
        expected_padded_width = _round_up(max(int(target["size"][1]) for target in targets), block_size)
        if (padded_height, padded_width) != (expected_padded_height, expected_padded_width):
            raise ValueError(
                f"collator used unexpected padding {(padded_width, padded_height)}; expected "
                f"{(expected_padded_width, expected_padded_height)}"
            )
        for target in targets:
            height, width = (int(value) for value in target["size"])
            if height > padded_height or width > padded_width:
                raise ValueError("collator cropped a transformed image")
        batch_count += 1
        max_padded_width = max(max_padded_width, padded_width)
        max_padded_height = max(max_padded_height, padded_height)

    return {
        "images_verified": len(dataset),
        "batches_verified": batch_count,
        "max_transformed_long_side": max_long_side,
        "max_long_side_cap_overshoot": max(0, max_long_side - 1333),
        "max_padded_shape": [max_padded_width, max_padded_height],
        "max_positive_boxes": max_positive_boxes,
        "max_ignored_boxes": max_ignored_boxes,
        "configured_scales": scales,
        "unique_observed_scale_counts": dict(sorted(unique_selected_scales.items())),
        "cap_ambiguous_images": cap_ambiguous_images,
        "transformed_size_counts": {f"{width}x{height}": count for (width, height), count in sorted(transformed_sizes.items())},
    }


def _verify_target(target: dict[str, Any]) -> None:
    boxes = target.get("boxes")
    ignored_boxes = target.get("ignored_boxes")
    if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("RF-DETR target has invalid ordinary boxes")
    if not isinstance(ignored_boxes, torch.Tensor) or ignored_boxes.ndim != 2 or ignored_boxes.shape[1] != 4:
        raise ValueError("RF-DETR target has invalid ignored boxes")
    if not torch.isfinite(boxes).all() or not torch.isfinite(ignored_boxes).all():
        raise ValueError("RF-DETR target contains non-finite boxes")


def _round_up(value: int, block_size: int) -> int:
    return (value + block_size - 1) // block_size * block_size


if __name__ == "__main__":
    main()