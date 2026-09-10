#!/usr/bin/env python3
"""Run one real ignore-aware RF-DETR 2XL data, loss, backward, and optimizer trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mot20.detection.checkpoint_loading import CLASSIFICATION_HEAD_MARKERS, prepare_expanded_checkpoint_state
from mot20.detection.ignore_criterion import IgnoreAwareSetCriterion
from mot20.detection.ignore_dataset import IgnoreAwareCocoDetection
from mot20.detection.rfdetr_integration import use_ignore_aware_rfdetr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing tracer output: {args.output}")
    if args.num_queries < 1 or args.num_queries % 13:
        raise ValueError("num_queries must be a positive multiple of 13")
    if not args.dataset_root.is_dir() or not args.checkpoint.is_file():
        raise ValueError("tracer requires an existing dataset root and checkpoint")

    from rfdetr.config import TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule
    from rfdetr_plus.models.detection import RFDETR2XLargeConfig

    model_config = RFDETR2XLargeConfig(
        pretrain_weights=str(args.checkpoint),
        num_classes=1,
        resolution=880,
        num_queries=args.num_queries,
        num_select=args.num_queries,
        group_detr=13,
        amp=True,
        device="cuda",
    )
    object.__setattr__(model_config, "pretrain_weights", None)
    train_config = TrainConfig(
        dataset_dir=str(args.dataset_root),
        output_dir=str(args.output.parent),
        dataset_file="roboflow",
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        multi_scale=False,
        use_ema=False,
        amp_dtype="bf16",
        eval_max_dets=args.num_queries,
    )
    device = torch.device("cuda")
    with use_ignore_aware_rfdetr():
        data_module = RFDETRDataModule(model_config, train_config)
        data_module.setup("fit")
        if not isinstance(data_module._dataset_train, IgnoreAwareCocoDetection):
            raise RuntimeError("ignore-aware COCO dataset was not installed")
        model_module = RFDETRModelModule(model_config, train_config).to(device)
        if not isinstance(model_module.criterion, IgnoreAwareSetCriterion):
            raise RuntimeError("ignore-aware criterion was not installed")
        checkpoint_queries = _load_expanded_pretrained_weights(model_module.model, args.checkpoint, args.num_queries)
        optimizer = torch.optim.AdamW(model_module.model.parameters(), lr=1e-4)
        model_module.train()
        for batch_index, batch in enumerate(data_module.train_dataloader()):
            if batch_index >= args.max_batches:
                raise RuntimeError("no transformed ignored boxes were observed within max_batches")
            samples, targets = data_module.transfer_batch_to_device(batch, device, 0)
            ignored_boxes = sum(int(target["ignored_boxes"].shape[0]) for target in targets)
            if not ignored_boxes:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model_module.model(samples, targets)
                losses = model_module.criterion(outputs, targets)
                loss = torch.stack(
                    [losses[key] * model_module.criterion.weight_dict[key] for key in losses if key in model_module.criterion.weight_dict]
                ).sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"tracer loss is not finite: {loss.item()}")
            loss.backward()
            optimizer.step()
            result = {
                "format": "mot20.rfdetr.ignore-tracer.v1",
                "num_queries": args.num_queries,
                "checkpoint_num_queries": checkpoint_queries,
                "batch_index": batch_index,
                "positive_boxes": sum(int(target["boxes"].shape[0]) for target in targets),
                "ignored_boxes": ignored_boxes,
                "loss": float(loss.detach().cpu()),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2, sort_keys=True))
            return


def _load_expanded_pretrained_weights(model: torch.nn.Module, checkpoint_path: Path, target_num_queries: int) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_state = checkpoint.get("model")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("trusted RF-DETR checkpoint has no model state dictionary")
    rows = checkpoint_state.get("query_feat.weight")
    if not isinstance(rows, torch.Tensor) or rows.shape[0] % 13:
        raise ValueError("trusted RF-DETR checkpoint has an invalid query table")
    checkpoint_num_queries = rows.shape[0] // 13
    prepared = prepare_expanded_checkpoint_state(
        checkpoint_state,
        model.state_dict(),
        checkpoint_num_queries,
        target_num_queries,
        group_detr=13,
    )
    incompatible = model.load_state_dict(prepared, strict=False)
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if name != "_kp_active_mask" and not any(marker in name for marker in CLASSIFICATION_HEAD_MARKERS)
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise ValueError(
            "unapproved checkpoint load result: "
            f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint_num_queries


if __name__ == "__main__":
    main()