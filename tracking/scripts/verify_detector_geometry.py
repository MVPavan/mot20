"""Phase 4 gate: prove inference geometry matches training validation.

RF-DETR's ``predict()`` resizes to a square, while training validation resizes
the short side toward ``resolution`` with an aspect-preserving 1333px cap. A
detection export built on the wrong geometry looks plausible and scores worse
for no visible reason, so this gate runs the library's own validation loop on
the selected checkpoint and compares the result against the metric that
checkpoint recorded during training.

Reproducing the expected value proves the whole dataset build, transform,
collate, and postprocess chain is the one that produced the reported mAP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "finetuning/src")

from mot20.detection.rfdetr_integration import use_ignore_aware_rfdetr  # noqa: E402
from mot20.detection.rfdetr_training import load_training_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="training TOML used for the run")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--expected-map",
        type=float,
        default=None,
        help="mAP_50_95 recorded for this checkpoint during training",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="override RF-DETR's 1333px long-side cap. At MOT20's 1920x1080 that cap, "
        "not --resolution, is what binds: the short side never reaches 1120. Raising it "
        "is a probe of whether localization is resolution-limited, and it deliberately "
        "departs from the geometry this gate otherwise verifies, so --expected-map will "
        "not match.",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional JSON result path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    model_config = config["model"]
    capacity = config["capacity"]
    training = config["training"]

    if args.max_size is not None:
        from rfdetr.datasets import coco as rfdetr_coco

        rfdetr_coco._COCO_MAX_SIZE = args.max_size
        print(f"long-side cap overridden to {args.max_size}px")

    from rfdetr import RFDETR2XLarge

    # num_queries/num_select/group_detr are ModelConfig fields, not evaluate()
    # kwargs. If they are not set at construction they silently fall back to
    # library defaults (300/300) and the metric will not match.
    model = RFDETR2XLarge(
        pretrain_weights=str(args.checkpoint),
        num_classes=model_config["num_classes"],
        resolution=model_config["resolution"],
        amp=model_config["amp"],
        gradient_checkpointing=model_config["gradient_checkpointing"],
        num_queries=capacity["num_queries"],
        num_select=capacity["num_select"],
        group_detr=capacity["group_detr"],
        device="cuda",
    )
    for field in ("num_queries", "num_select", "group_detr"):
        actual = getattr(model.model_config, field, None)
        if actual != capacity[field if field != "num_select" else "num_select"]:
            raise SystemExit(f"model_config.{field} is {actual}, expected {capacity[field]}")

    with use_ignore_aware_rfdetr(config["run"]["ignored_iou_threshold"]):
        metrics = model.evaluate(
            split="val",
            dataset_dir=str(args.dataset_root),
            dataset_file="roboflow",
            device="cuda",
            resolution=model_config["resolution"],
            class_names=["pedestrian"],
            eval_max_dets=capacity["eval_max_dets"],
            multi_scale=training["multi_scale"],
            do_random_resize_via_padding=training["do_random_resize_via_padding"],
            square_resize_div_64=training["square_resize_div_64"],
            scale_jitter=training["scale_jitter"],
        )

    plain = {str(key): float(value) for key, value in dict(metrics).items()}
    print(json.dumps(plain, indent=2, sort_keys=True))

    observed = plain.get("val/mAP_50_95")
    verdict = None
    if args.expected_map is not None:
        if observed is None:
            raise SystemExit("evaluate() returned no val/mAP_50_95 to compare")
        delta = abs(observed - args.expected_map)
        verdict = "MATCH" if delta <= args.tolerance else "MISMATCH"
        print(
            f"\nexpected {args.expected_map:.4f} observed {observed:.4f} "
            f"delta {delta:.5f} -> {verdict}"
        )

    if args.output is not None:
        payload = {
            "format": "mot20.tracking.geometry-gate.v1",
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "dataset_root": str(args.dataset_root),
            "expected_map_50_95": args.expected_map,
            "metrics": plain,
            "verdict": verdict,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if verdict == "MISMATCH":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
