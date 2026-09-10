"""Phase 5: export L1 detections from the verified validation pipeline.

``RFDETR.predict()`` resizes to a square and cannot reproduce validation
geometry, and hand-building the transform would miss the padding mask that the
validation collator supplies. So detections are captured as a side effect of
the library's own ``evaluate()`` run: ``validation_step`` is wrapped to record
each image's postprocessed boxes, which ``module_model`` already rescales to
original pixels via ``orig_size``.

The mAP that the same run reports is therefore a checksum on the export. If it
matches the metric the checkpoint recorded during training, the detections came
from the correct geometry.

Detections are exported at a low threshold on purpose. BoostTrack boosts
low-confidence detections that match existing tracks *before* applying its own
cut, so a pre-filtered file silently disables that mechanism. A higher-threshold
variant can always be derived by filtering; the reverse is impossible.

Score filtering and greedy NMS are available here as ``--min-score`` and
``--nms-iou``, so the measured-best export (score 0.10, NMS IoU 0.70, worth
+0.63 HOTA) can be produced in one step instead of by a second pass over the
written files. Both stay **off by default**, and deliberately so: the paragraph
above is the reason. An export that has already suppressed boxes cannot be
un-suppressed, so making filtering the default would make the base artifact
lossy and the derivation one-way. Applying either option requires the variant
slug to declare it — see ``_require_slug_declares_filtering``.

Note that ``--expected-map`` checksums the *library's* evaluation, which runs
before any of this filtering. It therefore remains a valid check on export
geometry, but it does not describe the filtered file that gets written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "finetuning/src")
sys.path.insert(0, "tracking/src")

from mot20.detection.rfdetr_integration import use_ignore_aware_rfdetr  # noqa: E402
from mot20.detection.rfdetr_training import (  # noqa: E402
    apply_long_side_cap,
    load_training_config,
)
from mot20_tracking.artifacts import (  # noqa: E402
    DETECTIONS_FORMAT,
    TrackingArtifacts,
    sha256_file,
)
from mot20_tracking.detections import (  # noqa: E402
    SequenceDetections,
    greedy_nms,
    validate_detections,
    write_detections,
)
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import validate_slug  # noqa: E402

PEDESTRIAN_LABEL = 0
"""Model label index for pedestrian.

The ignore-aware dataset installs a ``cat2label`` mapping, so the dataset's
single COCO category (id 1) is remapped to 0-based label 0. Other label values
are untrained class slots that top-k selection can still surface.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="detector variant slug")
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        help=(
            "apply greedy NMS at this IoU while exporting. 0.70 is the measured best "
            "setting (worth +0.63 HOTA); RF-DETR's set prediction applies none. Leaving "
            "this off keeps the export lossless so filtered variants can be derived from it"
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="drop detections below this score before NMS; distinct from --threshold, "
        "which is the capture threshold applied inside the validation loop",
    )
    parser.add_argument("--expected-map", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    return parser.parse_args()


def _require_slug_declares_filtering(variant: str, min_score: float | None, nms_iou: float | None) -> None:
    """Refuse to write a filtered export under a slug that does not say so.

    Every artifact path and cache key in the store is derived from the variant
    slug, so a filtered export named like a raw one is indistinguishable from it
    forever afterwards. The repository's convention encodes both settings in the
    name — ``rfdetr2xl-i4-e8-t010-nms070``. This enforces it rather than trusting
    the caller to remember.
    """
    problems = []
    if min_score is not None and "-t" not in variant:
        problems.append(
            # Convention is the score times 100, zero-padded to three digits:
            # 0.05 -> t005, 0.10 -> t010, matching -nms070 for IoU 0.70.
            f"--min-score {min_score} is applied but the slug has no -t<score> segment "
            f"(convention: -t{int(round(min_score * 100)):03d})"
        )
    if nms_iou is not None and "-nms" not in variant:
        problems.append(
            f"--nms-iou {nms_iou} is applied but the slug has no -nms<iou> segment "
            f"(convention: -nms{int(round(nms_iou * 100)):03d})"
        )
    if problems:
        raise SystemExit(
            f"variant slug {variant!r} does not declare its filtering:\n  "
            + "\n  ".join(problems)
        )


def _load_image_index(dataset_root: Path) -> dict[int, dict[str, object]]:
    """Map COCO image id to its MOT20 sequence and original frame number."""
    annotations = dataset_root / "valid" / "_annotations.coco.json"
    with annotations.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    index: dict[int, dict[str, object]] = {}
    for image in payload["images"]:
        missing = [key for key in ("source_sequence", "source_frame_id") if key not in image]
        if missing:
            raise SystemExit(f"COCO image {image.get('id')} lacks {missing}; cannot map to MOT20")
        index[int(image["id"])] = {
            "sequence": str(image["source_sequence"]),
            "source_frame_id": int(image["source_frame_id"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
        }
    return index


def main() -> None:
    args = parse_args()
    validate_slug(args.variant, "detector")
    _require_slug_declares_filtering(args.variant, args.min_score, args.nms_iou)
    if args.nms_iou is not None and not 0.0 < args.nms_iou <= 1.0:
        raise SystemExit(f"--nms-iou must be in (0, 1], got {args.nms_iou}")
    if args.min_score is not None and args.min_score < args.threshold:
        raise SystemExit(
            f"--min-score {args.min_score} is below the capture threshold {args.threshold}; "
            "detections under the capture threshold were never retained, so this would "
            "imply a completeness the export does not have"
        )
    artifacts = TrackingArtifacts(args.artifact_root)
    level_dir = artifacts.detections_dir(args.variant, args.split)
    if artifacts.manifest_path(level_dir).exists():
        raise SystemExit(f"variant already exported: {level_dir}")

    image_index = _load_image_index(args.dataset_root)
    sequences = {sequence.name: sequence for sequence in read_split(args.split)}

    config = load_training_config(args.config)
    model_config = config["model"]
    capacity = config["capacity"]
    training = config["training"]

    import rfdetr.training.module_model as module_model
    from rfdetr import RFDETR2XLarge

    captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    original_validation_step = module_model.RFDETRModelModule.validation_step
    label_values: set[int] = set()
    # Only PEDESTRIAN_LABEL is written; other label values are untrained class
    # slots that top-k selection can still surface, and writing them as
    # pedestrians would inject pure noise into the tracker.
    dropped_other_class = 0

    def capturing_validation_step(self, batch, batch_idx):  # type: ignore[no-untyped-def]
        nonlocal dropped_other_class
        output = original_validation_step(self, batch, batch_idx)
        for result, target in zip(output["results"], output["targets"]):
            image_id = int(target["image_id"].item())
            if image_id in captured:
                raise RuntimeError(f"image {image_id} evaluated twice; export would be ambiguous")
            scores = result["scores"].detach().float().cpu().numpy()
            boxes = result["boxes"].detach().float().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy()
            label_values.update(int(v) for v in labels.tolist())
            above = scores >= args.threshold
            pedestrian = labels == PEDESTRIAN_LABEL
            dropped_other_class += int((above & ~pedestrian).sum())
            keep = above & pedestrian
            captured[image_id] = (boxes[keep], scores[keep])
        return output

    module_model.RFDETRModelModule.validation_step = capturing_validation_step
    # Must precede dataset construction: the cap is a module-global read when the
    # validation transform is built. A checkpoint trained with a raised max_size
    # exported at the library default would run at geometry it never saw, and the
    # mAP checksum below is what would catch it.
    long_side_cap = apply_long_side_cap(model_config.get("max_size"))
    print(f"long-side cap in force: {long_side_cap}px")
    try:
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
    finally:
        module_model.RFDETRModelModule.validation_step = original_validation_step

    plain_metrics = {str(key): float(value) for key, value in dict(metrics).items()}
    observed_map = plain_metrics.get("val/mAP_50_95")
    print(json.dumps({k: round(v, 4) for k, v in plain_metrics.items() if "mAP" in k or k == "val/mAR"}, indent=2, sort_keys=True))

    if len(captured) != len(image_index):
        raise SystemExit(
            f"captured {len(captured)} images but the split has {len(image_index)}; export incomplete"
        )
    if PEDESTRIAN_LABEL not in label_values:
        raise SystemExit(f"no pedestrian-class predictions; saw labels {sorted(label_values)}")
    print(
        f"labels observed {sorted(label_values)}; dropped {dropped_other_class} "
        f"non-pedestrian detections above threshold"
    )

    if args.expected_map is not None:
        if observed_map is None:
            raise SystemExit("evaluate() returned no val/mAP_50_95")
        delta = abs(observed_map - args.expected_map)
        status = "MATCH" if delta <= args.tolerance else "MISMATCH"
        print(f"\ngeometry checksum: expected {args.expected_map:.4f} observed {observed_map:.4f} -> {status}")
        if status == "MISMATCH":
            raise SystemExit("refusing to write detections from a run that did not reproduce the metric")

    # Group by sequence, then renumber original MOT20 frames to the split's own
    # 1-based numbering, which is what the split's images and ground truth use.
    by_sequence: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    # A detector may emit a box with zero or inverted extent — arm C produced one
    # in MOT20-03 at the 0.05 export threshold. Such a box carries no usable
    # geometry and `validate_detections` rejects it outright, so it is dropped
    # here rather than written. The count is recorded per sequence and in the
    # manifest: dropping is permitted, dropping silently is not.
    degenerate_dropped: dict[str, int] = defaultdict(int)
    score_dropped = 0
    nms_suppressed = 0
    for image_id, meta in image_index.items():
        boxes, scores = captured[image_id]
        rows = np.column_stack([boxes, scores]).astype(np.float32) if len(scores) else np.empty((0, 5), np.float32)
        if rows.shape[0]:
            keep = (rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])
            dropped = int((~keep).sum())
            if dropped:
                degenerate_dropped[str(meta["sequence"])] += dropped
                rows = rows[keep]
        # Score filter before NMS, in that order: suppression should be decided
        # among the boxes that survive, not by boxes that are about to be
        # discarded. Degenerate boxes are already gone, so NMS never sees a
        # zero-area box and cannot divide by a zero union.
        if args.min_score is not None and rows.shape[0]:
            above = rows[:, 4] >= args.min_score
            score_dropped += int((~above).sum())
            rows = rows[above]
        if args.nms_iou is not None and rows.shape[0]:
            survivors = greedy_nms(rows, args.nms_iou)
            nms_suppressed += int((~survivors).sum())
            rows = rows[survivors]
        by_sequence[str(meta["sequence"])][int(meta["source_frame_id"])] = rows

    entries = []
    for name in sorted(by_sequence):
        sequence = sequences[name]
        source_frames = by_sequence[name]
        offset = min(source_frames) - 1
        renumbered = {frame - offset: rows for frame, rows in source_frames.items()}
        detections = SequenceDetections(sequence=name, frames=renumbered)
        statistics = validate_detections(
            detections, sequence.width, sequence.height, sequence.length
        )
        destination = artifacts.detections_file(args.variant, args.split, name)
        write_detections(destination, detections, sequence.length)
        entries.append(
            {
                "sequence": name,
                "source_frame_offset": offset,
                "written_sha256": sha256_file(destination),
                "width": sequence.width,
                "height": sequence.height,
                "degenerate_boxes_dropped": degenerate_dropped.get(name, 0),
                **statistics,
            }
        )
        print(
            f"{name}: {statistics['detections']:>7} detections, "
            f"max/frame {statistics['max_per_frame']:>3}, "
            f"score {statistics['min_score']:.4f}-{statistics['max_score']:.4f}, "
            f"empty frames {statistics['empty_frames']}, "
            f"degenerate dropped {degenerate_dropped.get(name, 0)}"
        )

    total_degenerate = sum(degenerate_dropped.values())
    print(f"degenerate boxes dropped across the split: {total_degenerate}")
    if args.min_score is not None:
        print(f"score filter at {args.min_score}: {score_dropped} detections dropped")
    if args.nms_iou is not None:
        print(f"greedy NMS at IoU {args.nms_iou}: {nms_suppressed} detections suppressed")
    if args.min_score is None and args.nms_iou is None:
        print("no score filter or NMS applied; this export is lossless")

    digest = artifacts.write_manifest(
        level_dir,
        {
            "format": DETECTIONS_FORMAT,
            "variant": args.variant,
            "split": args.split,
            "origin": "rfdetr-evaluate-capture",
            "description": "RF-DETR detections captured from the library validation loop",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "dataset_root": str(args.dataset_root),
            "export_threshold": args.threshold,
            "num_select": capacity["num_select"],
            "min_score": args.min_score,
            "nms": (
                f"greedy, IoU {args.nms_iou}, applied at export"
                if args.nms_iou is not None
                else "none (DETR set prediction)"
            ),
            "nms_iou": args.nms_iou,
            "filtering": {
                "order": "degenerate-box drop, then score filter, then NMS",
                "score_filtered": score_dropped,
                "nms_suppressed": nms_suppressed,
                "lossless": args.min_score is None and args.nms_iou is None,
            },
            "class_filter": {
                "kept_label": PEDESTRIAN_LABEL,
                "labels_observed": sorted(label_values),
                "dropped_non_pedestrian_above_threshold": dropped_other_class,
            },
            "geometry": {
                "resolution": model_config["resolution"],
                "max_size": long_side_cap,
                "transform": f"RandomResize([resolution], max_size={long_side_cap}), aspect-preserving",
                "collate_block_size": 40,
            },
            "verification": {
                "expected_map_50_95": args.expected_map,
                "observed_map_50_95": observed_map,
                "metrics": plain_metrics,
            },
            "sequences": entries,
            "totals": {
                "sequences": len(entries),
                "frames": sum(int(entry["frames"]) for entry in entries),
                "detections": sum(int(entry["detections"]) for entry in entries),
            },
        },
    )
    print(f"\nmanifest {artifacts.manifest_path(level_dir)} sha256 {digest}")


if __name__ == "__main__":
    main()
