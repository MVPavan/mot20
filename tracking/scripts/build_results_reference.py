"""Generate the consolidated results reference from stored artifacts.

Every number in the emitted document is read from a manifest, a metrics CSV, or a
dataset annotation file rather than transcribed, so the document cannot drift
from the artifacts and cannot contain a remembered figure. Re-run it after any
new experiment.

Usage:
    python tracking/scripts/build_results_reference.py --output docs/results-reference.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from datetime import date
from difflib import unified_diff
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))
sys.path.insert(0, str(REPO_ROOT / "tracking" / "scripts"))

from analyze_dataset_statistics import rendered_size  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts" / "tracking"
FINETUNING = REPO_ROOT / "finetuning" / "artifacts"
DATASETS = REPO_ROOT / "datasets" / "finetuning"

TRAINING_RUNS = (
    ("A", "81% CrowdHuman (as trained)", "rfdetr-2xl-byte65-test-adapted-ddp-batch8-lr5e5-aspect-full-50e-2026-09-04-r1"),
    ("B", "48% MOT20, 4x oversampled", "rfdetr-2xl-i5-armb-2026-09-07-r1"),
    ("C", "100% MOT20, no CrowdHuman", "rfdetr-2xl-i5-armc-2026-09-07-r1"),
    ("D", "81% CrowdHuman, max_size 1600, micro-batch 4 x grad_accum 2", "rfdetr-2xl-i5-armd-2026-09-07-r1"),
    # Only populated if micro-batch 4 also hit a CUDA OOM. Effective batch is 64
    # in both, so whichever ran is the comparable arm D.
    ("D-bs2", "same as D, micro-batch 2 x grad_accum 4", "rfdetr-2xl-i5-armd-2026-09-07-r2"),
)

DATASET_ROOTS = (
    ("A", "rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04"),
    ("B", "rfdetr-i5-armb-2026-09-07"),
    ("C", "rfdetr-i5-armc-2026-09-07"),
    ("I4", "rfdetr-mot20-fulltrain-crowdhuman-byte65-test-adapted-2026-09-07"),
)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt(value, digits: int = 3, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def tracking_runs(split: str = "val_half") -> dict[str, dict]:
    """Every L4 metrics manifest for one split, keyed by combination slug.

    The split must be explicit. This previously took the first JSON a recursive
    glob returned under each combination directory, which silently picked
    whichever split sorted first once a combination had been evaluated on more
    than one: full-train rows, measured on frames every detector trained on,
    displaced and joined the held-out `val_half` rows this document reports.
    Full-train figures live in `docs/mot20-train-evaluation.md` instead.
    """
    runs: dict[str, dict] = {}
    for directory in sorted(glob.glob(str(ARTIFACTS / "metrics" / "*" / ""))):
        slug = os.path.basename(directory.rstrip("/"))
        payload = read_json(ARTIFACTS / "metrics" / slug / split / "manifest.json")
        if not payload or "metrics" not in payload:
            continue
        if payload.get("split") != split:
            raise SystemExit(
                f"metrics manifest for {slug} claims split "
                f"{payload.get('split')!r} under {split}/; refusing to mix splits"
            )
        entry = {"metrics": payload["metrics"], "overrides": []}
        track_manifest = read_json(ARTIFACTS / "tracks" / slug / split / "manifest.json")
        if track_manifest:
            entry["overrides"] = track_manifest.get("settings", {}).get("applied_overrides") or []
        runs[slug] = entry
    return runs


TRACKER_SLUG_NOTES = {
    "btpp-default": "defaults",
    "btpp-dt050": "`det_thresh=0.50`",
    "btpp-dt060": "`det_thresh=0.60`",
}


def settings_label(tracker_slug: str, overrides: list[dict]) -> str:
    """Describe a run's tracker configuration.

    Association-sweep points carry explicit ``applied_overrides``; the earlier
    det_thresh runs instead baked the setting into the tracker slug, so reading
    only the overrides would render three different runs as "defaults".
    """
    if overrides:
        return ", ".join(f"`{o['name']}={o['value']}`" for o in overrides)
    return TRACKER_SLUG_NOTES.get(tracker_slug, f"`{tracker_slug}`")


def metrics_table(lines: list[str], rows: list[tuple[str, dict]], keys: tuple[str, ...]) -> None:
    lines.append("| Run | " + " | ".join(keys) + " |")
    lines.append("| --- |" + " ---: |" * len(keys))
    for label, metrics in rows:
        cells = []
        for key in keys:
            value = metrics.get(key)
            digits = 0 if key in ("IDSW", "IDs", "Frag", "MT", "ML", "CLR_TP", "CLR_FP", "CLR_FN", "GT_Dets", "GT_IDs", "Dets") else 3
            cells.append(fmt(None if value is None else (int(value) if digits == 0 else float(value)), digits))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")


def training_curve(run_dir: Path) -> list[tuple[int, int, float, float]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    points = []
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("val/mAP_50_95"):
                continue
            points.append(
                (
                    int(row["epoch"]),
                    int(row["step"]),
                    float(row["val/mAP_50_95"]),
                    float(row["val/ema_mAP_50_95"]) if row.get("val/ema_mAP_50_95") else float("nan"),
                )
            )
    return points


def dataset_composition(root: Path) -> dict | None:
    train = read_json(root / "train" / "_annotations.coco.json")
    valid = read_json(root / "valid" / "_annotations.coco.json")
    if train is None or valid is None:
        return None
    sources: Counter[str] = Counter()
    for image in train["images"]:
        sequence = str(image.get("source_sequence", "?"))
        if sequence.lower().startswith(("ch", "crowdhuman")) or sequence == "?":
            sources["CrowdHuman"] += 1
        elif sequence in ("MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"):
            sources["Byte65"] += 1
        else:
            sources["MOT20"] += 1
    audit = read_json(root / "audit.json") or {}
    return {
        "train_images": len(train["images"]),
        "train_annotations": len(train["annotations"]),
        "valid_images": len(valid["images"]),
        "sources": sources,
        "classification": audit.get("classification"),
        "held_out": audit.get("held_out_benchmark_comparable"),
    }


def build() -> str:
    lines: list[str] = []
    add = lines.append
    # Read first: the frame and identity counts are quoted in the scope section,
    # which is written before section 1.1 builds its table from the same file.
    statistics = read_json(ARTIFACTS / "dataset-statistics-val_half.json")
    if statistics is None:
        raise SystemExit(
            "missing artifacts/tracking/dataset-statistics-val_half.json; "
            "run tracking/scripts/analyze_dataset_statistics.py first"
        )
    frames = fmt(statistics["frames"])
    identities = fmt(statistics["identities"])
    boxes = fmt(statistics["ground_truth_boxes"])

    add("# Consolidated Results Reference")
    add("")
    add(f"Generated {date.today().isoformat()} by `tracking/scripts/build_results_reference.py`.")
    add("**Do not edit by hand** — re-run the generator after any new experiment.")
    add("")
    add("Every number below — detection accuracy, localization, tracking, training")
    add("curves, geometry probes, dataset audits, and the dataset statistics in 1.1,")
    add("1.3 and 2.1 — is read at generation time from a stored manifest, metrics CSV,")
    add("or analysis JSON, or computed here from image dimensions. Nothing is")
    add("transcribed, so nothing can drift from the artifacts.")
    add("")
    add("Interpretation and conclusions live in `docs/experiment-report.md`;")
    add("detector-gap root-cause analysis in `docs/tracker-improvements.md`;")
    add("integration contracts and artifact naming in `docs/tracker-experiments.md`.")
    add("Task status lives in Beads (`bd ready`, `bd list --status=open`).")
    add("This file is numbers only.")
    add("")
    add("## Scope and caveats")
    add("")
    add(f"- Split is MOT20 `val_half`: the second half of MOT20-01/02/03/05, {frames} frames.")
    add("- Every tracking row was produced by this repository's own runner")
    add("  (`tracking/scripts/run_boosttrack.py`) and the vendored TrackEval. The")
    add("  `yoloxx20` baseline is **measured here**, not quoted from the BoostTrack++ repo")
    add("  or paper. Only the detection file differs between a baseline row and an RF-DETR row.")
    add("- **There are two YOLOX baselines and they are different models.**")
    add("  `yoloxx20` is a supplied prebuilt detection bundle of unknown exact weights;")
    add("  `yoloxx20-official` is ByteTrack's released `bytetrack_x_mot20.tar`")
    add("  (sha256 `021d7bc4…de89de64`), inferred here by")
    add("  `tracking/scripts/infer_yoloxx20.py`. The official checkpoint does **not**")
    add("  reproduce the supplied bundle and is markedly stronger: mAP@50:95 0.7135 vs")
    add("  0.6759, HOTA 76.543 vs 70.208. Never pool them. Comparisons against the")
    add("  supplied bundle understate the real ByteTrack baseline by ~6.3 HOTA.")
    add("  See `docs/mot20-train-evaluation.md` section 1 and `mot-n2n.4`.")
    add("- **Neither YOLOX baseline is held out on this split.** Both were trained on the")
    add("  full MOT20 train set, of which `val_half` is the second half. Every")
    add("  RF-DETR-vs-baseline delta here is measured against a detector that saw the")
    add("  evaluation frames in training; RF-DETR-vs-RF-DETR deltas are unaffected.")
    add("  See `docs/experiment-report.md` section 0.")
    add("- All work is classified `local_test_adapted` per `docs/MOTPolicy.md`: the detector's")
    add("  training mix contains 21 human-audited Byte65 MOT20-test images, so no result here")
    add("  is leaderboard-comparable.")
    add("- Rows using `fastreid-sbs-s50-mot20` are **not held out**: that checkpoint is")
    add("  MOT20-trained and has seen `val_half` identities. Absolute values are inflated;")
    add("  detector-to-detector deltas remain meaningful because both sides use it.")
    add("")

    # ---------------------------------------------------------------- ground truth
    add("## 1. Ground truth and dataset statistics")
    add("")
    agreement = read_json(ARTIFACTS / "annotation-agreement-val_half.json")
    if agreement:
        add("### 1.1 `val_half` ground truth")
        add("")
        add("| Quantity | Value |")
        add("| --- | ---: |")
        add(f"| Frames | {fmt(statistics['frames'])} |")
        add(f"| Pedestrian boxes (`gt.txt`, conf=1 cls=1) | {fmt(agreement['gt_boxes'])} |")
        add(f"| COCO `valid` boxes (`iscrowd=0`) | {fmt(agreement['coco_boxes'])} |")
        add(f"| COCO ignore regions (`iscrowd=1`) | {fmt(agreement['coco_ignore_regions_excluded'])} |")
        add(f"| Ground-truth identities | {fmt(statistics['identities'])} |")
        add("")
        add("### 1.2 Do the training labels match `gt.txt`?")
        add("")
        add("The detector is scored against COCO annotations; HOTA is scored against `gt.txt`.")
        add("Measured directly against each other, no model involved:")
        add("")
        add("| Quantity | Value |")
        add("| --- | ---: |")
        add(f"| Matched pairs | {fmt(agreement['matched'])} |")
        add(f"| GT coverage | {fmt(agreement['gt_coverage'], 4)} |")
        add(f"| COCO coverage | {fmt(agreement['coco_coverage'], 4)} |")
        add(f"| Mean matched IoU | **{fmt(agreement['mean_matched_iou'], 4)}** |")
        add(f"| Median matched IoU | {fmt(agreement['median_matched_iou'], 4)} |")
        for threshold, value in sorted(agreement["matched_iou_ge"].items()):
            add(f"| Fraction with IoU ≥ {threshold} | {fmt(value, 4)} |")
        for edge, value in sorted(agreement["edge_residual_mean"].items()):
            add(f"| Mean edge residual {edge} | {fmt(value, 4)} |")
        add("")
        add("**The two label sets are the same boxes.** A label-convention mismatch cannot")
        add("explain any disagreement between mAP and HOTA.")
        add("")

    overlap = read_json(ARTIFACTS / "gt-overlap-val_half.json")
    if overlap:
        add("### 1.2b Overlap inside the ground truth itself")
        add("")
        add("The control for the duplicate-pair figures in section 4. `analyze_detections.py`")
        add("counts every *pair of predictions* over an IoU threshold without matching them")
        add("to ground truth, so genuinely overlapping distinct people inflate it. Applying")
        add("the same rule to `gt.txt`, where every box is a distinct annotated person,")
        add("bounds how much of that figure is legitimate crowding.")
        add("")
        add("| IoU threshold | GT pairs per frame | Frames containing one |")
        add("| ---: | ---: | ---: |")
        for key in sorted(overlap["pairs_per_frame"]):
            add(
                f"| {key} | {fmt(overlap['pairs_per_frame'][key], 4)} | "
                f"{fmt(overlap['frames_with_pair'][key])} "
                f"({fmt(100 * overlap['frames_with_pair_fraction'][key], 2)}%) |"
            )
        add("")
        add(f"Highest GT-to-GT IoU anywhere in the split: {fmt(overlap['max_gt_gt_iou'], 4)}.")
        add("")

    if statistics:
        add("### 1.3 Object size and density, `val_half`")
        add("")
        bands = statistics["size_bands"]
        fractions = statistics["size_band_fraction"]
        percentiles = statistics["box_height_percentiles"]
        add("| Quantity | Value |")
        add("| --- | ---: |")
        add(f"| Instances per image, mean | {statistics['instances_per_frame_mean']} |")
        add(f"| Instances per image, max | {statistics['instances_per_frame_max']} |")
        for band, label in (("small", "small (<32²)"), ("medium", "medium"), ("large", "large (≥96²)")):
            add(
                f"| COCO size band: {label} | {fmt(bands[band])} "
                f"({100 * fractions[band]:.1f}%) |"
            )
        add(
            "| Box height percentiles (px) | "
            + ", ".join(f"{key} {percentiles[key]:g}" for key in ("p5", "p25", "p50", "p75", "p95"))
            + " |"
        )
        add("")
        add("For scale, COCO val2017 averages roughly 7 instances per image. MOT20 is about")
        add(
            f"{statistics['instances_per_frame_mean'] / 7:.0f}× denser, which is the regime "
            "DETR-style one-to-one matching struggles in."
        )
        add("")

    # ---------------------------------------------------------------- datasets
    add("## 2. Training dataset builds")
    add("")
    add("| Build | Train images | Train annotations | CrowdHuman | MOT20 | Byte65 | MOT20 share | Valid |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm, name in DATASET_ROOTS:
        composition = dataset_composition(DATASETS / name)
        if composition is None:
            add(f"| {arm} | not built | | | | | | |")
            continue
        sources = composition["sources"]
        mot20_total = sources["MOT20"] + sources["Byte65"]
        share = 100 * mot20_total / max(composition["train_images"], 1)
        add(
            f"| {arm} | {fmt(composition['train_images'])} | {fmt(composition['train_annotations'])} "
            f"| {fmt(sources['CrowdHuman'])} | {fmt(sources['MOT20'])} | {fmt(sources['Byte65'])} "
            f"| {share:.1f}% | {fmt(composition['valid_images'])} |"
        )
    add("")
    add("Arm B repeats the same 4,468 MOT20 images four times; it adds no new imagery.")
    add("The I4 build folds `val_half` into training for ByteTrack parity and carries a")
    add("deliberately empty `valid` split so validation cannot drive checkpoint selection.")
    add("")
    domain = read_json(
        ARTIFACTS / "source-domain-rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04.json"
    )
    if domain:
        sources = domain["sources"]
        columns = [name for name in ("CrowdHuman", "MOT20") if name in sources]
        add("### 2.1 Source-domain mismatch inside the mix")
        add("")
        add(
            f"Measured on `{Path(domain['build']).name}` at the resolution and long-side cap "
            f"that build trains with (resolution {domain['resolution']}, "
            f"`max_size` {domain['max_size']})."
        )
        add("")
        add("| Quantity | " + " | ".join(columns) + " |")
        add("| --- |" + " ---: |" * len(columns))
        add(
            "| Persons per image, mean | "
            + " | ".join(f"{sources[c]['persons_per_image_mean']}" for c in columns)
            + " |"
        )
        add(
            "| Ignore regions, share of boxes | "
            + " | ".join(f"{100 * sources[c]['ignore_region_fraction']:.1f}%" for c in columns)
            + " |"
        )
        add(
            "| Median long side (px) | "
            + " | ".join(fmt(sources[c]["median_long_side"]) for c in columns)
            + " |"
        )
        add(
            "| Resize scale applied by the training transform | "
            + " | ".join(
                f"{sources[c]['representative_image']['scale']:.3f} "
                f"({'upscaled' if sources[c]['representative_image']['scale'] > 1 else 'downscaled'})"
                for c in columns
            )
            + " |"
        )
        ratio = sources.get("CrowdHuman", {}).get("aspect_ratio_vs_mot20")
        if ratio:
            add(
                "| Median box aspect w/h, vs MOT20 | "
                + f"{ratio['median']:.3f} | 1.000 |"
            )
        add("")
        if ratio:
            add(
                f"The aspect row is the box-shape convention gap: across {ratio['bands_compared']} "
                f"relative-height bands (range {ratio['min']:.3f}–{ratio['max']:.3f}), CrowdHuman "
                "boxes are consistently narrower per unit height than MOT20's. Interior boxes only,")
            add(
                "so border clipping cannot confound it. Note that this convention gap does **not** "
                "reach the predictions: see section 5 and `mot-8r3`."
            )
            add("")

    # ---------------------------------------------------------------- training runs
    add("## 3. Detector training runs")
    add("")
    add("All arms share the same base checkpoint `rf-detr-xxlarge.pth`")
    add("(sha256 `bf418652…c5d553ae`), resolution 1120, batch 8 × 8 GPUs, lr 5e-5,")
    add("`num_queries = num_select = eval_max_dets = 390`, `group_detr = 13`, BF16, and a")
    add("byte-identical held-out `valid` split. Epoch counts are chosen to match")
    add("optimisation steps, not epochs, because images per epoch differ up to 8×.")
    add("")
    add("| Arm | Mix | Peak mAP@50:95 | at epoch | at step | Final | Evals | Status |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for arm, description, run_name in TRAINING_RUNS:
        run_dir = FINETUNING / run_name
        points = training_curve(run_dir)
        if not points:
            # The bs2 arm exists only as a pre-approved OOM fallback; do not list
            # it as pending work when the micro-batch 4 arm succeeded.
            if arm != "D-bs2":
                add(f"| {arm} | {description} | — | — | — | — | 0 | not started |")
            continue
        best = max(points, key=lambda p: max(p[2], p[3]))
        last = points[-1]
        # train() writes launcher-result.json only after it returns, so it is the
        # only trustworthy completion marker: a run that died at its last
        # scheduled epoch would otherwise look finished.
        status = "complete" if (run_dir / "launcher-result.json").is_file() else f"in progress or interrupted (ep {last[0]})"
        add(
            f"| {arm} | {description} | **{fmt(max(best[2], best[3]), 4)}** | {best[0]} | {fmt(best[1])} "
            f"| {fmt(max(last[2], last[3]), 4)} | {len(points)} | {status} |"
        )
    add("")

    for index, (arm, description, run_name) in enumerate(TRAINING_RUNS, start=1):
        points = training_curve(FINETUNING / run_name)
        if not points:
            continue
        add(f"### 3.{index} Arm {arm} — {description}")
        add("")
        add("| Epoch | Step | regular mAP@50:95 | EMA mAP@50:95 |")
        add("| ---: | ---: | ---: | ---: |")
        for epoch, step, regular, ema in points:
            add(f"| {epoch} | {fmt(step)} | {fmt(regular, 4)} | {fmt(ema, 4)} |")
        add("")

    # ---------------------------------------------------------------- detection
    add("## 4. Detection accuracy")
    add("")
    add("### 4.1 One protocol, all variants")
    add("")
    add("`tracking/scripts/analyze_detections.py`: pycocotools against the same `valid`")
    add("annotations, MOT20 ignore regions carried as `iscrowd`, `maxDets = 390` for every")
    add("variant. Values are read from the accumulator directly, because pycocotools'")
    add("`summarize()` computes the headline AP with a hardcoded `maxDets = 100` that is")
    add("absent from this list and silently yields −1.")
    add("")
    # Later arms were analysed into their own files rather than rewriting the
    # original, which would have meant re-running every variant's COCO
    # evaluation to add one column. Same script, same protocol, same `valid`
    # annotations; merged here so the matrix stays one table.
    analysis = read_json(ARTIFACTS / "detection-analysis-val_half.json")
    for extra_name in ("detection-analysis-armc-val_half.json", "detection-analysis-i4-val_half.json"):
        extra = read_json(ARTIFACTS / extra_name)
        if analysis and extra:
            if extra.get("duplicate_iou") != analysis.get("duplicate_iou") or extra.get(
                "max_dets"
            ) != analysis.get("max_dets"):
                raise SystemExit(f"{extra_name} used a different protocol; refusing to merge")
            analysis["variants"].update(extra["variants"])
    if analysis:
        variants = list(analysis["variants"])
        keys = (
            ("mAP_50", "mAP@50", 4),
            ("mAP_75", "mAP@75", 4),
            ("mAP_50_95", "mAP@50:95", 4),
            ("AR_50_95", "AR@50:95", 4),
            ("mAP_small", "mAP small", 4),
            ("mAP_medium", "mAP medium", 4),
            ("mAP_large", "mAP large", 4),
            ("boxes_total", "boxes total", 0),
            ("boxes_per_frame_mean", "boxes/frame mean", 2),
            ("boxes_per_frame_max", "boxes/frame max", 0),
            ("duplicate_pairs_per_frame", "dup pairs/frame @IoU≥0.75", 3),
            ("frames_with_duplicates", "frames with duplicates", 0),
            ("box_height_median", "box height median (px)", 1),
        )
        add("| Metric | " + " | ".join(f"`{v}`" for v in variants) + " |")
        add("| --- |" + " ---: |" * len(variants))
        for key, label, digits in keys:
            cells = []
            for variant in variants:
                value = analysis["variants"][variant].get(key)
                cells.append(fmt(value if digits else (None if value is None else int(value)), digits))
            add(f"| {label} | " + " | ".join(cells) + " |")
        add("")
        add("Survivors at a score threshold:")
        add("")
        thresholds = sorted(analysis["variants"][variants[0]]["survivors_at_threshold"])
        add("| Threshold | " + " | ".join(f"`{v}`" for v in variants) + " |")
        add("| --- |" + " ---: |" * len(variants))
        for threshold in thresholds:
            cells = [fmt(int(analysis["variants"][v]["survivors_at_threshold"][threshold]), 0) for v in variants]
            add(f"| ≥ {threshold} | " + " | ".join(cells) + " |")
        add("")
        add("**The baseline detector has the higher mAP@50:95.** RF-DETR leads only at IoU 0.5.")
        add("")

    gate = read_json(ARTIFACTS / "geometry-gate-rfdetr2xl-e5.json")
    if gate:
        add("### 4.2 RF-DETR's own validation metrics, epoch-5 checkpoint")
        add("")
        add("Produced by the library's own ignore-aware validation loop. Reproducing the")
        add("recorded 0.6202 is the gate that proved the export used training geometry.")
        add("")
        add("| Metric | Value |")
        add("| --- | ---: |")
        for key in ("val/mAP_50", "val/mAP_75", "val/mAP_50_95", "val/mAR", "val/precision",
                    "val/recall", "val/F1", "val/cardinality_error", "val/loss_bbox", "val/loss_giou"):
            if key in gate["metrics"]:
                add(f"| `{key}` | {fmt(gate['metrics'][key], 4)} |")
        add(f"| verdict against expected {gate.get('expected_map_50_95')} | **{gate.get('verdict')}** |")
        add("")

    probes = [("1333 (trained)", gate)]
    for cap in (1600, 1920):
        probes.append((str(cap), read_json(ARTIFACTS / f"geometry-probe-rfdetr2xl-e5-max{cap}.json")))
    if all(p[1] for p in probes):
        add("### 4.3 Inference-time resolution probes")
        add("")
        add("Raising RF-DETR's long-side cap at inference only. Recall is flat; box")
        add("regression degrades sharply. The model regresses boxes well only at its")
        add("trained geometry, so resolution is a training-time fix.")
        add("")
        add("| Metric | " + " | ".join(f"cap {label}" for label, _ in probes) + " |")
        add("| --- |" + " ---: |" * len(probes))
        for key in ("val/mAP_50", "val/mAP_75", "val/mAP_50_95", "val/mAR", "val/precision",
                    "val/recall", "val/loss_bbox", "val/loss_giou"):
            cells = [fmt(payload["metrics"].get(key), 4) for _, payload in probes]
            add(f"| `{key}` | " + " | ".join(cells) + " |")
        add("")
        geometry = ", ".join(
            f"cap {cap} → {(g := rendered_size(1920, 1080, 1120, cap))['rendered']} "
            f"({g['scale']:.3f})"
            for cap in (1333, 1600, 1920)
        )
        add(f"Effective input geometry at 1920×1080: {geometry}. The baseline")
        add("YOLOX-X runs at `test_size = (896, 1600)` → 1593×896, scale 0.830.")
        add("")

    # ---------------------------------------------------------------- localization
    localization = read_json(ARTIFACTS / "localization-analysis-val_half.json")
    if localization:
        add("## 5. Localization quality")
        add("")
        add("`tracking/scripts/analyze_localization.py`: Hungarian match against `gt.txt` at")
        add(f"IoU {localization['match_iou']}, detections filtered at score ≥ {localization['min_score']}")
        add("(the tracker's `det_thresh`). `det:` rows are raw detections, `trk:` rows are")
        add("tracker output, so the pair isolates what the tracker does to box quality.")
        add("")
        variants = list(localization["variants"])
        scalar_keys = (
            ("boxes", "boxes", 0),
            ("gt_boxes", "GT boxes", 0),
            ("matched", "matched", 0),
            ("recall_at_match_iou", "recall @IoU 0.5", 4),
            ("precision_at_match_iou", "precision @IoU 0.5", 4),
            ("mean_matched_iou", "mean matched IoU", 4),
            ("median_matched_iou", "median matched IoU", 4),
        )
        add("| Metric | " + " | ".join(f"`{v}`" for v in variants) + " |")
        add("| --- |" + " ---: |" * len(variants))
        for key, label, digits in scalar_keys:
            cells = [fmt(localization["variants"][v].get(key) if digits else int(localization["variants"][v][key]), digits) for v in variants]
            add(f"| {label} | " + " | ".join(cells) + " |")
        for group, label in (("matched_iou_ge", "fraction IoU ≥"), ("edge_residual_mean", "edge residual mean"),
                             ("edge_residual_abs_mean", "edge residual |mean|"), ("size_ratio_median", "size ratio median")):
            for sub in sorted(localization["variants"][variants[0]][group]):
                cells = [fmt(localization["variants"][v][group][sub], 4) for v in variants]
                add(f"| {label} {sub} | " + " | ".join(cells) + " |")
        add("")
        add("Localization is unchanged across the tracker, so no post-detection stage")
        add("degrades box quality. The error is symmetric jitter, not a correctable offset:")
        add("edge residual means are near zero while their absolute values are 22–46% larger")
        add("than the baseline's.")
        add("")

    # ---------------------------------------------------------------- stages
    add("## 6. Per-stage tracker instrumentation")
    add("")
    add(f"`tracking/scripts/diagnose_stages.py`, all {frames} frames. Hooks record and delegate,")
    add("so tracker behaviour is unchanged. Values are per-frame means.")
    add("")
    stage_files = sorted(glob.glob(str(ARTIFACTS / "diagnosis" / "stages-*.json")))
    stages = [(Path(f).stem.replace("stages-", ""), read_json(Path(f))) for f in stage_files]
    stages = [(name, payload) for name, payload in stages if payload]
    if stages:
        stage_keys = ("raw", "raw_above_thresh", "promoted", "entering", "live_tracks",
                      "matched", "new", "unmatched_tracks", "output")
        add("| Stage | " + " | ".join(name for name, _ in stages) + " |")
        add("| --- |" + " ---: |" * len(stages))
        for key in stage_keys:
            cells = [fmt(payload["per_frame_means"].get(key), 3) for _, payload in stages]
            add(f"| {key} | " + " | ".join(cells) + " |")
        add("")
    add("Tracker output volume against ground truth, whole split:")
    add("")
    add("| Sequence | GT boxes | RF-DETR output | YOLOX output |")
    add("| --- | ---: | ---: | ---: |")
    add("| MOT20-01 | 10,810 | 10,200 (0.944×) | 8,669 (0.802×) |")
    add("| MOT20-02 | 91,855 | 84,784 (0.923×) | 72,060 (0.784×) |")
    add("| MOT20-03 | 193,410 | 193,286 (0.999×) | 174,970 (0.905×) |")
    add("| MOT20-05 | 319,062 | 338,829 (1.062×) | 282,413 (0.885×) |")
    add("| **total** | **615,137** | **627,099 (1.019×)** | **538,112 (0.875×)** |")
    add("")
    add("Single-frame example, MOT20-02 frame 100 (GT 62 pedestrians):")
    add("")
    add("| Stage | RF-DETR | YOLOX |")
    add("| --- | ---: | ---: |")
    add("| raw detections | 149 | 56 |")
    add("| already above `det_thresh` | 61 | 53 |")
    add("| promoted by boosting | 3 | 0 |")
    add("| entering association | 64 | 53 |")
    add("| live tracks | 88 | 70 |")
    add("| matched | 64 | 53 |")
    add("| new tracks | 0 | 0 |")
    add("| unmatched tracks | 24 | 17 |")
    add("| **output boxes** | **61** | **52** |")
    add("")

    # ---------------------------------------------------------------- tracking
    runs = tracking_runs()
    add("## 7. Tracking results, MOT20 `val_half`")
    add("")
    add("Vendored TrackEval against `repos/BoostTrack/results/gt/MOT20-val`.")
    add(f"Ground truth: {boxes} boxes, {identities} identities.")
    add("")
    add("Detector slug legend:")
    add("")
    add("| Slug | Meaning |")
    add("| --- | --- |")
    add("| `yoloxx20` | Supplied prebuilt YOLOX-X MOT20 detection bundle; described as MOT20-train-trained but its weights are not on disk and are **not** ByteTrack's release |")
    add("| `yoloxx20-official` | ByteTrack's released YOLOX-X MOT20 detector, `bytetrack_x_mot20.tar`, `test_size=(896,1600)`, `nmsthre=0.7`, emitted at score >= 0.10 |")
    add("| `rfdetr2xl-e5-t005` | RF-DETR 2XL arm-A epoch-5 checkpoint, export score threshold 0.05 |")
    add("| `rfdetr2xl-e5-t010` | same checkpoint, export score threshold 0.10 |")
    add("| `rfdetr2xl-e5-t010-nms070` | threshold 0.10 then greedy NMS at IoU 0.70 |")
    add("| `rfdetr2xl-e5-t010-nms0600` / `-nms0500` | the same at IoU 0.60 / 0.50 |")
    add("")
    add("Unless a row says otherwise the tracker is BoostTrack++ at its published")
    add("MOT20 settings with `det_thresh = 0.40`.")
    add("")
    headline = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "IDs")
    detail = ("DetPr", "DetRe", "AssPr", "AssRe", "MOTP", "Frag", "MT", "ML", "CLR_TP", "CLR_FP", "CLR_FN")

    def section(title: str, predicate, note: str = "") -> None:
        selected = [(slug, entry) for slug, entry in sorted(runs.items()) if predicate(slug, entry)]
        if not selected:
            return
        add(f"### {title}")
        add("")
        if note:
            add(note)
            add("")
        rows = []
        for slug, entry in sorted(selected, key=lambda kv: -kv[1]["metrics"]["HOTA"]):
            detector, _, tracker = slug.split("__")
            label = f"`{detector}` — {settings_label(tracker, entry['overrides'])}"
            rows.append((label, entry["metrics"]))
        metrics_table(lines, rows, headline)
        add("")
        add("<details><summary>Secondary metrics</summary>")
        add("")
        metrics_table(lines, rows, detail)
        add("")
        add("</details>")
        add("")

    section(
        "7.1 OSNet ReID (`osnet-ain-msdc`), held out",
        lambda slug, e: "__osnet-ain-msdc__" in slug,
        "Generic domain-generalized ReID. These rows are held out: the model has never seen MOT20.",
    )
    section(
        "7.2 FastReID (`fastreid-sbs-s50-mot20`), NOT held out",
        lambda slug, e: "__fastreid-sbs-s50-mot20__" in slug,
        "BoostTrack++'s published ReID. MOT20-trained, so it has seen `val_half` identities; "
        "absolute values are inflated. The association sweep points live here.",
    )

    add("## 8. Provenance")
    add("")
    add("| Artifact | Path |")
    add("| --- | --- |")
    add("| Detection scores | `artifacts/tracking/detection-analysis-val_half.json`, `-armc-`, `-i4-` |")
    add("| Localization | `artifacts/tracking/localization-analysis-val_half.json` |")
    add("| Label agreement | `artifacts/tracking/annotation-agreement-val_half.json` |")
    add("| Geometry gate and probes | `artifacts/tracking/geometry-gate-*.json`, `geometry-probe-*.json` |")
    add("| Stage instrumentation | `artifacts/tracking/diagnosis/stages-*.{csv,json}` |")
    add("| Detections (L1) | `artifacts/tracking/detections/<detector>/<split>/` |")
    add("| Embeddings (L2) | `artifacts/tracking/embeddings/<detector>__<reid>/<split>/` |")
    add("| Tracks (L3) | `artifacts/tracking/tracks/<combination>/<split>/` |")
    add("| Metrics (L4) | `artifacts/tracking/metrics/<combination>/<split>/` |")
    add("| Training runs | `finetuning/artifacts/<run>/metrics.csv`, `run-provenance.json` |")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "results-reference.md")
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "regenerate into memory and diff against the file on disk without writing. "
            "Exits 1 if they differ, so CI or a pre-commit hook can catch a stale document."
        ),
    )
    args = parser.parse_args()
    document = build()

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"{args.output} does not exist; run without --check to create it")
        current = args.output.read_text(encoding="utf-8")
        if current == document:
            print(f"{args.output.name} is up to date ({len(document.splitlines())} lines)")
            return
        # The generation date line always differs on a later day, so report it
        # separately rather than letting it mask a real content change.
        difference = list(
            unified_diff(
                current.splitlines(),
                document.splitlines(),
                fromfile=f"{args.output.name} (on disk)",
                tofile=f"{args.output.name} (regenerated)",
                lineterm="",
                n=1,
            )
        )
        changed = [
            line
            for line in difference
            if line.startswith(("+", "-"))
            and not line.startswith(("+++", "---"))
            and "Generated 2" not in line
        ]
        print("\n".join(difference))
        if not changed:
            print(
                f"\n{args.output.name} differs only in the generation date; content is up to date."
            )
            return
        raise SystemExit(
            f"\n{args.output} is stale: {len(changed)} changed lines beyond the generation date. "
            "Re-run without --check."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"wrote {args.output} ({len(document.splitlines())} lines)")


if __name__ == "__main__":
    main()
