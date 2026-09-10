#!/usr/bin/env python3
"""Run ByteTrack's YOLOX-X MOT20 detector over a MOT20 split.

The `det_yoloxx20` detections shipped in `datasets/val_half/` were supplied as a
prebuilt bundle and only ever covered `val_half` and the test sequences. Scoring
any other split against that baseline requires re-running the detector, which is
what this does.

PROTOCOL. Reproduces `repos/ByteTrack/exps/example/mot/yolox_x_mix_mot20_ch.py`
exactly: YOLOX-X (depth 1.33, width 1.25), one class, `test_size = (896, 1600)`,
`nmsthre = 0.7`, letterbox preprocessing with ImageNet mean/std, and
`yolox.utils.postprocess`. Boxes are divided by the letterbox scale to return to
original image pixels, matching `tools/track.py`.

SCORE THRESHOLD. The supplied bundle's minimum score is 0.1001, so it was cut at
0.10 rather than at the experiment's `test_conf = 0.001`. `--min-score` defaults
to 0.10 to match it. Running NMS at a lower confidence and filtering afterwards
is equivalent for the retained set, because a lower-scoring box can never
suppress a higher-scoring one in `batched_nms`; so `--nms-conf` stays at the
experiment value and only the emitted set is cut.

OUTPUT. The repo's nine-column L1 detection rows,
`frame,-1,left,top,width,height,score,0,1.0`, byte-compatible with the supplied
files: boxes to two decimals, scores to four.

Verify against the supplied bundle before trusting a new split:

    tracking/scripts/infer_yoloxx20.py --split val_half --output-root artifacts/tmp/yolox-check
    tracking/scripts/compare_yoloxx20.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "repos" / "ByteTrack"))
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mot20_tracking.sequences import read_split  # noqa: E402
from yolox.data.data_augment import preproc  # noqa: E402
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

# Values are the experiment file's, not defaults; changing one changes the detector.
TEST_SIZE = (896, 1600)
NMS_THRESHOLD = 0.7
NMS_CONF = 0.001
NUM_CLASSES = 1
DEPTH = 1.33
WIDTH = 1.25
RGB_MEANS = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "val_half", "test"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/bytetrack_x_mot20.tar"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--nms-conf", type=float, default=NMS_CONF)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--sequences", nargs="*", default=None, help="subset by name")
    return parser.parse_args()


def build_model(checkpoint: Path, device: str, fp16: bool) -> torch.nn.Module:
    """Construct YOLOX-X at the experiment's depth/width and load the checkpoint."""
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(DEPTH, WIDTH, in_channels=in_channels)
    head = YOLOXHead(NUM_CLASSES, WIDTH, in_channels=in_channels)
    model = YOLOX(backbone, head)
    # ByteTrack ships a torch.save of {"model": state_dict, ...}; weights_only is
    # safe here and refuses anything that tries to execute on load.
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint does not match the YOLOX-X MOT20 architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys"
        )
    model.eval().to(device)
    if fp16:
        model = model.half()
    return model


@torch.no_grad()
def detect_frame(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str,
    fp16: bool,
    nms_conf: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Return (N, 5) rows of x1, y1, x2, y2, score in original image pixels.

    Boxes are clipped to the image rectangle and degenerate ones dropped. YOLOX
    regresses past the frame edge for people cut off by it, and every consumer
    here clips anyway: BoostTrack clips in `tracker/embedding.py`, the COCO
    conversion clips ground truth in `_clip_box`, and the L1 validator rejects
    out-of-image boxes outright. The supplied `det_yoloxx20` bundle is also
    fully in bounds, so clipping matches it rather than diverging from it.
    """
    tensor, ratio = preproc(image, TEST_SIZE, RGB_MEANS, RGB_STD)
    batch = torch.from_numpy(tensor).unsqueeze(0).to(device)
    batch = batch.half() if fp16 else batch.float()
    outputs = postprocess(model(batch), NUM_CLASSES, nms_conf, NMS_THRESHOLD)
    if outputs[0] is None:
        return np.zeros((0, 5), dtype=np.float32)
    detections = outputs[0].float().cpu().numpy()
    boxes = detections[:, :4] / ratio
    boxes[:, 0] = boxes[:, 0].clip(0, width)
    boxes[:, 1] = boxes[:, 1].clip(0, height)
    boxes[:, 2] = boxes[:, 2].clip(0, width)
    boxes[:, 3] = boxes[:, 3].clip(0, height)
    # postprocess emits obj_conf and class_conf separately; ByteTrack scores
    # detections with their product.
    scores = detections[:, 4] * detections[:, 5]
    keep = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    return np.concatenate([boxes[keep], scores[keep, None]], axis=1)


def main() -> None:
    args = parse_args()
    checkpoint = (
        REPO_ROOT / args.checkpoint if not args.checkpoint.is_absolute() else args.checkpoint
    )
    output_root = (
        REPO_ROOT / args.output_root if not args.output_root.is_absolute() else args.output_root
    )
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    sequences = read_split(args.split, REPO_ROOT)
    if args.sequences:
        wanted = set(args.sequences)
        sequences = [s for s in sequences if s.name in wanted]
        if not sequences:
            raise SystemExit(f"no sequences matched {sorted(wanted)}")

    model = build_model(checkpoint, args.device, args.fp16)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    for sequence in sequences:
        destination = output_root / sequence.name / "det_yoloxx20"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "det_yoloxx20.txt"
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing detections: {target}")

        started = time.time()
        kept = 0
        emitted_scores: list[float] = []
        with target.open("w", encoding="utf-8") as stream:
            for frame_id in range(1, sequence.length + 1):
                path = sequence.frame_path(frame_id)
                image = cv2.imread(str(path))
                if image is None:
                    raise SystemExit(f"unreadable frame: {path}")
                rows = detect_frame(
                    model,
                    image,
                    args.device,
                    args.fp16,
                    args.nms_conf,
                    sequence.width,
                    sequence.height,
                )
                for x1, y1, x2, y2, score in rows:
                    if score < args.min_score:
                        continue
                    kept += 1
                    emitted_scores.append(float(score))
                    stream.write(
                        f"{frame_id},-1,{x1:.2f},{y1:.2f},"
                        f"{x2 - x1:.2f},{y2 - y1:.2f},{score:.4f},0,1.0\n"
                    )
                if frame_id % 200 == 0:
                    elapsed = time.time() - started
                    print(
                        f"{sequence.name} {frame_id}/{sequence.length} "
                        f"{frame_id / elapsed:.1f} fps",
                        flush=True,
                    )
        summary[sequence.name] = {
            "frames": sequence.length,
            "detections": kept,
            "mean_per_frame": round(kept / sequence.length, 3),
            "min_score": round(min(emitted_scores), 4) if emitted_scores else None,
            "max_score": round(max(emitted_scores), 4) if emitted_scores else None,
            "seconds": round(time.time() - started, 1),
            "path": str(target.relative_to(REPO_ROOT)),
        }
        print(json.dumps({sequence.name: summary[sequence.name]}), flush=True)

    print(
        json.dumps(
            {
                "split": args.split,
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "test_size": list(TEST_SIZE),
                "nms_threshold": NMS_THRESHOLD,
                "nms_conf": args.nms_conf,
                "min_score": args.min_score,
                "fp16": args.fp16,
                "sequences": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
