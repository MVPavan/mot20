#!/usr/bin/env python3
"""Build immutable COCO manifests for ByteTrack-style MOT20/CrowdHuman training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mot20.detection.coco_conversion import (
    assemble_rfdetr_coco_dataset,
    convert_crowdhuman_split,
    convert_mot20_split,
    merge_bytetrack_mot20_crowdhuman,
    write_coco_manifest,
)
from mot20.detection.dataset_audit import audit_rfdetr_coco_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot20-root", type=Path, required=True)
    parser.add_argument("--crowdhuman-train-annotations", type=Path, required=True)
    parser.add_argument("--crowdhuman-train-images", type=Path, required=True)
    parser.add_argument("--crowdhuman-val-annotations", type=Path, required=True)
    parser.add_argument("--crowdhuman-val-images", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="A new RF-DETR COCO dataset root; existing directories are refused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest directory: {output_dir}")

    mot20_train_half = convert_mot20_split(args.mot20_root, "train_half")
    mot20_val_half = convert_mot20_split(args.mot20_root, "val_half")
    crowdhuman_train = convert_crowdhuman_split(
        args.crowdhuman_train_annotations,
        args.crowdhuman_train_images,
        "train",
    )
    crowdhuman_val = convert_crowdhuman_split(
        args.crowdhuman_val_annotations,
        args.crowdhuman_val_images,
        "val",
    )
    train_manifest = merge_bytetrack_mot20_crowdhuman(
        mot20_train_half,
        crowdhuman_train,
        crowdhuman_val,
    )

    assemble_rfdetr_coco_dataset(
        output_dir,
        train_manifest,
        mot20_val_half,
        args.mot20_root,
        args.crowdhuman_train_images,
        args.crowdhuman_val_images,
    )
    source_manifests = {
        "mot20_train_half.json": mot20_train_half,
        "mot20_val_half.json": mot20_val_half,
        "crowdhuman_train.json": crowdhuman_train,
        "crowdhuman_val.json": crowdhuman_val,
    }
    checksums = {
        name: write_coco_manifest(manifest, output_dir / "source-manifests" / name)
        for name, manifest in source_manifests.items()
    }
    audit = audit_rfdetr_coco_dataset(output_dir)
    audit["source_input_sha256"] = _source_input_hashes(
        args.mot20_root,
        args.crowdhuman_train_annotations,
        args.crowdhuman_val_annotations,
    )
    audit["source_manifest_sha256"] = checksums
    write_coco_manifest(audit, output_dir / "audit.json")
    summary_path = output_dir / "checksums.json"
    write_coco_manifest({"source_manifest_sha256": checksums}, summary_path)
    print(json.dumps({"audit_path": str(output_dir / "audit.json"), "output_dir": str(output_dir), "sha256": checksums}, indent=2, sort_keys=True))


def _source_input_hashes(
    mot20_root: Path,
    crowdhuman_train_annotations: Path,
    crowdhuman_val_annotations: Path,
) -> dict[str, object]:
    mot20_sequences = sorted(path for path in mot20_root.iterdir() if path.is_dir())
    return {
        "crowdhuman_train_annotations": _sha256_file(crowdhuman_train_annotations),
        "crowdhuman_val_annotations": _sha256_file(crowdhuman_val_annotations),
        "mot20_gt": {
            sequence.name: _sha256_file(sequence / "gt" / "gt.txt")
            for sequence in mot20_sequences
        },
        "mot20_seqinfo": {
            sequence.name: _sha256_file(sequence / "seqinfo.ini")
            for sequence in mot20_sequences
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()