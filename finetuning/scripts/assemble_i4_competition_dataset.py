#!/usr/bin/env python3
"""Assemble I4's immutable full-MOT20 local test-adapted competition mix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mot20.detection.coco_conversion import (
    assemble_rfdetr_competition_dataset,
    build_i4_competition_train_manifest,
    write_coco_manifest,
)
from mot20.detection.dataset_audit import audit_rfdetr_coco_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-dataset-dir",
        type=Path,
        required=True,
        help="Existing immutable I4 ablation RF-DETR dataset root with the four source manifests.",
    )
    parser.add_argument(
        "--byte65-root",
        type=Path,
        required=True,
        help="Existing immutable materialized Byte65 dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New I4 competition RF-DETR dataset root; existing directories are refused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_dataset_dir = args.ablation_dataset_dir
    byte65_root = args.byte65_root
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing I4 competition dataset: {output_dir}")
    if not ablation_dataset_dir.is_dir():
        raise ValueError(f"I4 ablation dataset root is not a directory: {ablation_dataset_dir}")
    if not byte65_root.is_dir():
        raise ValueError(f"Byte65 root is not a directory: {byte65_root}")

    source_manifest_dir = ablation_dataset_dir / "source-manifests"
    mot20_train = _read_manifest(source_manifest_dir / "mot20_train_half.json")
    mot20_val = _read_manifest(source_manifest_dir / "mot20_val_half.json")
    crowdhuman_train = _read_manifest(source_manifest_dir / "crowdhuman_train.json")
    crowdhuman_val = _read_manifest(source_manifest_dir / "crowdhuman_val.json")
    byte65 = _read_manifest(byte65_root / "annotations.coco.json")
    byte65["metadata"] = {
        **byte65.get("metadata", {}),
        "human_audit": "exhaustive",
        "human_audit_authority": "user_confirmed",
    }
    train_manifest = build_i4_competition_train_manifest(
        mot20_train,
        mot20_val,
        crowdhuman_train,
        crowdhuman_val,
        byte65,
    )
    train_manifest["metadata"] = {
        **train_manifest["metadata"],
        "byte65_dataset_name": byte65["metadata"].get("dataset_name"),
        "byte65_label_source": byte65["metadata"].get("label_source"),
        "byte65_source_archive_sha256": byte65["metadata"].get("source_archive_sha256"),
        "byte65_test_sequences": sorted({image["source_sequence"] for image in byte65["images"]}),
    }

    assemble_rfdetr_competition_dataset(
        output_dir,
        train_manifest,
        {
            "mot20_train": _linked_directory(ablation_dataset_dir / "train" / "mot20_train"),
            "crowdhuman_train": _linked_directory(ablation_dataset_dir / "train" / "crowdhuman_train"),
            "crowdhuman_val": _linked_directory(ablation_dataset_dir / "train" / "crowdhuman_val"),
            "byte65": byte65_root.resolve(),
        },
    )
    source_manifests = {
        "mot20_train_half.json": mot20_train,
        "mot20_val_half.json": mot20_val,
        "crowdhuman_train.json": crowdhuman_train,
        "crowdhuman_val.json": crowdhuman_val,
        "byte65_human_audited.json": byte65,
    }
    checksums = {
        name: write_coco_manifest(manifest, output_dir / "source-manifests" / name)
        for name, manifest in source_manifests.items()
    }
    audit = audit_rfdetr_coco_dataset(output_dir)
    audit["classification"] = "local_test_adapted"
    audit["held_out_benchmark_comparable"] = False
    audit["source_manifest_sha256"] = checksums
    audit["byte65_human_audit_authority"] = "user_confirmed"
    write_coco_manifest(audit, output_dir / "audit.json")
    write_coco_manifest({"source_manifest_sha256": checksums}, output_dir / "checksums.json")
    print(
        json.dumps(
            {
                "audit_path": str(output_dir / "audit.json"),
                "output_dir": str(output_dir),
                "sha256": checksums,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"source manifest is not a file: {path}")
    with path.open(encoding="utf-8") as stream:
        content = json.load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"source manifest is not an object: {path}")
    return content


def _linked_directory(path: Path) -> Path:
    if not path.is_dir():
        raise ValueError(f"linked ablation image root is not a directory: {path}")
    return path.resolve()


if __name__ == "__main__":
    main()
