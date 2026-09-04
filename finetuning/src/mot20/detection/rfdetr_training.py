"""Versioned configuration validation for RF-DETR detector runs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = ("run", "model", "capacity", "training")


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Read the versioned TOML configuration required by the run launcher."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"training configuration is not a file: {config_path}")
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid training TOML: {config_path}") from error
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"training configuration requires [{section}]")
    return config


def validate_training_config(config: dict[str, Any], audit: dict[str, Any] | None = None) -> None:
    """Reject runs that lack a reviewed status or audited query capacity."""
    if config["run"].get("status") != "approved":
        raise ValueError("training configuration is not approved for training")
    classification = config["run"].get("classification")
    if classification not in ("clean_held_out_validation", "local_test_adapted"):
        raise ValueError("run classification must be clean_held_out_validation or local_test_adapted")
    model = config["model"]
    capacity = config["capacity"]
    training = config["training"]
    if model.get("name") != "RFDETR2XLarge" or model.get("num_classes") != 1:
        raise ValueError("training configuration must select one-class RFDETR2XLarge")
    resolution = _positive_int(model, "resolution")
    if resolution % 40:
        raise ValueError("RF-DETR 2XL resolution must be divisible by 40")
    group_detr = _positive_int(capacity, "group_detr")
    if group_detr != 13:
        raise ValueError("group_detr must remain 13 until a reviewed change is recorded")
    num_queries = _positive_int(capacity, "num_queries")
    num_select = _positive_int(capacity, "num_select")
    eval_max_dets = _positive_int(capacity, "eval_max_dets")
    if num_queries % group_detr:
        raise ValueError("num_queries must be divisible by group_detr")
    if audit is None:
        return
    audit_classification = audit.get("classification", "clean_held_out_validation")
    if audit_classification != classification:
        raise ValueError("run classification does not match the audited dataset classification")
    maximum_labels = audit.get("query_capacity", {}).get("maximum_loss_participating_labels")
    if not isinstance(maximum_labels, int) or maximum_labels < 0:
        raise ValueError("dataset audit lacks a valid maximum loss-participating label count")
    if num_queries <= maximum_labels:
        raise ValueError("num_queries must be greater than the observed maximum loss-participating labels")
    if num_select != num_queries:
        raise ValueError("num_select must equal num_queries for high-density evaluation")
    if eval_max_dets < num_queries:
        raise ValueError("eval_max_dets must be at least num_queries")
    batch_size = training.get("batch_size")
    if batch_size != "auto":
        _positive_int(training, "batch_size")
    for key in ("grad_accum_steps", "epochs"):
        _positive_int(training, key)
    if "seed" in training:
        _nonnegative_int(training, "seed")
    if training.get("amp_dtype") not in ("bf16", "fp16", "auto"):
        raise ValueError("amp_dtype must be bf16, fp16, or auto")


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value