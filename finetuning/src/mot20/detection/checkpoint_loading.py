"""Explicit checkpoint adaptation for reviewed RF-DETR query-capacity expansion."""

from __future__ import annotations

from typing import Mapping

from torch import Tensor


QUERY_PARAMETER_NAMES = ("refpoint_embed.weight", "query_feat.weight")
CLASSIFICATION_HEAD_MARKERS = ("class_embed.", "enc_out_class_embed.")


def expand_query_parameters(
    checkpoint_parameters: Mapping[str, Tensor],
    initialized_parameters: Mapping[str, Tensor],
    checkpoint_num_queries: int,
    target_num_queries: int,
    group_detr: int,
) -> dict[str, Tensor]:
    """Copy pretrained per-group queries into a larger initialized query table."""
    if checkpoint_num_queries < 1 or target_num_queries < checkpoint_num_queries or group_detr < 1:
        raise ValueError("query expansion requires positive groups and a non-decreasing query count")
    expanded: dict[str, Tensor] = {}
    for name in QUERY_PARAMETER_NAMES:
        checkpoint = checkpoint_parameters.get(name)
        initialized = initialized_parameters.get(name)
        if checkpoint is None or initialized is None:
            raise ValueError(f"missing query parameter: {name}")
        if checkpoint.ndim != initialized.ndim or checkpoint.shape[1:] != initialized.shape[1:]:
            raise ValueError(f"incompatible query parameter shape: {name}")
        if checkpoint.shape[0] != checkpoint_num_queries * group_detr:
            raise ValueError(f"checkpoint query parameter has unexpected row count: {name}")
        if initialized.shape[0] != target_num_queries * group_detr:
            raise ValueError(f"initialized query parameter has unexpected row count: {name}")
        target = initialized.clone()
        for group_index in range(group_detr):
            source_start = group_index * checkpoint_num_queries
            target_start = group_index * target_num_queries
            target[target_start : target_start + checkpoint_num_queries] = checkpoint[
                source_start : source_start + checkpoint_num_queries
            ]
        expanded[name] = target
    return expanded


def prepare_expanded_checkpoint_state(
    checkpoint_state: Mapping[str, Tensor],
    initialized_state: Mapping[str, Tensor],
    checkpoint_num_queries: int,
    target_num_queries: int,
    group_detr: int,
) -> dict[str, Tensor]:
    """Prepare a strict one-class state dict with expanded per-group queries."""
    expanded_queries = expand_query_parameters(
        checkpoint_state,
        initialized_state,
        checkpoint_num_queries,
        target_num_queries,
        group_detr,
    )
    prepared: dict[str, Tensor] = {}
    for name, checkpoint_value in checkpoint_state.items():
        initialized_value = initialized_state.get(name)
        if name in expanded_queries:
            prepared[name] = expanded_queries[name]
        elif initialized_value is None:
            raise ValueError(f"unexpected checkpoint parameter: {name}")
        elif checkpoint_value.shape == initialized_value.shape:
            prepared[name] = checkpoint_value
        elif any(marker in name for marker in CLASSIFICATION_HEAD_MARKERS):
            continue
        else:
            raise ValueError(f"unapproved checkpoint shape mismatch: {name}")
    return prepared