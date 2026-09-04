"""Helpers for bounded RF-DETR capacity probes."""

from __future__ import annotations

import torch

from rfdetr.utilities.tensors import NestedTensor


def pad_batch_to_envelope(samples: NestedTensor, height: int, width: int) -> NestedTensor:
    """Extend a collated batch's padded region without changing image content."""
    if height < 1 or width < 1:
        raise ValueError("capacity-probe envelope dimensions must be positive")
    tensors, mask = samples.decompose()
    if mask is None:
        raise ValueError("capacity probe requires a collated padding mask")
    if tensors.ndim != 4 or mask.shape != (tensors.shape[0], tensors.shape[2], tensors.shape[3]):
        raise ValueError("capacity probe received an invalid collated batch")
    current_height, current_width = tensors.shape[-2:]
    if current_height > height or current_width > width:
        raise ValueError(
            f"capacity-probe envelope {height}x{width} cannot contain collated batch {current_height}x{current_width}"
        )
    tensor_padding = (0, width - current_width, 0, height - current_height)
    return NestedTensor(
        torch.nn.functional.pad(tensors, tensor_padding),
        torch.nn.functional.pad(mask, tensor_padding, value=True),
    )