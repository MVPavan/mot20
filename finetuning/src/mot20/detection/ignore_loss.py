"""Loss masking for predictions overlapping ignored pedestrian annotations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor


def ignore_aware_sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    num_boxes: Tensor,
    predicted_boxes: Tensor,
    ignored_boxes: Sequence[Tensor],
    matched_indices: Sequence[tuple[Tensor, Tensor]],
    iou_threshold: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Compute focal loss excluding unmatched queries overlapping ignored boxes.

    Boxes use normalized ``cx, cy, width, height`` coordinates. Ignored boxes
    are never matcher targets; a matched query remains supervised even when it
    also overlaps an ignored box.
    """
    _validate_inputs(logits, targets, predicted_boxes, ignored_boxes, matched_indices, iou_threshold)

    query_mask = ignored_unmatched_query_mask(
        predicted_boxes,
        ignored_boxes,
        matched_indices,
        iou_threshold,
    )
    probabilities = logits.sigmoid()
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_of_target = probabilities * targets + (1 - probabilities) * (1 - targets)
    loss = cross_entropy * (1 - probability_of_target).pow(gamma)

    if alpha >= 0:
        alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_weight * loss

    retained_queries = (~query_mask).sum(dim=1).clamp(min=1).to(dtype=loss.dtype)
    return (loss * (~query_mask).unsqueeze(-1)).sum(dim=1).div(retained_queries.unsqueeze(-1)).sum() / num_boxes


def ignored_unmatched_query_mask(
    predicted_boxes: Tensor,
    ignored_boxes: Sequence[Tensor],
    matched_indices: Sequence[tuple[Tensor, Tensor]],
    iou_threshold: float,
) -> Tensor:
    """Return a per-image/query mask for ignored, unmatched predictions."""
    batch_size, query_count, _ = predicted_boxes.shape
    ignored_mask = torch.zeros((batch_size, query_count), dtype=torch.bool, device=predicted_boxes.device)
    predicted_xyxy = _cxcywh_to_xyxy(predicted_boxes)

    for image_index, image_ignored_boxes in enumerate(ignored_boxes):
        if image_ignored_boxes.numel() == 0:
            continue
        ignored_xyxy = _cxcywh_to_xyxy(image_ignored_boxes.to(predicted_boxes))
        overlap = _pairwise_iou(predicted_xyxy[image_index], ignored_xyxy) >= iou_threshold
        ignored_mask[image_index] = overlap.any(dim=1)
        matched_queries = matched_indices[image_index][0].to(device=predicted_boxes.device, dtype=torch.long)
        ignored_mask[image_index, matched_queries] = False

    return ignored_mask


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    centers, dimensions = boxes[..., :2], boxes[..., 2:].clamp(min=0)
    return torch.cat((centers - dimensions / 2, centers + dimensions / 2), dim=-1)


def _pairwise_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp(min=0).prod(dim=-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp(min=0).prod(dim=-1)
    return intersection / (area_a[:, None] + area_b - intersection).clamp(min=1e-7)


def _validate_inputs(
    logits: Tensor,
    targets: Tensor,
    predicted_boxes: Tensor,
    ignored_boxes: Sequence[Tensor],
    matched_indices: Sequence[tuple[Tensor, Tensor]],
    iou_threshold: float,
) -> None:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}")
    if logits.ndim != 3 or predicted_boxes.shape != (*logits.shape[:2], 4):
        raise ValueError("expected logits [batch, queries, classes] and predicted_boxes [batch, queries, 4]")
    if len(ignored_boxes) != logits.shape[0] or len(matched_indices) != logits.shape[0]:
        raise ValueError("ignored_boxes and matched_indices must each contain one entry per batch image")
    if not 0 <= iou_threshold <= 1:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
    for image_ignored_boxes in ignored_boxes:
        if image_ignored_boxes.ndim != 2 or image_ignored_boxes.shape[1:] != (4,):
            raise ValueError("each ignored_boxes tensor must have shape [count, 4]")