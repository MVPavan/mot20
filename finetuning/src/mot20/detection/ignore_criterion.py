"""RF-DETR criterion extension for ignored pedestrian regions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.math import accuracy
from rfdetr.utilities import box_ops

from mot20.detection.ignore_loss import ignore_aware_sigmoid_focal_loss, ignored_unmatched_query_mask


class IgnoreAwareSetCriterion(SetCriterion):
    """Exclude ignored-region overlaps from RF-DETR classification supervision.

    Targets may include normalized ``ignored_boxes`` in ``cx, cy, width,
    height`` format. Those boxes are not matcher targets; only unmatched
    predictions whose IoU reaches ``ignored_iou_threshold`` lose classification
    supervision.
    """

    def __init__(self, *args: Any, ignored_iou_threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= ignored_iou_threshold <= 1:
            raise ValueError(f"ignored_iou_threshold must be in [0, 1], got {ignored_iou_threshold}")
        self.ignored_iou_threshold = ignored_iou_threshold

    def loss_labels(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
        log: bool = True,
    ) -> dict[str, Tensor]:
        """Compute classification loss while excluding ignored-region overlaps."""
        if not any(target.get("ignored_boxes", torch.empty(0)).numel() for target in targets):
            return super().loss_labels(outputs, targets, indices, num_boxes, log=log)
        if self.use_varifocal_loss or self.use_position_supervised_loss:
            raise ValueError(
                "IgnoreAwareSetCriterion supports ignored_boxes only with IA-BCE or sigmoid focal loss; "
                "disable varifocal and position-supervised loss or implement their masked reductions."
            )

        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"]
        ignored_boxes = [
            target.get("ignored_boxes", src_boxes.new_empty((0, 4))).to(device=src_boxes.device, dtype=src_boxes.dtype)
            for target in targets
        ]
        query_mask = ignored_unmatched_query_mask(
            src_boxes,
            ignored_boxes,
            indices,
            self.ignored_iou_threshold,
        )
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes_o = torch.cat([target["labels"][target_indices] for target, (_, target_indices) in zip(targets, indices)])
        target_classes[idx] = target_classes_o

        if self.ia_bce_loss:
            loss_ce = self._ignore_aware_ia_bce_loss(
                src_logits,
                src_boxes,
                targets,
                indices,
                idx,
                target_classes_o,
                query_mask,
                num_boxes,
            )
        else:
            target_classes_onehot = torch.zeros(
                [*src_logits.shape[:2], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            loss_ce = ignore_aware_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot[:, :, :-1],
                num_boxes,
                src_boxes,
                ignored_boxes,
                indices,
                self.ignored_iou_threshold,
                alpha=self.focal_alpha,
                gamma=2,
            )

        losses = {"loss_ce": loss_ce}
        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _ignore_aware_ia_bce_loss(
        self,
        src_logits: Tensor,
        src_boxes: Tensor,
        targets: list[dict[str, Tensor]],
        matching_indices: list[tuple[Tensor, Tensor]],
        source_indices: tuple[Tensor, Tensor],
        target_classes_o: Tensor,
        query_mask: Tensor,
        num_boxes: Tensor,
    ) -> Tensor:
        target_boxes = torch.cat(
            [target["boxes"][target_indices] for target, (_, target_indices) in zip(targets, matching_indices)],
            dim=0,
        )
        iou_targets, _ = box_ops.elementwise_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[source_indices].detach()),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )
        probabilities = src_logits.sigmoid()
        positive_weights = torch.zeros_like(src_logits)
        negative_weights = probabilities.pow(2)
        positive_indices = [*source_indices, target_classes_o]
        target_weight = probabilities[tuple(positive_indices)].pow(self.focal_alpha) * iou_targets.detach().pow(1 - self.focal_alpha)
        target_weight = torch.clamp(target_weight, 0.01).detach()
        positive_weights[tuple(positive_indices)] = target_weight.to(positive_weights.dtype)
        negative_weights[tuple(positive_indices)] = 1 - target_weight.to(negative_weights.dtype)
        loss = negative_weights * src_logits - functional.logsigmoid(src_logits) * (positive_weights + negative_weights)
        return (loss * (~query_mask).unsqueeze(-1)).sum() / num_boxes