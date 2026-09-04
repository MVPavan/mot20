"""Scoped RF-DETR integration for ignored pedestrian annotations."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch

from mot20.detection.ignore_criterion import IgnoreAwareSetCriterion
from mot20.detection.ignore_dataset import IgnoreAwareCocoDetection


@contextmanager
def use_ignore_aware_rfdetr(ignored_iou_threshold: float = 0.5) -> Generator[None, None, None]:
    """Temporarily install ignore-aware COCO and criterion factories for RF-DETR.

    The context supports box-only COCO/Roboflow datasets and RF-DETR 1.9.4's
    default IA-BCE or sigmoid-focal classification modes. Keep the context open
    for the entire ``RFDETR.train()`` call so delayed dataset construction uses
    the project-owned components.
    """
    import rfdetr.datasets.coco as coco_module
    import rfdetr.training.module_model as module_model

    stock_dataset = coco_module.CocoDetection
    stock_criterion_factory = module_model.build_criterion_from_config

    def build_ignore_aware_criterion(model_config: Any, train_config: Any) -> tuple[IgnoreAwareSetCriterion, Any]:
        criterion, postprocess = stock_criterion_factory(model_config, train_config)
        if model_config.segmentation_head or model_config.use_grouppose_keypoints:
            raise ValueError("Ignored-region loss masking supports box-only RF-DETR detection models")
        ignore_aware_criterion = IgnoreAwareSetCriterion(
            num_classes=criterion.num_classes,
            matcher=criterion.matcher,
            weight_dict=criterion.weight_dict,
            focal_alpha=criterion.focal_alpha,
            losses=criterion.losses,
            group_detr=criterion.group_detr,
            sum_group_losses=criterion.sum_group_losses,
            use_varifocal_loss=criterion.use_varifocal_loss,
            use_position_supervised_loss=criterion.use_position_supervised_loss,
            ia_bce_loss=criterion.ia_bce_loss,
            mask_point_sample_ratio=criterion.mask_point_sample_ratio,
            num_keypoints_per_class=criterion.num_keypoints_per_class,
            ignored_iou_threshold=ignored_iou_threshold,
        )
        ignore_aware_criterion.to(torch.device(model_config.device))
        return ignore_aware_criterion, postprocess

    coco_module.CocoDetection = IgnoreAwareCocoDetection
    module_model.build_criterion_from_config = build_ignore_aware_criterion
    try:
        yield
    finally:
        coco_module.CocoDetection = stock_dataset
        module_model.build_criterion_from_config = stock_criterion_factory