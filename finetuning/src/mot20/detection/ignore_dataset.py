"""RF-DETR COCO adapters that preserve ignored boxes for loss masking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from rfdetr.datasets.coco import CocoDetection, ConvertCoco


class IgnoreAwareConvertCoco(ConvertCoco):
    """Convert positive COCO targets and preserve crowd boxes separately."""

    def __call__(self, image: Image.Image, target: dict[str, Any]) -> tuple[Image.Image, dict[str, Tensor]]:
        image_width, image_height = image.size
        ignored_boxes = _clipped_crowd_boxes(target["annotations"], image_width, image_height)
        image, converted_target = super().__call__(image, target)
        converted_target["ignored_boxes"] = ignored_boxes
        return image, converted_target


class IgnoreAwareTransform:
    """Apply an RF-DETR transform to positive and ignored boxes together."""

    def __init__(self, transform: Callable[[Image.Image, dict[str, Tensor]], tuple[Image.Image, dict[str, Tensor]]]) -> None:
        self._transform = transform

    def __call__(self, image: Image.Image, target: dict[str, Tensor]) -> tuple[Image.Image, dict[str, Tensor]]:
        target = dict(target)
        ignored_boxes = target.pop("ignored_boxes")
        if ignored_boxes.numel() == 0:
            image, transformed = self._transform(image, target)
            transformed["ignored_boxes"] = ignored_boxes
            return image, transformed

        combined_target = dict(target)
        combined_target["boxes"] = torch.cat((target["boxes"], ignored_boxes))
        combined_target["labels"] = torch.cat(
            (target["labels"], torch.full((ignored_boxes.shape[0],), -1, dtype=target["labels"].dtype))
        )
        if "area" in target:
            ignored_areas = _box_areas(ignored_boxes).to(dtype=target["area"].dtype)
            combined_target["area"] = torch.cat((target["area"], ignored_areas))
        if "iscrowd" in target:
            combined_target["iscrowd"] = torch.cat(
                (target["iscrowd"], torch.ones(ignored_boxes.shape[0], dtype=target["iscrowd"].dtype))
            )

        image, transformed = self._transform(image, combined_target)
        ignored_rows = transformed["labels"] == -1
        transformed["ignored_boxes"] = transformed["boxes"][ignored_rows]
        for key in ("boxes", "labels", "area", "iscrowd"):
            value = transformed.get(key)
            if value is not None and value.ndim > 0 and value.shape[0] == ignored_rows.shape[0]:
                transformed[key] = value[~ignored_rows]
        return image, transformed


class IgnoreAwareCocoDetection(CocoDetection):
    """Use RF-DETR's COCO dataset while preserving ignored boxes through transforms."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.include_masks or self.include_keypoints:
            raise ValueError("IgnoreAwareCocoDetection supports detection boxes only")
        self.prepare = IgnoreAwareConvertCoco(cat2label=self.cat2label)
        if self._transforms is not None:
            self._transforms = IgnoreAwareTransform(self._transforms)


def _clipped_crowd_boxes(annotations: list[dict[str, Any]], image_width: int, image_height: int) -> Tensor:
    boxes = torch.as_tensor(
        [annotation["bbox"] for annotation in annotations if int(annotation.get("iscrowd", 0)) == 1],
        dtype=torch.float32,
    ).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=image_width)
    boxes[:, 1::2].clamp_(min=0, max=image_height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid]


def _box_areas(boxes: Tensor) -> Tensor:
    return (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)