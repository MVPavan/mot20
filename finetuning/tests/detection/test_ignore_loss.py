import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import torch
from PIL import Image

from mot20.detection.ignore_dataset import IgnoreAwareTransform, IgnoreAwareConvertCoco
from mot20.detection.ignore_criterion import IgnoreAwareSetCriterion
from mot20.detection.ignore_loss import ignore_aware_sigmoid_focal_loss
from mot20.detection.rfdetr_integration import use_ignore_aware_rfdetr
from rfdetr.models.criterion import sigmoid_focal_loss


class IgnoreAwareSigmoidFocalLossTest(unittest.TestCase):
    def test_ignored_overlap_masks_only_the_unmatched_query(self) -> None:
        logits = torch.tensor([[[2.0], [4.0]]])
        targets = torch.tensor([[[1.0], [0.0]]])
        predicted_boxes = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]])
        ignored_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
        matched_indices = [(torch.tensor([0]), torch.tensor([0]))]

        loss = ignore_aware_sigmoid_focal_loss(
            logits,
            targets,
            num_boxes=torch.tensor(1.0),
            predicted_boxes=predicted_boxes,
            ignored_boxes=ignored_boxes,
            matched_indices=matched_indices,
            iou_threshold=0.5,
        )

        expected = sigmoid_focal_loss(logits[:, :1], targets[:, :1], num_boxes=torch.tensor(1.0))

        torch.testing.assert_close(loss, expected)


class IgnoreAwareSetCriterionTest(unittest.TestCase):
    def test_ia_bce_masks_an_unmatched_ignored_query(self) -> None:
        criterion = IgnoreAwareSetCriterion(
            num_classes=2,
            matcher=Mock(),
            weight_dict={"loss_ce": 1.0},
            focal_alpha=0.25,
            losses=["labels"],
            ia_bce_loss=True,
        )
        outputs = {
            "pred_logits": torch.tensor([[[2.0], [4.0]]]),
            "pred_boxes": torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]]),
        }
        targets = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.1, 0.1, 0.1, 0.1]]),
                "ignored_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            }
        ]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        loss = criterion.loss_labels(outputs, targets, indices, num_boxes=torch.tensor(1.0))["loss_ce"]

        probability = outputs["pred_logits"][:, :1].sigmoid()
        positive_weight = probability.pow(0.25)
        negative_weight = 1 - positive_weight
        expected = (
            negative_weight * outputs["pred_logits"][:, :1]
            - torch.nn.functional.logsigmoid(outputs["pred_logits"][:, :1]) * (positive_weight + negative_weight)
        ).sum()

        torch.testing.assert_close(loss, expected)


class IgnoreAwareConvertCocoTest(unittest.TestCase):
    def test_preserves_ignored_boxes_through_the_transform_pipeline(self) -> None:
        image = Image.new("RGB", (100, 100))
        converter = IgnoreAwareConvertCoco(cat2label={1: 0})
        _, target = converter(
            image,
            {
                "image_id": 1,
                "annotations": [
                    {"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                    {"category_id": 1, "bbox": [60, 60, 20, 20], "area": 400, "iscrowd": 1},
                ],
            },
        )

        _, transformed = IgnoreAwareTransform(_normalize_boxes)(image, target)

        torch.testing.assert_close(transformed["boxes"], torch.tensor([[0.1, 0.1, 0.3, 0.3]]))
        torch.testing.assert_close(transformed["ignored_boxes"], torch.tensor([[0.6, 0.6, 0.8, 0.8]]))
        torch.testing.assert_close(transformed["labels"], torch.tensor([0]))


def _normalize_boxes(image: Image.Image, target: dict[str, torch.Tensor]) -> tuple[Image.Image, dict[str, torch.Tensor]]:
    target["boxes"] = target["boxes"] / 100
    return image, target


class RFDETRIgnoreIntegrationTest(unittest.TestCase):
    def test_criterion_factory_builds_an_ignore_aware_criterion(self) -> None:
        import rfdetr.training.module_model as module_model
        from rfdetr.models.criterion import SetCriterion

        stock_factory = module_model.build_criterion_from_config
        stock_criterion = SetCriterion(
            num_classes=2,
            matcher=Mock(),
            weight_dict={"loss_ce": 1.0},
            focal_alpha=0.25,
            losses=["labels"],
            ia_bce_loss=True,
        )
        module_model.build_criterion_from_config = Mock(return_value=(stock_criterion, "postprocess"))
        model_config = SimpleNamespace(
            segmentation_head=False,
            use_grouppose_keypoints=False,
            device="cpu",
        )

        try:
            with use_ignore_aware_rfdetr(ignored_iou_threshold=0.6):
                criterion, postprocess = module_model.build_criterion_from_config(model_config, Mock())
        finally:
            module_model.build_criterion_from_config = stock_factory

        self.assertIsInstance(criterion, IgnoreAwareSetCriterion)
        self.assertEqual(criterion.ignored_iou_threshold, 0.6)
        self.assertEqual(postprocess, "postprocess")

    def test_restores_stock_rfdetr_components_after_training_context(self) -> None:
        import rfdetr.datasets.coco as coco
        import rfdetr.training.module_model as module_model

        stock_dataset = coco.CocoDetection
        stock_criterion_factory = module_model.build_criterion_from_config

        with use_ignore_aware_rfdetr():
            self.assertIsNot(coco.CocoDetection, stock_dataset)
            self.assertIsNot(module_model.build_criterion_from_config, stock_criterion_factory)

        self.assertIs(coco.CocoDetection, stock_dataset)
        self.assertIs(module_model.build_criterion_from_config, stock_criterion_factory)