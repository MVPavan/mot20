from __future__ import annotations

import unittest

import torch

from rfdetr.utilities.tensors import make_collate_fn

from mot20.detection.ignore_dataset import IgnoreAwareTransform


class LoaderGeometryTest(unittest.TestCase):
    def test_aspect_preserving_resize_keeps_ordinary_and_ignored_boxes_and_collator_rounds_up(self) -> None:
        from rfdetr.datasets._torchvision import RandomResize

        transform = IgnoreAwareTransform(RandomResize([920], max_size=1333))
        image = torch.zeros((3, 200, 400))
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
            "ignored_boxes": torch.tensor([[100.0, 10.0, 150.0, 50.0]]),
        }

        resized_image, resized_target = transform(image, target)

        self.assertEqual(tuple(resized_image.shape[-2:]), (666, 1332))
        self.assertEqual(resized_target["boxes"].shape[0], 1)
        self.assertEqual(resized_target["ignored_boxes"].shape[0], 1)

        collate = make_collate_fn(block_size=40)
        samples, _ = collate(
            [
                (resized_image, resized_target),
                (torch.zeros((3, 650, 1300)), {"size": torch.tensor([650, 1300])}),
            ]
        )
        self.assertEqual(tuple(samples.tensors.shape), (2, 3, 680, 1360))


if __name__ == "__main__":
    unittest.main()