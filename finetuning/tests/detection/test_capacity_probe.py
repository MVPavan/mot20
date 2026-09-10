from __future__ import annotations

import unittest

import torch

from rfdetr.utilities.tensors import NestedTensor

from mot20.detection.capacity_probe import pad_batch_to_envelope


class CapacityProbeTest(unittest.TestCase):
    def test_pad_batch_to_envelope_extends_only_masked_padding(self) -> None:
        tensors = torch.ones((2, 3, 680, 1320))
        mask = torch.zeros((2, 680, 1320), dtype=torch.bool)

        padded = pad_batch_to_envelope(NestedTensor(tensors, mask), 1360, 1360)

        self.assertEqual(tuple(padded.tensors.shape), (2, 3, 1360, 1360))
        self.assertEqual(tuple(padded.mask.shape), (2, 1360, 1360))
        self.assertTrue(torch.equal(padded.tensors[:, :, :680, :1320], tensors))
        self.assertFalse(padded.mask[:, :680, :1320].any())
        self.assertTrue(padded.mask[:, 680:, :].all())
        self.assertTrue(padded.mask[:, :, 1320:].all())

    def test_pad_batch_to_envelope_rejects_shrinking(self) -> None:
        samples = NestedTensor(torch.zeros((1, 3, 680, 1360)), torch.ones((1, 680, 1360), dtype=torch.bool))

        with self.assertRaisesRegex(ValueError, "cannot contain"):
            pad_batch_to_envelope(samples, 1360, 1320)


if __name__ == "__main__":
    unittest.main()