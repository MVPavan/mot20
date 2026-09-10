from __future__ import annotations

import unittest

import torch

from mot20.detection.checkpoint_loading import expand_query_parameters, prepare_expanded_checkpoint_state


class QueryExpansionTest(unittest.TestCase):
    def test_preserves_each_pretrained_group_and_keeps_new_rows_initialized(self) -> None:
        checkpoint = {
            "refpoint_embed.weight": torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2),
            "query_feat.weight": torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3),
        }
        initialized = {
            "refpoint_embed.weight": torch.full((10, 2), -1.0),
            "query_feat.weight": torch.full((10, 3), -2.0),
        }

        expanded = expand_query_parameters(checkpoint, initialized, checkpoint_num_queries=3, target_num_queries=5, group_detr=2)

        torch.testing.assert_close(expanded["refpoint_embed.weight"][:3], checkpoint["refpoint_embed.weight"][:3])
        torch.testing.assert_close(expanded["refpoint_embed.weight"][5:8], checkpoint["refpoint_embed.weight"][3:])
        torch.testing.assert_close(expanded["refpoint_embed.weight"][3:5], torch.full((2, 2), -1.0))
        torch.testing.assert_close(expanded["query_feat.weight"][8:], torch.full((2, 3), -2.0))

    def test_rejects_missing_or_mismatched_query_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing query parameter"):
            expand_query_parameters({}, {"query_feat.weight": torch.zeros((2, 3))}, 1, 1, 2)
        with self.assertRaisesRegex(ValueError, "incompatible query parameter shape"):
            expand_query_parameters(
                {
                    "refpoint_embed.weight": torch.zeros((2, 4)),
                    "query_feat.weight": torch.zeros((2, 4)),
                },
                {
                    "refpoint_embed.weight": torch.zeros((2, 4)),
                    "query_feat.weight": torch.zeros((2, 3)),
                },
                1,
                1,
                2,
            )

    def test_prepares_only_compatible_weights_and_intentional_head_reinitialization(self) -> None:
        checkpoint = {
            "refpoint_embed.weight": torch.ones((2, 4)),
            "query_feat.weight": torch.ones((2, 3)),
            "encoder.weight": torch.ones((3, 3)),
            "class_embed.weight": torch.ones((91, 3)),
        }
        initialized = {
            "refpoint_embed.weight": torch.zeros((4, 4)),
            "query_feat.weight": torch.zeros((4, 3)),
            "encoder.weight": torch.zeros((3, 3)),
            "class_embed.weight": torch.zeros((2, 3)),
        }

        prepared = prepare_expanded_checkpoint_state(checkpoint, initialized, checkpoint_num_queries=1, target_num_queries=2, group_detr=2)

        self.assertEqual(set(prepared), {"refpoint_embed.weight", "query_feat.weight", "encoder.weight"})
        torch.testing.assert_close(prepared["encoder.weight"], checkpoint["encoder.weight"])
        with self.assertRaisesRegex(ValueError, "unapproved checkpoint shape mismatch"):
            prepare_expanded_checkpoint_state(
                {**checkpoint, "encoder.weight": torch.ones((2, 3))},
                initialized,
                checkpoint_num_queries=1,
                target_num_queries=2,
                group_detr=2,
            )


if __name__ == "__main__":
    unittest.main()