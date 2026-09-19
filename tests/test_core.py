import unittest

import torch

from eval.eval_utils import (
    average_precision,
    best_threshold_accuracy,
    deterministic_example_split,
    roc_auc,
)
from pqm_loss import pqm_loss
from reward_heads import CNNHead


class MetricTests(unittest.TestCase):
    def test_deterministic_split_is_disjoint_and_reproducible(self):
        calibration_a, test_a = deterministic_example_split(20, 0.5, 42)
        calibration_b, test_b = deterministic_example_split(20, 0.5, 42)
        self.assertTrue(torch.equal(calibration_a, calibration_b))
        self.assertTrue(torch.equal(test_a, test_b))
        self.assertFalse((calibration_a & test_a).any())
        self.assertTrue((calibration_a | test_a).all())

    def test_tied_scores_have_unbiased_auc_and_ap(self):
        labels = torch.tensor([1, 0, 1, 0, 1])
        scores = torch.ones(5)
        self.assertAlmostEqual(roc_auc(scores, labels), 0.5)
        self.assertAlmostEqual(average_precision(scores, labels), 0.6, places=6)

    def test_threshold_search_can_predict_all_positive(self):
        scores = torch.ones(4)
        labels = torch.ones(4, dtype=torch.long)
        _, accuracy = best_threshold_accuracy(scores, labels)
        self.assertEqual(accuracy, 1.0)


class StabilityTests(unittest.TestCase):
    def test_pqm_loss_is_finite_for_extreme_rewards(self):
        rewards = torch.tensor([[1000.0, -1000.0, 0.0]], requires_grad=True)
        labels = torch.tensor([[1, 0, -100]])
        loss = pqm_loss(rewards, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(rewards.grad).all())

    def test_pqm_loss_handles_unsupervised_example(self):
        rewards = torch.randn(1, 3, requires_grad=True)
        labels = torch.full((1, 3), -100)
        loss = pqm_loss(rewards, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)

    def test_cnn_valid_outputs_ignore_padded_values(self):
        torch.manual_seed(0)
        head = CNNHead(hidden_size=8, channels=4)
        mask = torch.tensor([[True, True, True, False, False]])
        first = torch.randn(1, 5, 8)
        second = first.clone()
        second[:, 3:] = 1000.0
        with torch.no_grad():
            first_q = head(first, mask)
            second_q = head(second, mask)
        self.assertTrue(torch.allclose(first_q[:, :3], second_q[:, :3], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
