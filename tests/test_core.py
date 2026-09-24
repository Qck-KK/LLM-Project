import unittest

import torch
import torch.nn as nn

from analysis.analyze_offline_pruning import (
    aggregate_pruning_records,
    calibrate_threshold,
    causal_prefix_scores,
    classify_trajectory,
    stopping_index,
)
from analysis.holm_correction import bootstrap_p_value, holm_adjust
from eval.eval_utils import (
    average_precision,
    best_threshold_accuracy,
    deterministic_example_split,
    roc_auc,
)
from pqm_loss import pqm_loss
from reward_heads import (
    AttentionPoolingHead,
    AttentionPoolingPositionHead,
    CNNHead,
    LinearHead,
)


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


class MultipleComparisonTests(unittest.TestCase):
    def test_holm_matches_hand_computed_values(self):
        # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> 0.06
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
        for got, expected in zip(adjusted, [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, expected)

    def test_holm_ignores_nan_in_family_size(self):
        adjusted = holm_adjust([0.02, float("nan")])
        self.assertAlmostEqual(adjusted[0], 0.02)
        self.assertTrue(adjusted[1] != adjusted[1])

    def test_bootstrap_p_value_is_floored_and_two_sided(self):
        self.assertAlmostEqual(bootstrap_p_value(0.0, 2000), 2 / 2001)
        self.assertAlmostEqual(bootstrap_p_value(1.0, 2000), 2 / 2001)
        self.assertEqual(bootstrap_p_value(0.5, 2000), 1.0)


class ArchitectureTests(unittest.TestCase):
    def test_plain_attention_is_permutation_equivariant(self):
        """Self-attention without positions cannot tell step order apart.

        Documents why `attention` scores exactly 0.0000 ROC-AUC change under the
        reverse and swap perturbations: permuting the steps merely permutes the
        outputs, so the (score, label) pairs are unchanged.
        """
        torch.manual_seed(0)
        head = AttentionPoolingHead(hidden_size=32, n_heads=4).eval()
        step_hidden = torch.randn(1, 5, 32)
        step_mask = torch.ones(1, 5, dtype=torch.bool)
        order = torch.tensor([4, 3, 2, 1, 0])
        with torch.no_grad():
            straight = head(step_hidden, step_mask)[0]
            permuted = head(step_hidden[:, order], step_mask[:, order])[0]
        self.assertTrue(torch.allclose(straight.flip(0), permuted, atol=1e-5))

    def test_positional_attention_is_order_sensitive(self):
        """The sinusoidal variant must break that equivariance, at equal size."""
        torch.manual_seed(0)
        head = AttentionPoolingPositionHead(hidden_size=32, n_heads=4).eval()
        step_hidden = torch.randn(1, 5, 32)
        step_mask = torch.ones(1, 5, dtype=torch.bool)
        order = torch.tensor([4, 3, 2, 1, 0])
        with torch.no_grad():
            straight = head(step_hidden, step_mask)[0]
            permuted = head(step_hidden[:, order], step_mask[:, order])[0]
        self.assertFalse(torch.allclose(straight.flip(0), permuted, atol=1e-5))

        plain = AttentionPoolingHead(hidden_size=32, n_heads=4)
        self.assertEqual(sum(p.numel() for p in head.parameters()),
                         sum(p.numel() for p in plain.parameters()))


class StabilityTests(unittest.TestCase):
    def test_attention_head_survives_all_padded_trajectory(self):
        """A zero-step trajectory must not turn the whole batch into NaN.

        ~0.1% of cached trajectories have no usable steps. Feeding an all-True
        key_padding_mask to MultiheadAttention makes softmax return NaN, which
        the loss mask hides but backward still propagates into every weight.
        """
        torch.manual_seed(0)
        head = AttentionPoolingHead(hidden_size=32, n_heads=4)
        step_hidden = torch.randn(2, 4, 32)
        step_mask = torch.tensor([[True, True, False, False],
                                  [False, False, False, False]])
        q_values = head(step_hidden, step_mask)
        self.assertFalse(torch.isnan(q_values).any())

        q_values.sum().backward()
        for name, param in head.named_parameters():
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")


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

    def test_pqm_loss_excludes_unsupervised_example_from_batch_mean(self):
        """An all-padding row must neither poison nor dilute the other rows."""
        torch.manual_seed(0)
        rewards = torch.randn(2, 3)
        labels = torch.tensor([[1, 0, -100], [-100, -100, -100]])
        batch_loss = pqm_loss(rewards, labels)
        alone_loss = pqm_loss(rewards[:1], labels[:1])
        self.assertTrue(torch.isfinite(batch_loss))
        self.assertAlmostEqual(float(batch_loss), float(alone_loss), places=6)

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


class OfflinePruningTests(unittest.TestCase):
    def test_pointwise_head_causal_scores_equal_full_scores(self):
        """A pointwise head cannot see the future, so prefix scoring is a no-op.

        Any discrepancy here means the causal/full comparison is measuring a
        masking artefact rather than the contextual heads' use of later steps.
        """
        torch.manual_seed(0)
        head = LinearHead(hidden_size=16)
        step_hidden = torch.randn(3, 5, 16)
        step_mask = torch.tensor([[True, True, True, False, False],
                                  [True, True, False, False, False],
                                  [True, True, True, True, True]])
        causal = causal_prefix_scores(head, step_hidden, step_mask)
        full = head(step_hidden, step_mask)
        self.assertTrue(torch.allclose(causal[step_mask], full[step_mask], atol=1e-5))


    def test_trajectory_types_separate_recovery_cases(self):
        self.assertEqual(classify_trajectory(torch.tensor([1, 1, -100]))["kind"], "clean")
        self.assertEqual(classify_trajectory(torch.tensor([1, 0, 0]))["kind"], "monotone_error")
        self.assertEqual(classify_trajectory(torch.tensor([1, 0, 1]))["kind"], "recovery")

    def test_causal_prefix_scoring_hides_future_steps(self):
        class SequenceMeanHead(nn.Module):
            def forward(self, step_hidden, step_mask):
                masked = step_hidden.squeeze(-1) * step_mask
                mean = masked.sum(dim=1) / step_mask.sum(dim=1)
                return mean.unsqueeze(1).expand(-1, step_hidden.shape[1])

        hidden = torch.tensor([[[1.0], [3.0], [5.0]]])
        mask = torch.tensor([[True, True, True]])
        scores = causal_prefix_scores(SequenceMeanHead(), hidden, mask)
        self.assertTrue(torch.allclose(scores, torch.tensor([[1.0, 2.0, 3.0]])))

    def test_threshold_calibration_respects_clean_risk_budget(self):
        scores = torch.tensor([
            [0.1, 0.8],
            [0.2, 0.8],
            [0.3, 0.8],
            [0.4, 0.8],
        ])
        labels = torch.ones(4, 2, dtype=torch.long)
        calibration = torch.ones(4, dtype=torch.bool)
        threshold, observed, n_clean = calibrate_threshold(
            scores, labels, calibration, "single_low", 0.25
        )
        self.assertEqual(n_clean, 4)
        self.assertLessEqual(observed, 0.25)
        self.assertEqual(
            sum(stopping_index(row, threshold, "single_low") is not None for row in scores),
            1,
        )

    def test_safe_saving_excludes_false_prunes(self):
        records = [
            {"kind": "clean", "length": 4, "first_error": None,
             "stop": 1, "saved": 2, "delay": None},
            {"kind": "monotone_error", "length": 5, "first_error": 2,
             "stop": 2, "saved": 2, "delay": 0},
            {"kind": "monotone_error", "length": 5, "first_error": 3,
             "stop": 1, "saved": 3, "delay": -2},
        ]
        metrics = aggregate_pruning_records(records)
        self.assertAlmostEqual(metrics["clean_false_prune_rate"], 1.0)
        self.assertAlmostEqual(metrics["pre_error_false_prune_rate"], 0.5)
        self.assertAlmostEqual(metrics["error_coverage"], 0.5)
        self.assertAlmostEqual(metrics["safe_step_saving_rate"], 2 / 14)


if __name__ == "__main__":
    unittest.main()
