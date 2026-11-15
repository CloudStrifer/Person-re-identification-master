"""Hand-computable tests of A2DFP aggregation and CDFP mining."""
import math
import unittest
import torch
from model.a2dfp_module import A2DFP, ContrastiveHardMining


class A2DFPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_uniform_case_matches_both_weighting_steps_and_projection(self):
        model = A2DFP(2, 2).double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for layer in (model.projection[0], model.projection[2]):
                layer.weight.copy_(torch.eye(2, dtype=torch.float64))
        output, info = model(torch.tensor([[[2., 4.], [6., 8.]]], dtype=torch.float64), return_diagnostics=True)
        torch.testing.assert_close(output, torch.tensor([[2., 3.]], dtype=torch.float64))
        for key in ('attention_weights', 'pool_weights'):
            torch.testing.assert_close(info[key], torch.tensor([[.5, .5]], dtype=torch.float64))

    def test_importance_times_context_does_not_have_extra_sequence_scaling(self):
        model = A2DFP(1, 1).double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for layer in (model.attention_fc[0], model.attention_fc[2], model.projection[0], model.projection[2]):
                layer.weight.fill_(1)
        features = torch.tensor([[[1.], [3.]]], dtype=torch.float64)
        output, info = model(features, return_diagnostics=True)
        expected_weights = torch.softmax(features.squeeze(-1).sigmoid() * .5, dim=1)
        torch.testing.assert_close(info['attention_weights'], expected_weights)
        expected = (features.squeeze(-1) * expected_weights).sum(1, keepdim=True) / 2
        torch.testing.assert_close(output, expected)

    def test_padding_even_nan_is_ignored_in_output_and_gradient(self):
        model = A2DFP(4, 8)
        source = torch.randn(2, 3, 4)
        padded = torch.cat((source, torch.full((2, 2, 4), float('nan'))), dim=1).requires_grad_()
        mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
        actual, info = model(padded, mask, return_diagnostics=True)
        torch.testing.assert_close(actual, model(source), rtol=0, atol=0)
        self.assertTrue(info['attention_weights'][:, 3:].eq(0).all())
        self.assertTrue(info['pool_weights'][:, 3:].eq(0).all())
        actual.sum().backward()
        self.assertTrue(torch.isfinite(padded.grad).all())
        self.assertTrue(padded.grad[:, 3:].eq(0).all())

    def test_all_invalid_and_empty_return_connected_zero(self):
        for count in (0, 3):
            with self.subTest(count=count):
                model = A2DFP(4, 8)
                features = torch.full((2, count, 4), float('nan'), requires_grad=True)
                output, info = model(features, torch.zeros(2, count, dtype=torch.bool), return_diagnostics=True)
                self.assertTrue(output.eq(0).all())
                self.assertFalse(info['valid_samples'].any())
                output.sum().backward()
                self.assertIsNotNone(features.grad)
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertTrue(features.grad.eq(0).all())

    def test_attention_gate_pool_and_projection_each_receive_nonzero_gradients(self):
        model = A2DFP(16, 32)
        features = torch.randn(4, 7, 16, requires_grad=True)
        output = model(features)
        (output * torch.randn_like(output)).sum().backward()
        for name in ('attention_fc', 'context_gate', 'pool_fc', 'projection'):
            with self.subTest(name=name):
                parameters = list(getattr(model, name).parameters())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in parameters), 0)
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_cpu_autocast_preserves_finite_outputs_softmax_and_gradients(self):
        model = A2DFP(16, 32)
        features = torch.randn(4, 7, 16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output, info = model(features, return_diagnostics=True)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(info['attention_weights'].dtype, torch.float32)
        output.float().square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_inputs_and_nonfinite_valid_features_fail_explicitly(self):
        model = A2DFP(4, 8)
        with self.assertRaisesRegex(ValueError, 'finite'):
            model(torch.full((1, 2, 4), float('nan')))
        with self.assertRaisesRegex(ValueError, 'valid_mask'):
            model(torch.ones(1, 2, 4), torch.ones(1, 2))
        with self.assertRaisesRegex(ValueError, 'dimension'):
            model(torch.ones(1, 2, 3))


class ContrastiveHardMiningTests(unittest.TestCase):
    def test_hand_calculated_top_two_hinge_and_nonzero_gradients(self):
        features = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [-1., 0.]], requires_grad=True)
        loss, info = ContrastiveHardMining()(features, torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(info['hard_positive'], torch.tensor([0., 0., -1., -1.]))
        torch.testing.assert_close(info['hard_negative_mean'], torch.tensor([0., 0., .5, -.5]))
        torch.testing.assert_close(info['anchor_losses'], torch.tensor([.1, .1, 1.6, .6]))
        self.assertAlmostEqual(float(loss.detach()), .6, places=6)
        self.assertEqual(int(info['valid_anchors']), 4)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_positive_is_minimum_and_negative_top_k_are_largest(self):
        features = torch.tensor([[1., 0.], [.8, .6], [-1., 0.], [0., 1.], [math.sqrt(3) / 2, .5], [-.8, -.6]])
        _, info = ContrastiveHardMining(k_negatives=2)(features, torch.tensor([0, 0, 0, 1, 1, 2]))
        self.assertAlmostEqual(float(info['hard_positive'][0]), -1, places=6)
        self.assertAlmostEqual(float(info['hard_negative_mean'][0]), math.sqrt(3) / 4, places=6)
        self.assertEqual(int(info['selected_negative_counts'][0]), 2)

    def test_anchors_missing_positives_are_excluded_from_mean(self):
        features = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        loss, info = ContrastiveHardMining()(features, torch.tensor([0, 0, 1]))
        self.assertAlmostEqual(float(loss.detach()), .6, places=6)
        self.assertEqual(int(info['valid_anchors']), 2)
        self.assertEqual(int(info['invalid_anchors']), 1)
        self.assertEqual(float(info['anchor_losses'][2]), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_no_positive_no_negative_singleton_and_empty_have_connected_zero(self):
        for labels in ([0, 1, 2], [0, 0, 0], [0], []):
            with self.subTest(labels=labels):
                features = torch.randn(len(labels), 4, requires_grad=True)
                loss, info = ContrastiveHardMining()(features, torch.tensor(labels, dtype=torch.long))
                self.assertEqual(float(loss.detach()), 0)
                self.assertEqual(int(info['valid_anchors']), 0)
                self.assertTrue(info['hard_positive'].eq(0).all())
                self.assertTrue(info['hard_negative_mean'].eq(0).all())
                loss.backward()
                self.assertIsNotNone(features.grad)
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertTrue(features.grad.eq(0).all())

    def test_fewer_than_k_negatives_uses_available_and_self_is_excluded(self):
        features = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        first, info = ContrastiveHardMining(k_negatives=2)(features, torch.tensor([0, 0, 1]))
        second, _ = ContrastiveHardMining(k_negatives=20)(features, torch.tensor([0, 0, 1]))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(info['positive_counts'], torch.tensor([1, 1, 0]))
        torch.testing.assert_close(info['selected_negative_counts'], torch.tensor([1, 1, 2]))

    def test_zero_features_and_ties_are_finite_and_temperature_is_not_supported(self):
        features = torch.zeros(4, 4, requires_grad=True)
        loss, _ = ContrastiveHardMining(margin=.1)(features, torch.tensor([0, 0, 1, 1]))
        self.assertAlmostEqual(float(loss.detach()), .1, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(list(ContrastiveHardMining().parameters()), [])
        with self.assertRaises(TypeError):
            ContrastiveHardMining(temperature=.07)

    def test_cpu_autocast_keeps_cosine_loss_fp32(self):
        features = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [-1., 0.]], requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, _ = ContrastiveHardMining()(features, torch.tensor([0, 0, 1, 1]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(float(loss.detach()), .6, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_arguments_and_nonfinite_features_are_rejected(self):
        for arguments in ({'margin': float('nan')}, {'margin': -.1}, {'k_negatives': 0}, {'k_negatives': True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ContrastiveHardMining(**arguments)
        with self.assertRaises(TypeError):
            ContrastiveHardMining()(torch.ones(2, 4), torch.ones(2))
        with self.assertRaises(ValueError):
            ContrastiveHardMining()(torch.full((2, 4), float('inf')), torch.tensor([0, 1]))


if __name__ == '__main__':
    unittest.main()
