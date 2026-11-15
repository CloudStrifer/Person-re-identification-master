"""NAC mask, generation-gradient and corrected-retrieval contracts."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from model import build_model
from model import objectives
from model.nac_module import NoiseAwareCorrection
from tests.test_baseline_model import example_batch, model_args, tiny_clip_config, tiny_model


def nac_model(**overrides):
    values = dict(use_nac=True, generator_depth=1, cmt_depth=1)
    values.update(overrides)
    return tiny_model(**values)


def supervised_batch():
    batch = example_batch()
    batch["clean_ids"] = batch["caption_ids"].clone()
    batch["clean_ids"][:, 1] = 11
    positions = torch.arange(8)[None]
    batch["target_valid_mask"] = ((positions > 0) & (positions < 3)).expand(4, -1).clone()
    batch["corruption_mask"] = torch.zeros_like(batch["caption_ids"], dtype=torch.bool)
    batch["corruption_mask"][:, 1] = True
    return batch


class NACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_cosine_uses_local_patches_and_position_masks_preserve_ordinary_zero(self):
        nac = NoiseAwareCorrection(4, 8, 6, 7, mask_token_id=5, cmt_depth=1, generator_depth=1)
        # CLS matches every text token, but local patches are orthogonal.
        image = torch.tensor([[[0., 3., 0., 0.], [2., 0., 0., 0.]]])
        text = torch.tensor([[[0., 8., 0., 0.]]]).expand(1, 8, -1).clone()
        ids = torch.tensor([[6, 0, 6, 5, 2, 7, 0, 0]])
        attention = torch.tensor([[True, True, True, True, False, True, False, False]])
        with torch.no_grad():
            nac.vocab_projection.weight.zero_()
            nac.vocab_projection.bias.zero_()
            nac.vocab_projection.bias[6] = 100  # forbidden SOT
            nac.vocab_projection.bias[7] = 90   # forbidden EOT
            nac.vocab_projection.bias[5] = 80   # forbidden MASK
            nac.vocab_projection.bias[3] = 10
        output = nac(image, text, ids, attention)
        self.assertEqual(output["detected_noise_mask"].tolist(), [[False, True, False, False, False, False, False, False]])
        torch.testing.assert_close(output["noise_scores"], torch.ones(1, 8))
        self.assertEqual(output["corrected_ids"].tolist(), [[6, 3, 6, 5, 2, 7, 0, 0]])
        # Token ID zero is also a permitted ordinary prediction.
        with torch.no_grad():
            nac.vocab_projection.bias[0] = 20
        self.assertEqual(int(nac(image, text, ids, attention)["corrected_ids"][0, 1]), 0)
        torch.testing.assert_close(ids, torch.tensor([[6, 0, 6, 5, 2, 7, 0, 0]]))

    def test_generation_ce_is_global_masked_mean_and_truth_is_only_diagnostic(self):
        nac = NoiseAwareCorrection(4, 8, 6, 7, cmt_depth=1, generator_depth=1)
        image = torch.tensor([[[5., 0., 0., 0.], [1., 0., 0., 0.]]]).expand(2, -1, -1)
        text = torch.randn(2, 5, 4)
        text[:, 1:3] = torch.tensor([0., 2., 0., 0.])
        ids = torch.tensor([[6, 0, 2, 7, 0], [6, 1, 3, 7, 0]])
        clean = ids.clone()
        clean[:, 1:3] = torch.tensor([3, 4])
        valid = torch.tensor([[False, True, True, False, False], [False, True, False, False, False]])
        truth = torch.zeros_like(valid)
        truth[0, 1] = True
        output = nac(image, text, ids, clean_ids=clean, target_valid_mask=valid, corruption_mask=truth)
        mask = output["detected_noise_mask"] & valid
        torch.testing.assert_close(output["gen_loss"], F.cross_entropy(output["logits"][mask], clean[mask]))
        self.assertEqual(int(output["valid_token_count"]), 3)
        self.assertEqual(int(output["metrics"]["nac_detected_tokens"]), 4)
        self.assertEqual(int(output["metrics"]["nac_detection_true_positives"]), 1)
        self.assertEqual(int(output["metrics"]["nac_detection_false_positives"]), 3)
        with self.assertRaisesRegex(ValueError, "explicit target_valid_mask"):
            nac(image, text, ids, clean_ids=clean)

    def test_generation_ce_trains_fusion_generator_and_both_original_backbone_passes(self):
        model = nac_model(noise_threshold=0.0)
        output = model(supervised_batch())
        self.assertGreater(int(output["metrics"]["nac_valid_target_tokens"]), 0)
        output["losses"]["gen"].backward()
        for parameter in (model.nac.text_self_attn.in_proj_weight,
                          model.nac.image_self_attn.in_proj_weight,
                          model.nac.cross_attn.in_proj_weight,
                          model.nac.cross_modal_transformer.layers[0].linear1.weight,
                          model.nac.generator.layers[0].linear1.weight,
                          model.nac.vocab_projection.weight,
                          model.base_model.visual.conv1.weight,
                          model.base_model.token_embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_repaired_sequence_feeds_all_retrieval_losses_before_a2dfp(self):
        model = nac_model(use_a2dfp=True, a2dfp_embed_dim=32, use_cdfp=True)
        batch = supervised_batch()
        detected = torch.zeros_like(batch["target_valid_mask"])
        detected[0, 1:3] = True  # other samples must retain their original encoding
        with torch.no_grad():
            model.nac.vocab_projection.weight.zero_()
            model.nac.vocab_projection.bias.zero_()
            model.nac.vocab_projection.bias[11] = 10
            model.nac.vocab_projection.bias[31] = 100  # EOT cannot be generated
        original_encode = model.base_model.encode_text
        calls = []

        def record_encode(ids):
            calls.append((ids.detach().clone(), torch.is_grad_enabled()))
            return original_encode(ids)

        with patch.object(model.nac, "detect_noise", return_value=(detected, torch.ones_like(detected).float())), \
                patch.object(model.base_model, "encode_text", side_effect=record_encode):
            output = model(batch)
        corrected = batch["caption_ids"].clone()
        corrected[0, 1:3] = 11
        torch.testing.assert_close(output["nac"]["corrected_ids"], corrected)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1])
        self.assertFalse(calls[1][1])
        torch.testing.assert_close(calls[1][0], corrected[:1])
        with torch.no_grad():
            image = model.encode_image(batch["images"])
            text = model.encode_text(corrected, batch["attention_mask"])
            stale = model.encode_text(batch["caption_ids"], batch["attention_mask"])
            expected_sdm = objectives.compute_sdm(image, text, batch["pids"], model.logit_scale.exp())
            expected_id = objectives.compute_id(model.classifier(image), model.classifier(text), batch["pids"], .1)
            expected_cdfp = (model.contrastive_mining(image, batch["pids"])[0]
                             + model.contrastive_mining(text, batch["pids"])[0]) / 2
            stale_id = objectives.compute_id(model.classifier(image), model.classifier(stale), batch["pids"], .1)
        torch.testing.assert_close(output["losses"]["sdm"], expected_sdm)
        torch.testing.assert_close(output["losses"]["id"], expected_id)
        torch.testing.assert_close(output["losses"]["cdfp"], expected_cdfp)
        self.assertGreater(abs(float((expected_id - stale_id))), 1e-6)
        losses = output["losses"]
        torch.testing.assert_close(output["loss"], losses["sdm"] + losses["id"] + .1 * losses["cdfp"] + .5 * losses["gen"])
        output["losses"]["id"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.nac.parameters()))
        self.assertGreater(float(model.textual_enhance.projection[0].weight.grad.abs().sum()), 0)

    def test_empty_detection_or_targets_return_connected_zero_and_missing_clean_still_repairs(self):
        for mode in ("no_detection", "no_valid_targets", "no_clean", "empty_caption"):
            with self.subTest(mode=mode):
                model = nac_model(noise_threshold=2.0 if mode == "no_detection" else 0.0)
                batch = supervised_batch()
                if mode == "no_valid_targets":
                    batch["target_valid_mask"].zero_()
                if mode == "no_clean":
                    del batch["clean_ids"], batch["target_valid_mask"]
                    with torch.no_grad():
                        model.nac.vocab_projection.weight.zero_()
                        model.nac.vocab_projection.bias.zero_()
                        model.nac.vocab_projection.bias[11] = 10
                if mode == "empty_caption":
                    batch["caption_ids"][:] = torch.tensor([30, 31, 0, 0, 0, 0, 0, 0])
                    batch["attention_mask"] = batch["caption_ids"].ne(0)
                    batch["target_valid_mask"].zero_()
                output = model(batch)
                self.assertEqual(float(output["losses"]["gen"]), 0)
                self.assertEqual(int(output["metrics"]["nac_valid_target_tokens"]), 0)
                output["losses"]["gen"].backward()
                self.assertIsNotNone(model.nac.vocab_projection.weight.grad)
                self.assertTrue(model.nac.vocab_projection.weight.grad.eq(0).all())
                if mode in ("no_detection", "empty_caption"):
                    torch.testing.assert_close(output["nac"]["corrected_ids"], batch["caption_ids"])
                else:
                    self.assertGreater(int(output["metrics"]["nac_detected_samples"]), 0)
                    if mode == "no_clean":
                        mask = output["nac"]["detected_noise_mask"]
                        self.assertTrue(output["nac"]["corrected_ids"][mask].eq(11).all())

    def test_nac_is_batch_independent_and_evaluation_bypasses_it(self):
        model = nac_model().train()
        batch = example_batch()
        image = model.encode_image_sequence(batch["images"]).detach()
        text = model.encode_text_sequence(batch["caption_ids"]).detach()
        order = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            full = model.nac(image, text, batch["caption_ids"], batch["attention_mask"])
            permuted = model.nac(image[order], text[order], batch["caption_ids"][order], batch["attention_mask"][order])
            single = model.nac(image[:1], text[:1], batch["caption_ids"][:1], batch["attention_mask"][:1])
        for key in ("logits", "detected_noise_mask", "noise_scores", "corrected_ids"):
            torch.testing.assert_close(permuted[key], full[key][order])
            torch.testing.assert_close(single[key], full[key][:1])
        model.eval()
        with patch.object(model.nac, "forward", side_effect=AssertionError("NAC used in evaluation")), torch.no_grad():
            result = model(batch)
            image_features = model.encode_image(batch["images"])
            text_features = model.encode_text(batch["caption_ids"], batch["attention_mask"])
            torch.testing.assert_close(model.encode_image(batch["images"][:1]), image_features[:1])
            torch.testing.assert_close(model.encode_text(batch["caption_ids"][:1]), text_features[:1])
        self.assertNotIn("gen", result["losses"])
        self.assertNotIn("nac", result)

    def test_defaults_reconstruction_disabled_state_and_frozen_backbone(self):
        defaults = NoiseAwareCorrection(4, 8, 6, 7)
        self.assertEqual(len(defaults.generator.layers), 6)
        self.assertEqual(len(defaults.cross_modal_transformer.layers), 4)
        self.assertTrue(all(module.batch_first for module in defaults.modules()
                            if isinstance(module, torch.nn.MultiheadAttention)))
        torch.manual_seed(12)
        baseline = tiny_model()
        torch.manual_seed(12)
        disabled = tiny_model(use_nac=False)
        self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
        for key in baseline.state_dict():
            torch.testing.assert_close(baseline.state_dict()[key], disabled.state_dict()[key], atol=0, rtol=0)
        model = nac_model(freeze_backbone=True)
        restored = build_model(model_args(include_nac=True, freeze_backbone=True, generator_depth=1, cmt_depth=1),
                               2, backbone_cfg=tiny_clip_config(), load_pretrained=False)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(all(name.startswith("nac.") for name in set(model.state_dict()) - set(baseline.state_dict())))
        batch = supervised_batch()
        with torch.no_grad():
            torch.testing.assert_close(model(batch)["loss"], restored(batch)["loss"])
        model(batch)["loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.base_model.parameters()))
        self.assertGreater(float(model.nac.vocab_projection.weight.grad.abs().sum()), 0)
        for loss_names in ("sdm+nac", "sdm+mlm", "sdm+ttc"):
            with self.assertRaises(NotImplementedError):
                nac_model(loss_names=loss_names)


if __name__ == "__main__":
    unittest.main()
