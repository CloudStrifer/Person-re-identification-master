"""Feature-learning wiring checks with a small, real CLIP architecture."""
import io
import unittest

import torch
from torch.nn import functional as F

from model import build_model
from model.objectives import compute_id, compute_sdm
from test_baseline_model import example_batch, model_args, tiny_model


def manual_hard_loss(features, labels, margin=0.1, k=2):
    normalized = F.normalize(features, dim=-1)
    similarity = normalized @ normalized.t()
    terms = []
    for row in range(len(features)):
        positives = [j for j in range(len(features)) if j != row and labels[j] == labels[row]]
        negatives = [j for j in range(len(features)) if labels[j] != labels[row]]
        if positives and negatives:
            hardest_positive = similarity[row, positives].min()
            hardest_negatives = similarity[row, negatives].topk(min(k, len(negatives))).values.mean()
            terms.append(F.relu(hardest_negatives - hardest_positive + margin))
    return torch.stack(terms).mean() if terms else features.sum() * 0


class Stage2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_disabled_flags_preserve_baseline_state_and_results(self):
        torch.manual_seed(7)
        baseline = tiny_model()
        torch.manual_seed(7)
        disabled = tiny_model(use_a2dfp=False, use_cdfp=False)
        self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
        batch = example_batch()
        torch.testing.assert_close(baseline(batch)["loss"], disabled(batch)["loss"], rtol=0, atol=0)
        self.assertFalse(hasattr(disabled, "visual_enhance"))

    def test_a2dfp_modalities_use_independent_parameters_and_all_parts_learn(self):
        model = tiny_model(use_a2dfp=True, a2dfp_embed_dim=32)
        result = model(example_batch())
        self.assertEqual(set(result["losses"]), {"sdm", "id"})
        result["loss"].backward()
        visual = dict(model.visual_enhance.named_parameters())
        textual = dict(model.textual_enhance.named_parameters())
        self.assertEqual(visual.keys(), textual.keys())
        self.assertTrue(visual)
        for name in visual:
            self.assertIsNot(visual[name], textual[name])
            for parameter in (visual[name], textual[name]):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_cdfp_only_keeps_global_features_and_has_no_pooling_parameters(self):
        torch.manual_seed(7)
        baseline = tiny_model()
        torch.manual_seed(7)
        model = tiny_model(use_cdfp=True)
        self.assertFalse(hasattr(model, "visual_enhance"))
        self.assertFalse(hasattr(model, "textual_enhance"))
        self.assertEqual(set(model.state_dict()), set(baseline.state_dict()))
        batch = example_batch()
        with torch.no_grad():
            torch.testing.assert_close(model.encode_image(batch["images"]), baseline.encode_image(batch["images"]), rtol=0, atol=0)
            torch.testing.assert_close(model.encode_text(batch["caption_ids"]), baseline.encode_text(batch["caption_ids"]), rtol=0, atol=0)
        self.assertIn("cdfp", model(batch)["losses"])

    def test_total_objective_uses_final_encodings_and_each_weight_exactly_once(self):
        model = tiny_model(use_a2dfp=True, use_cdfp=True, a2dfp_embed_dim=32,
                           cdfp_margin=0.3, cdfp_k_negatives=1,
                           cdfp_loss_weight=0.37, id_loss_weight=0.8)
        batch = example_batch()
        result = model(batch)
        image, text = model.encode_image(batch["images"]), model.encode_text(batch["caption_ids"], batch["attention_mask"])
        scale = model.logit_scale.clamp(max=torch.log(torch.tensor(100.0))).exp()
        expected_sdm = compute_sdm(image, text, batch["pids"], scale)
        expected_id = compute_id(model.classifier(image), model.classifier(text), batch["pids"], label_smoothing=0.1)
        image_cdfp = manual_hard_loss(image, batch["pids"], margin=0.3, k=1)
        text_cdfp = manual_hard_loss(text, batch["pids"], margin=0.3, k=1)
        expected_cdfp = (image_cdfp + text_cdfp) / 2
        torch.testing.assert_close(result["losses"]["sdm"], expected_sdm)
        torch.testing.assert_close(result["losses"]["id"], expected_id)
        torch.testing.assert_close(result["losses"]["cdfp"], expected_cdfp)
        torch.testing.assert_close(result["loss"], expected_sdm + 0.8 * expected_id + 0.37 * expected_cdfp)
        torch.testing.assert_close(result["metrics"]["cdfp_image"], image_cdfp.detach())
        torch.testing.assert_close(result["metrics"]["cdfp_text"], text_cdfp.detach())
        self.assertEqual(int(result["metrics"]["cdfp_valid_anchors"]), 4)
        self.assertFalse(result["metrics"]["cdfp_image"].requires_grad)

    def test_cdfp_loss_alone_reaches_both_encoders(self):
        model = tiny_model(use_cdfp=True, cdfp_margin=2.5)
        result = model(example_batch())
        result["losses"]["cdfp"].backward()
        for parameter in (model.base_model.visual.conv1.weight, model.base_model.text_projection):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(model.classifier.weight.grad)

    def test_text_padding_exclusion_and_ordinary_zero_token(self):
        model = tiny_model(use_a2dfp=True, a2dfp_embed_dim=32).eval()
        tokens = torch.tensor([[30, 0, 5, 31, 0, 0, 0, 0]])
        valid = torch.tensor([[True, True, True, True, False, False, False, False]])
        positions = model._eot_positions(tokens, valid)
        local = model._text_local_mask(tokens, positions, valid)
        torch.testing.assert_close(local, torch.tensor([[False, True, True, False, False, False, False, False]]))
        modified = tokens.clone()
        modified[:, 4:] = torch.tensor([9, 8, 7, 6])
        with torch.no_grad():
            torch.testing.assert_close(model.encode_text(tokens, valid), model.encode_text(modified, valid))
            torch.testing.assert_close(model.encode_text(tokens), model.encode_text(modified))
            sequence = model.encode_text_sequence(tokens)
            changed = sequence.clone()
            changed[:, 4:] = torch.nan
            torch.testing.assert_close(model._text_features(changed, tokens, positions, valid), model._text_features(sequence, tokens, positions, valid))

    def test_empty_local_text_falls_back_to_original_and_cls_is_not_pooled(self):
        model = tiny_model(use_a2dfp=True, a2dfp_embed_dim=32).eval()
        tokens = torch.tensor([[30, 31, 0, 0, 0, 0, 0, 0]])
        with torch.no_grad():
            text_sequence = model.encode_text_sequence(tokens)
            torch.testing.assert_close(model.encode_text(tokens), text_sequence[:, 1].float(), rtol=0, atol=0)
            image_sequence = model.encode_image_sequence(torch.randn(1, 3, 32, 16))
            original = model._image_features(image_sequence)
            modified = image_sequence.clone()
            modified[:, 0] += 3
            expected = original + 3 * (1 - model.a2dfp_blend_weight)
            torch.testing.assert_close(model._image_features(modified), expected)
            torch.testing.assert_close(model._image_features(image_sequence[:, :1]), image_sequence[:, 0].float())

    def test_eval_with_both_modules_is_batch_independent_without_identity_labels(self):
        model = tiny_model(use_a2dfp=True, use_cdfp=True, a2dfp_embed_dim=32).eval()
        batch, order = example_batch(), torch.tensor([3, 0, 2, 1])
        with torch.no_grad():
            image = model.encode_image(batch["images"])
            text = model.encode_text(batch["caption_ids"])
            self.assertEqual(image.shape, (4, 512))
            self.assertEqual(text.shape, (4, 512))
            torch.testing.assert_close(model.encode_image(batch["images"][order]), image[order])
            torch.testing.assert_close(model.encode_text(batch["caption_ids"][order]), text[order])
            torch.testing.assert_close(model.encode_image(batch["images"][:1]), image[:1])
            torch.testing.assert_close(model.encode_text(batch["caption_ids"][:1]), text[:1])

    def test_frozen_backbone_with_feature_modules_still_trains_pooling(self):
        model = tiny_model(use_a2dfp=True, use_cdfp=True, a2dfp_embed_dim=32, freeze_backbone=True)
        model(example_batch())["loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.base_model.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.visual_enhance.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.textual_enhance.parameters()))

    def test_variants_reconstruct_from_saved_architecture_and_feature_configuration(self):
        batch = example_batch()
        for a2dfp, cdfp in ((True, False), (False, True), (True, True)):
            args = model_args(use_a2dfp=a2dfp, use_cdfp=cdfp, a2dfp_embed_dim=32)
            model = tiny_model(**vars(args)).eval()
            stream = io.BytesIO()
            torch.save({"args": vars(args), "backbone": model.backbone_config, "model": model.state_dict()}, stream)
            stream.seek(0)
            saved = torch.load(stream, weights_only=True)
            restored = build_model(model_args(**saved["args"]), 2, backbone_cfg=saved["backbone"], load_pretrained=False).eval()
            restored.load_state_dict(saved["model"], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored.encode_image(batch["images"]), model.encode_image(batch["images"]), rtol=0, atol=0)
                torch.testing.assert_close(restored.encode_text(batch["caption_ids"]), model.encode_text(batch["caption_ids"]), rtol=0, atol=0)

    def test_unsupported_temperature_and_ambiguous_loss_tokens_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw cosine"):
            tiny_model(use_cdfp=True, cdfp_temperature=0.07)
        with self.assertRaisesRegex(NotImplementedError, "--use_cdfp"):
            tiny_model(loss_names="sdm+id+cdfp")
        for overrides in ({"a2dfp_blend_weight": 1.1}, {"a2dfp_blend_weight": float("nan")}, {"a2dfp_embed_dim": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                tiny_model(use_a2dfp=True, **overrides)


if __name__ == "__main__":
    unittest.main()
