"""Algorithm and model-interface checks using a small, real CLIP transformer."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from model import build_model
from model.clip_model import CLIP, build_CLIP_from_openai_pretrained
from model.objectives import compute_id, compute_sdm


def tiny_clip_config(image_resolution=(32, 16)):
    return dict(embed_dim=512, image_resolution=image_resolution, vision_layers=1,
                vision_width=64, vision_patch_size=16, stride_size=16,
                context_length=8, vocab_size=32, transformer_width=64,
                transformer_heads=1, transformer_layers=1)


def model_args(**overrides):
    values = dict(loss_names="sdm+id", text_length=8, temperature=0.02,
                  freeze_backbone=False, id_loss_weight=1.0, label_smoothing=0.1)
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_model(**overrides):
    return build_model(model_args(**overrides), 2, backbone=CLIP(**tiny_clip_config()))


def example_batch():
    tokens = torch.tensor([[30, 2, 4, 31, 0, 0, 0, 0],
                           [30, 2, 6, 31, 0, 0, 0, 0],
                           [30, 7, 5, 31, 0, 0, 0, 0],
                           [30, 9, 5, 31, 0, 0, 0, 0]])
    return dict(images=torch.randn(4, 3, 32, 16), caption_ids=tokens,
                attention_mask=tokens.ne(0), pids=torch.tensor([0, 0, 1, 1]))


class BaselineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_backward_has_explicit_objective_and_both_modalities_learn(self):
        model = tiny_model()
        batch = example_batch()
        original = batch["caption_ids"].clone()
        result = model(batch)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(set(result["losses"]), {"sdm", "id"})
        torch.testing.assert_close(result["loss"], sum(result["losses"].values()))
        result["loss"].backward()
        for parameter in (model.base_model.visual.conv1.weight,
                          model.base_model.token_embedding.weight,
                          model.base_model.text_projection,
                          model.classifier.weight, model.logit_scale):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(model.classifier.bias)
        self.assertEqual(model.classifier.out_features, 2)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        torch.testing.assert_close(batch["caption_ids"], original)

    def test_label_smoothing_matches_the_paper_distribution_and_sum(self):
        logits_i = torch.tensor([[2., 0., -1.], [0., 1., -1.]])
        logits_t = torch.tensor([[0., 1., 2.], [-1., 2., 0.]])
        labels = torch.tensor([0, 1])
        q = F.one_hot(labels, 3).float() * 0.9 + 0.1 / 3
        expected = -(q * F.log_softmax(logits_i, 1)).sum(1).mean()
        expected -= (q * F.log_softmax(logits_t, 1)).sum(1).mean()
        torch.testing.assert_close(compute_id(logits_i, logits_t, labels), expected)

    def test_eval_features_are_independent_of_batch_membership_and_order(self):
        model = tiny_model().eval()
        batch = example_batch()
        order = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            image = model.encode_image(batch["images"])
            text = model.encode_text(batch["caption_ids"], batch["attention_mask"])
            self.assertEqual(image.shape, (4, 512))
            self.assertEqual(text.shape, (4, 512))
            torch.testing.assert_close(model.encode_image(batch["images"][order]), image[order])
            torch.testing.assert_close(model.encode_text(batch["caption_ids"][order]), text[order])
            torch.testing.assert_close(model.encode_image(batch["images"][:1]), image[:1])
            torch.testing.assert_close(model.encode_text(batch["caption_ids"][:1]), text[:1])

    def test_clip_never_splits_or_doubles_a_training_batch(self):
        model = tiny_model().train()
        batch = example_batch()
        image, text = model.base_model(batch["images"][:2], batch["caption_ids"])
        self.assertEqual(image.shape[0], 2)
        self.assertEqual(text.shape[0], 4)
        torch.testing.assert_close(image, model.base_model.encode_image(batch["images"][:2]))
        torch.testing.assert_close(text, model.base_model.encode_text(batch["caption_ids"]))

    def test_eot_is_selected_explicitly_and_missing_or_duplicate_eot_is_rejected(self):
        model = tiny_model(eot_token_id=29).eval()
        tokens = torch.tensor([[30, 31, 2, 29, 0, 0, 0, 0]])
        with torch.no_grad():
            sequence = model.base_model.encode_text(tokens)
            torch.testing.assert_close(model.encode_text(tokens), sequence[:, 3].float())
        for broken in (torch.zeros_like(tokens), tokens.clone()):
            if broken.any():
                broken[0, 2] = 29
            with self.assertRaisesRegex(ValueError, "exactly one EOT"):
                model.encode_text(broken)

    def test_frozen_preset_still_trains_classifier_and_has_no_backbone_gradients(self):
        model = tiny_model(freeze_backbone=True)
        model(example_batch())["loss"].backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base_model.parameters()))
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)

    def test_saved_architecture_reconstructs_without_pretrained_file(self):
        model = tiny_model().eval()
        stream = io.BytesIO()
        torch.save({"config": model.backbone_config, "model": model.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = build_model(model_args(), 2, backbone_cfg=saved["config"], load_pretrained=False).eval()
        restored.load_state_dict(saved["model"], strict=True)
        batch = example_batch()
        with torch.no_grad():
            torch.testing.assert_close(restored.encode_image(batch["images"]), model.encode_image(batch["images"]))
            torch.testing.assert_close(restored.encode_text(batch["caption_ids"]), model.encode_text(batch["caption_ids"]))

    def test_local_raw_clip_loading_adapts_positions_and_preserves_all_other_weights(self):
        source = CLIP(**tiny_clip_config(image_resolution=(32, 32)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny_clip.pt"
            torch.save(source.state_dict(), path)
            restored, config = build_CLIP_from_openai_pretrained(path, (32, 16), 16)
        self.assertEqual(config["image_resolution"], (32, 16))
        self.assertEqual(restored.visual.positional_embedding.shape[0], 3)
        for name, value in source.state_dict().items():
            if name != "visual.positional_embedding":
                torch.testing.assert_close(value, restored.state_dict()[name])

    def test_missing_clip_never_attempts_an_implicit_download(self):
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(FileNotFoundError, "No download was attempted"):
                build_CLIP_from_openai_pretrained("ViT-B/16", (384, 128), 16,
                                                  download_root=directory)
            network.assert_not_called()

    def test_sdm_zero_features_remains_finite(self):
        features = torch.zeros(4, 512, requires_grad=True)
        loss = compute_sdm(features, features, torch.tensor([0, 0, 1, 1]), torch.tensor(50.))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_unrecovered_modules_fail_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "Stage 3"):
            tiny_model(MLM=True)
        with self.assertRaisesRegex(NotImplementedError, "Stage 3"):
            tiny_model(loss_names="sdm+nac")


if __name__ == "__main__":
    unittest.main()
