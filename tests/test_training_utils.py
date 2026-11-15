"""Regression checks for recovery-stage configuration, state and retrieval math."""
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from solver import build_lr_scheduler, build_optimizer
from solver.lr_scheduler import LRSchedulerWithWarmup
from utils.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint, strip_prefix_if_present
from utils.metrics import Evaluator, rank, retrieval_metrics
from utils.options import get_args


class TinyModel(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.base_model = nn.Linear(3, 2)
        self.classifier = nn.Linear(2, classes, bias=False)

    def forward(self, value):
        return self.classifier(self.base_model(value))


class TrainingUtilityTests(unittest.TestCase):
    def test_parser_defaults_and_real_booleans(self):
        args = get_args([])
        self.assertEqual((args.seed, args.num_epoch, args.warmup_epochs, args.batch_size), (42, 30, 5, 64))
        self.assertEqual(args.loss_names, "sdm+id")
        self.assertEqual(args.img_size, (384, 128))
        self.assertAlmostEqual(args.head_lr, 1e-4)
        self.assertFalse(args.MLM or args.use_nac or args.use_a2dfp or args.use_cdfp)
        args = get_args(["--use_a2dfp", "false", "--include_nac", "0", "--noise_threshold", ".65"])
        self.assertFalse(args.use_a2dfp or args.use_nac)
        self.assertEqual(args.noise_threshold, .65)
        self.assertEqual(get_args(["--clip_weights", "local.pt"]).clip_checkpoint, "local.pt")
        self.assertTrue(get_args(["--resume", "run.pth"]).resume)

    def test_invalid_scheduler_configuration_fails_early(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_args(["--num_epoch", "5", "--warmup_epochs", "5"])
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=1.)
        with self.assertRaises(ValueError):
            LRSchedulerWithWarmup(optimizer, total_epochs=5, warmup_epochs=5)

    def test_optimizer_groups_and_warmup_cosine_endpoints(self):
        model = TinyModel()
        args = get_args(["--num_epoch", "4", "--warmup_epochs", "1"])
        optimizer = build_optimizer(args, model)
        self.assertEqual({group["group_name"] for group in optimizer.param_groups}, {"backbone", "heads"})
        base_rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(base_rates["backbone"], 1e-5)
        self.assertAlmostEqual(base_rates["heads"], 1e-4)
        scheduler = build_lr_scheduler(args, optimizer)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)
        for step in range(4):
            optimizer.zero_grad()
            model(torch.ones(2, 3)).sum().backward()
            optimizer.step()
            scheduler.step()
            if step == 0:
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(scheduler.get_last_lr(), [0., 0.])
        optimizer.step()
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), [0., 0.])

    def test_checkpoint_restores_rng_optimizer_scheduler_and_next_update(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        args = get_args(["--num_epoch", "4", "--warmup_epochs", "1"])
        original = TinyModel()
        optimizer = build_optimizer(args, original)
        scheduler = build_lr_scheduler(args, optimizer)
        inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        original(inputs).square().sum().backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            save_checkpoint(path, original, optimizer, scheduler,
                            model_config={"classes": 2}, train_config=vars(args),
                            label_map={7: 0, 42: 1}, epoch=1, global_step=1,
                            scaler={"scale": 65536.0})
            expected_random = (random.random(), float(np.random.random()), torch.rand(3))
            restored = TinyModel()
            restored_optimizer = build_optimizer(args, restored)
            restored_scheduler = build_lr_scheduler(args, restored_optimizer)
            payload = load_checkpoint(path, restored, restored_optimizer, restored_scheduler,
                                      resume=True, restore_rng=True)
            self.assertEqual((payload["epoch"], payload["global_step"]), (1, 1))
            self.assertEqual(payload["label_map"], {"7": 0, "42": 1})
            self.assertEqual(payload["scaler"]["scale"], 65536.)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(float(np.random.random()), expected_random[1])
            torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            for model, opt, sched in [(original, optimizer, scheduler), (restored, restored_optimizer, restored_scheduler)]:
                opt.zero_grad()
                model(inputs).square().sum().backward()
                opt.step()
                sched.step()
            for left, right in zip(original.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(read_checkpoint(path)["format_version"], 1)

    def test_unprefixed_and_mixed_prefix_weights_are_not_dropped(self):
        original, restored = TinyModel(), TinyModel()
        for mixed in (False, True):
            state = {(("module." + key) if mixed and key.startswith("classifier.") else key): value
                     for key, value in original.state_dict().items()}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.pth"
                torch.save({"model": state}, path)
                load_checkpoint(path, restored)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            strip_prefix_if_present({"weight": torch.ones(1), "module.weight": torch.ones(1)})

    def test_shape_mismatch_is_explicit_and_classifier_skip_is_reported(self):
        source, target = TinyModel(3), TinyModel(2)
        unchanged = target.classifier.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pth"
            save_checkpoint(path, source)
            with self.assertRaisesRegex(RuntimeError, "shape_mismatch"):
                load_checkpoint(path, target)
            payload = load_checkpoint(path, target, allow_classifier_mismatch=True)
            self.assertIn("classifier.weight", payload["load_report"]["skipped_keys"])
            torch.testing.assert_close(target.classifier.weight, unchanged)
            torch.testing.assert_close(target.base_model.weight, source.base_model.weight)

    def test_new_identity_mapping_resets_same_size_classifier(self):
        source, target = TinyModel(2), TinyModel(2)
        original_classifier = target.classifier.weight.detach().clone()
        with torch.no_grad():
            source.classifier.weight.fill_(12.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_identities.pth"
            save_checkpoint(path, source, label_map={100: 0, 200: 1})
            # The target's {300: 0, 400: 1} mapping has the same class count.
            # Its classifier shape alone cannot identify the label semantics.
            payload = load_checkpoint(path, target, skip_classifier=True)
            torch.testing.assert_close(target.classifier.weight, original_classifier, rtol=0, atol=0)
            torch.testing.assert_close(target.base_model.weight, source.base_model.weight, rtol=0, atol=0)
            self.assertEqual(payload["load_report"]["skipped_keys"], ["classifier.weight"])
            self.assertIn("classifier.weight", payload["load_report"]["missing_keys"])
            with self.assertRaisesRegex(ValueError, "Exact resume cannot skip"):
                load_checkpoint(path, target, resume=True, skip_classifier=True)

    def test_hand_calculated_retrieval_and_small_gallery(self):
        similarity = torch.tensor([[.9, .8, .7], [.9, .8, .7]])
        cmc, mean_ap, mean_inp, _ = rank(similarity, torch.tensor([1, 2]), torch.tensor([1, 2, 1]))
        self.assertEqual(len(cmc), 10)
        self.assertEqual(float(cmc[0]), 50.)
        self.assertEqual(float(cmc[4]), 100.)
        self.assertAlmostEqual(float(mean_ap), 100 * (5 / 6 + 1 / 2) / 2, places=4)
        self.assertAlmostEqual(float(mean_inp), 100 * (2 / 3 + 1 / 2) / 2, places=4)
        _, indices = rank(similarity, [1, 2], [1, 2, 1], get_mAP=False)
        self.assertEqual(indices.shape, (2, 3))

    def test_chunked_metrics_exclude_and_count_unmatched_queries(self):
        queries = torch.tensor([[1., 0.], [0., 1.], [1., -1.]])
        gallery = torch.tensor([[1., 0.], [.5, .5], [1., .1]])
        one = retrieval_metrics(queries, gallery, [1, 2, 99], [1, 2, 1], chunk_size=1)
        all_at_once = retrieval_metrics(queries, gallery, [1, 2, 99], [1, 2, 1], chunk_size=100)
        self.assertEqual(one, all_at_once)
        self.assertEqual((one["num_valid_queries"], one["num_ignored_queries"]), (2, 1))
        with self.assertRaisesRegex(ValueError, "No query"):
            retrieval_metrics(queries[:1], gallery, [99], [1, 2, 1])

    def test_evaluator_restores_model_training_mode(self):
        class Encoder(nn.Module):
            def encode_text(self, value, attention_mask=None):
                return value.float()

            def encode_image(self, value):
                return value.float()
        model = Encoder().train()
        images = [(torch.tensor([1, 2]), torch.eye(2))]
        text = [{"pids": torch.tensor([1, 2]), "caption_ids": torch.eye(2), "attention_mask": torch.ones(2, 2)}]
        metrics = Evaluator(images, text).evaluate(model)
        self.assertEqual(metrics["R1"], 100.)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
