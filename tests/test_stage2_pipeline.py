"""Offline end-to-end checks for the stage-2 feature-learning variants.

Fixtures use original-format annotations, actual CLIP BPE and a real, small CLIP
architecture. Their retrieval scores are not benchmark measurements.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from datasets.build import build_loaders
from model.clip_model import CLIP
from processor import baseline
from utils.checkpoint import read_checkpoint
from utils.options import get_args


class Stage2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".recovery" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        self.data = self._dataset(self.root / "source", (10, 40))
        baseline.set_seed(7)
        clip = CLIP(embed_dim=512, image_resolution=(32, 32), vision_layers=1,
            vision_width=64, vision_patch_size=16, stride_size=16,
            context_length=77, vocab_size=49408, transformer_width=64,
            transformer_heads=1, transformer_layers=1)
        self.weights = self.root / "tiny_clip.pt"
        torch.save(clip.state_dict(), self.weights)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _dataset(root, train_ids):
        folder = root / "RSTPReid"
        (folder / "imgs").mkdir(parents=True)
        annotations = []
        for split, identities in (("train", train_ids), ("val", (180, 190)),
                                  ("test", (280, 290))):
            for identity in identities:
                for view in range(2):
                    filename = f"{split}_{identity}_{view}.png"
                    Image.new("RGB", (16, 32), (identity % 255, 20 + 50 * view, 90)).save(folder / "imgs" / filename)
                    annotations.append({"id": identity, "split": split,
                        "img_path": filename, "captions": [
                            "a person wears a red coat", "a person wears black trousers"]})
        (folder / "data_captions.json").write_text(json.dumps(annotations), encoding="utf-8")
        return root

    def _flags(self, *, a2dfp=False, cdfp=False, epochs=1, data=None):
        flags = ["--root_dir", str(data or self.data), "--clip_weights", str(self.weights),
            "--device", "cpu", "--img_size", "32", "16", "--num_epoch", str(epochs),
            "--warmup_epochs", "0", "--batch_size", "4", "--test_batch_size", "2",
            "--num_instance", "2", "--num_workers", "0", "--seed", "42",
            "--a2dfp_embed_dim", "32"]
        if a2dfp:
            flags.append("--use_a2dfp")
        if cdfp:
            flags.append("--use_cdfp")
        return flags

    def test_three_variants_train_save_and_evaluate_without_clip_source(self):
        saved = []
        for name, use_a2, use_c in (("a2dfp", True, False), ("cdfp", False, True),
                                   ("joint", True, True)):
            with self.subTest(variant=name):
                flags = self._flags(a2dfp=use_a2, cdfp=use_c)
                # Verify actual loader semantics, rather than only parser flags.
                bundle = build_loaders(get_args(flags))
                if use_c:
                    for batch in bundle["train_loader"]:
                        _, counts = batch["pids"].unique(return_counts=True)
                        self.assertEqual(counts.tolist(), [2, 2])
                result = baseline.run_training(get_args(flags + ["--output_dir", str(self.root / name)]))
                payload = read_checkpoint(result["last_checkpoint"])
                for config in (payload["model_config"], payload["train_config"]):
                    self.assertEqual(config["use_a2dfp"], use_a2)
                    self.assertEqual(config["use_cdfp"], use_c)
                enhancement_keys = [key for key in payload["model"]
                    if key.startswith(("visual_enhance.", "textual_enhance."))]
                self.assertEqual(bool(enhancement_keys), use_a2)
                self.assertEqual(payload["train_config"]["sampler"], "identity" if use_c else "random")
                history = [json.loads(line) for line in (Path(result["output_dir"]) / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual("loss_cdfp" in history[0], use_c)
                if use_c:
                    self.assertGreaterEqual(history[0]["loss_cdfp"], 0)
                saved.append(result["selected_checkpoint"])
        self.weights.rename(self.root / "original_clip_is_not_available.pt")
        for checkpoint in saved:
            result = baseline.evaluate_checkpoint(checkpoint, device="cpu", batch_size=1)
            self.assertEqual(result["split"], "test")
            self.assertEqual(result["metrics"]["num_queries"], 8)
            for name in ("R1", "R5", "R10", "mAP", "mINP"):
                self.assertTrue(0 <= result["metrics"][name] <= 100)

    def test_joint_training_resume_matches_uninterrupted_parameters(self):
        flags = self._flags(a2dfp=True, cdfp=True, epochs=2)
        expected_result = baseline.run_training(get_args(flags + ["--output_dir", str(self.root / "reference")]))
        resumed_directory = self.root / "interrupted"
        real_save = baseline.save_checkpoint

        def interrupt_after_epoch_boundary(path, *args, **kwargs):
            result = real_save(path, *args, **kwargs)
            if Path(path).name == "last.pth" and kwargs["epoch"] == 1:
                raise InterruptedError("simulated interruption after complete epoch checkpoint")
            return result

        with patch.object(baseline, "save_checkpoint", side_effect=interrupt_after_epoch_boundary):
            with self.assertRaises(InterruptedError):
                baseline.run_training(get_args(flags + ["--output_dir", str(resumed_directory)]))
        self.weights.rename(self.root / "original_clip_is_not_available.pt")
        resume_flags = ["--resume", str(resumed_directory / "last.pth"), "--device", "cpu"]
        actual_result = baseline.run_training(get_args(resume_flags), argv=resume_flags)
        expected = read_checkpoint(expected_result["last_checkpoint"])
        actual = read_checkpoint(actual_result["last_checkpoint"])
        self.assertEqual(actual["epoch"], 2)
        self.assertEqual(actual["global_step"], expected["global_step"])
        self.assertEqual(actual["train_config"]["sampler"], "identity")
        for key in expected["model"]:
            torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
        torch.testing.assert_close(actual["dataloader_rng_state"], expected["dataloader_rng_state"], rtol=0, atol=0)
        result = baseline.evaluate_checkpoint(actual_result["selected_checkpoint"], device="cpu")
        self.assertEqual(result["metrics"]["num_valid_queries"], 8)

    def test_baseline_to_a2dfp_initialization_is_explicit(self):
        source = baseline.run_training(get_args(self._flags() + ["--output_dir", str(self.root / "baseline")]))
        target = self._dataset(self.root / "target", (60, 90))
        flags = self._flags(a2dfp=True, data=target) + ["--mode", "finetune",
            "--finetune", source["last_checkpoint"], "--output_dir", str(self.root / "finetuned")]
        with self.assertRaises((RuntimeError, ValueError)):
            baseline.run_training(get_args(flags))
        self.weights.rename(self.root / "original_clip_is_not_available.pt")
        result = baseline.run_training(get_args(flags + ["--initialize_new_modules"]))
        checkpoint = read_checkpoint(result["last_checkpoint"])
        self.assertEqual(checkpoint["label_map"], {"60": 0, "90": 1})
        self.assertTrue(checkpoint["model_config"]["use_a2dfp"])
        self.assertFalse(checkpoint["model_config"]["use_cdfp"])
        self.assertTrue(any(key.startswith("visual_enhance.") for key in checkpoint["model"]))
        self.assertTrue(any(key.startswith("textual_enhance.") for key in checkpoint["model"]))
        result = baseline.evaluate_checkpoint(result["last_checkpoint"], device="cpu")
        self.assertEqual(result["metrics"]["num_valid_queries"], 8)


if __name__ == "__main__":
    unittest.main()
