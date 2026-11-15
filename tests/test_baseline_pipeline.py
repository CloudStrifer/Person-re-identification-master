"""Synthetic end-to-end validation; this is not a benchmark experiment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from model.clip_model import CLIP
from processor import baseline
from utils.checkpoint import read_checkpoint
from utils.options import get_args


class BaselinePipelineTests(unittest.TestCase):
    def test_training_resume_and_self_contained_evaluation(self):
        torch.set_num_threads(1)
        scratch = Path(__file__).resolve().parents[1] / ".recovery" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            folder = root / "data" / "RSTPReid"
            (folder / "imgs").mkdir(parents=True)
            annotations = []
            for split, identities in (("train", (10, 40)), ("val", (80, 100)), ("test", (200, 300))):
                for identity in identities:
                    for view in range(2):
                        filename = f"{split}_{identity}_{view}.png"
                        Image.new("RGB", (16, 32), (identity % 255, 30 + view * 20, 90)).save(folder / "imgs" / filename)
                        annotations.append({"id": identity, "split": split, "img_path": filename, "captions": ["a person wears a red coat", "a person wears black trousers"]})
            (folder / "data_captions.json").write_text(json.dumps(annotations), encoding="utf-8")
            baseline.set_seed(7)
            tiny_clip = CLIP(embed_dim=512, image_resolution=(32, 32), vision_layers=1,
                vision_width=64, vision_patch_size=16, stride_size=16,
                context_length=77, vocab_size=49408, transformer_width=64,
                transformer_heads=1, transformer_layers=1)
            weights = root / "tiny_clip.pt"
            torch.save(tiny_clip.state_dict(), weights)
            flags = ["--root_dir", str(root / "data"), "--clip_weights", str(weights),
                "--device", "cpu", "--img_size", "32", "16", "--num_epoch", "2",
                "--warmup_epochs", "0", "--batch_size", "4", "--test_batch_size", "2",
                "--num_workers", "0", "--seed", "42", "--eval_period", "3"]
            reference_dir, resumed_dir = root / "reference", root / "resumed"
            reference = baseline.run_training(get_args(flags + ["--output_dir", str(reference_dir)]))
            real_save = baseline.save_checkpoint

            def interrupt_after_save(path, *args, **kwargs):
                result = real_save(path, *args, **kwargs)
                if Path(path).name == "last.pth" and kwargs["epoch"] == 1:
                    raise InterruptedError("simulated interruption at an epoch boundary")
                return result

            with patch.object(baseline, "save_checkpoint", side_effect=interrupt_after_save):
                with self.assertRaises(InterruptedError):
                    baseline.run_training(get_args(flags + ["--output_dir", str(resumed_dir)]))
            # Neither resume nor evaluation should need the original CLIP file.
            weights.rename(root / "tiny_clip_unavailable.pt")
            resume_flags = ["--resume", str(resumed_dir / "last.pth"), "--device", "cpu"]
            resumed = baseline.run_training(get_args(resume_flags), argv=resume_flags)
            expected = read_checkpoint(reference["last_checkpoint"])
            actual = read_checkpoint(resumed["last_checkpoint"])
            self.assertEqual(actual["epoch"], 2)
            self.assertEqual(actual["global_step"], expected["global_step"])
            for key in expected["model"]:
                torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
            self.assertTrue(Path(resumed["selected_checkpoint"]).exists())
            self.assertEqual(actual["label_map"], {"10": 0, "40": 1})
            result = baseline.evaluate_checkpoint(resumed["selected_checkpoint"], device="cpu", batch_size=1)
            self.assertEqual(result["split"], "test")
            for metric in ("R1", "R5", "R10", "mAP", "mINP"):
                self.assertTrue(0 <= result["metrics"][metric] <= 100)

    def test_resume_rejects_changed_schedule(self):
        flags = ["--resume", "dummy.pth", "--num_epoch", "40"]
        saved = vars(get_args([]))
        with self.assertRaisesRegex(ValueError, "num_epoch"):
            baseline._resume_configuration(get_args(flags), flags,
                {"format_version": 1, "train_config": saved})

    def test_resume_rejects_new_directory_and_worker_configuration(self):
        for flags, message in ((["--resume", "dummy.pth", "--output_dir", "a_different_run"], "run directory"),
                               (["--resume", "dummy.pth", "--num_workers", "1"], "num_workers")):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, message):
                baseline._resume_configuration(get_args(flags), flags,
                    {"format_version": 1, "train_config": vars(get_args([]))})

    def test_damaged_image_reports_error(self):
        from utils.iotools import read_image
        scratch = Path(__file__).resolve().parents[1] / ".recovery" / "tmp"
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            image = Path(temporary) / "broken.jpg"
            image.write_bytes(b"invalid image bytes")
            with self.assertRaisesRegex(OSError, "decode image"):
                read_image(str(image))


if __name__ == "__main__":
    unittest.main()
