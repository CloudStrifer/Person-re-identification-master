"""NAC training/epoch-resume/evaluation with small actual CLIP computation."""
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from model.clip_model import CLIP
from processor import baseline
from utils.checkpoint import read_checkpoint
from utils.options import get_args


@pytest.fixture
def nac_run(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / "data" / "RSTPReid"
    (root / "imgs").mkdir(parents=True)
    entries = []
    for split, identities in (("train", (10, 40)), ("val", (80, 90)), ("test", (100, 110))):
        for identity in identities:
            for view in range(2):
                name = f"{split}_{identity}_{view}.png"
                Image.new("RGB", (16, 32), (identity, 30 + view * 30, 90)).save(root / "imgs" / name)
                entries.append({"id": identity, "split": split, "img_path": name,
                    "captions": ["a person in a red coat", "a person with black trousers"]})
    (root / "data_captions.json").write_text(json.dumps(entries), encoding="utf-8")
    baseline.set_seed(7)
    clip = CLIP(embed_dim=64, image_resolution=(32, 32), vision_layers=1,
        vision_width=64, vision_patch_size=16, stride_size=16, context_length=16,
        vocab_size=49408, transformer_width=64, transformer_heads=1, transformer_layers=1)
    weights = tmp_path / "tiny_clip.pt"
    torch.save(clip.state_dict(), weights)
    flags = ["--root_dir", str(tmp_path / "data"), "--clip_weights", str(weights),
        "--device", "cpu", "--img_size", "32", "16", "--text_length", "16",
        "--batch_size", "4", "--num_instance", "2", "--test_batch_size", "2", "--num_workers", "0",
        "--warmup_epochs", "0", "--generator_depth", "1", "--cmt_depth", "1", "--a2dfp_embed_dim", "32",
        "--nac_corruption_rate", "1", "--use_nac"]
    return tmp_path, weights, flags


def test_nac_alone_and_joint_saved_models_evaluate_without_source_clip(nac_run):
    root, weights, flags = nac_run
    paths = []
    for name, extra in (("nac", []), ("joint", ["--use_a2dfp", "--use_cdfp"])):
        run = baseline.run_training(get_args(flags + extra + ["--num_epoch", "1", "--output_dir", str(root / name)]))
        saved = read_checkpoint(run["last_checkpoint"])
        assert saved["model_config"]["use_nac"]
        assert any(key.startswith("nac.") for key in saved["model"])
        history = json.loads((Path(run["output_dir"]) / "history.jsonl").read_text().splitlines()[0])
        assert history["loss_gen"] > 0 and history["metric_nac_valid_target_tokens"] > 0
        paths.append(run["selected_checkpoint"])
    weights.rename(root / "original_weights_unavailable.pt")
    for checkpoint in paths:
        result = baseline.evaluate_checkpoint(checkpoint, device="cpu", batch_size=1)
        assert result["split"] == "test" and result["metrics"]["num_valid_queries"] == 8


def test_nac_joint_epoch_resume_preserves_corruption_rng_and_parameters(nac_run):
    root, weights, flags = nac_run
    flags += ["--use_a2dfp", "--use_cdfp", "--num_epoch", "2"]
    reference = baseline.run_training(get_args(flags + ["--output_dir", str(root / "reference")]))
    interrupted = root / "interrupted"
    real_save = baseline.save_checkpoint
    def stop_after_epoch(path, *args, **kwargs):
        value = real_save(path, *args, **kwargs)
        if Path(path).name == "last.pth" and kwargs["epoch"] == 1:
            raise InterruptedError("epoch-boundary interruption")
        return value
    with patch.object(baseline, "save_checkpoint", side_effect=stop_after_epoch), pytest.raises(InterruptedError):
        baseline.run_training(get_args(flags + ["--output_dir", str(interrupted)]))
    weights.rename(root / "original_weights_unavailable.pt")
    resume_flags = ["--resume", str(interrupted / "last.pth"), "--device", "cpu"]
    resumed = baseline.run_training(get_args(resume_flags), argv=resume_flags)
    expected, actual = read_checkpoint(reference["last_checkpoint"]), read_checkpoint(resumed["last_checkpoint"])
    assert actual["epoch"] == 2 and actual["global_step"] == expected["global_step"]
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)


def test_existing_model_requires_explicit_initialization_of_nac(nac_run):
    root, weights, flags = nac_run
    base_flags = [flag for flag in flags if flag != "--use_nac"]
    previous = baseline.run_training(get_args(base_flags + ["--num_epoch", "1", "--output_dir", str(root / "base")]))
    target = flags + ["--num_epoch", "1", "--mode", "finetune", "--finetune", previous["last_checkpoint"],
        "--output_dir", str(root / "new_nac")]
    with pytest.raises(RuntimeError, match="missing"):
        baseline.run_training(get_args(target))
    result = baseline.run_training(get_args(target + ["--initialize_new_modules"]))
    assert any(key.startswith("nac.") for key in read_checkpoint(result["last_checkpoint"])["model"])
