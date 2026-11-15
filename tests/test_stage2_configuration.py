"""Configuration and explicit checkpoint expansion boundaries for stage 2."""
from collections import OrderedDict
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.options import get_args
from utils.checkpoint import save_checkpoint, load_checkpoint


def test_cdfp_uses_identity_sampler_and_a2_only_preserves_baseline_default():
    assert get_args([]).sampler == "random"
    assert get_args(["--use_a2dfp"]).sampler == "random"
    args = get_args(["--use_cdfp", "--batch_size", "4", "--num_instance", "2"])
    assert args.sampler == "identity" and not args.use_a2dfp


def test_resume_auto_sampler_uses_saved_module_configuration(tmp_path):
    from processor.baseline import _resume_configuration
    saved = vars(get_args(["--use_cdfp"]))
    flags = ["--resume", str(tmp_path / "last.pth"), "--sampler", "auto"]
    restored = _resume_configuration(get_args(flags), flags,
        {"format_version": 1, "train_config": saved})
    assert restored.use_cdfp and restored.sampler == "identity"


@pytest.mark.parametrize("flags", [
    ["--use_cdfp", "--sampler", "random"],
    ["--use_cdfp", "--batch_size", "2", "--num_instance", "2"],
    ["--use_cdfp", "--cdfp_temperature", "0.07"],
    ["--cdfp_k_negatives", "0"],
    ["--cdfp_margin", "nan"],
    ["--a2dfp_blend_weight", "1.1"],
    ["--a2dfp_loss_weight", "0.1"],
    ["--initialize_new_modules"],
])
def test_invalid_feature_configuration_fails_before_training(flags):
    with pytest.raises(SystemExit):
        get_args(flags)


class SmallModel(nn.Module):
    def __init__(self, enhanced=False):
        super().__init__()
        self.base_model = nn.Linear(2, 2)
        if enhanced:
            self.visual_enhance = nn.Linear(2, 2)
            self.textual_enhance = nn.Linear(2, 2)


def test_explicit_new_module_initialization_never_accepts_partial_or_broken_backbone(tmp_path):
    source, target = SmallModel(), SmallModel(enhanced=True)
    path = tmp_path / "source.pth"
    save_checkpoint(path, source)
    with pytest.raises(RuntimeError, match="missing"):
        load_checkpoint(path, target)
    new = target.visual_enhance.weight.detach().clone()
    report = load_checkpoint(path, target, initialize_new_modules=True)["load_report"]
    assert len(report["initialized_keys"]) == 4
    torch.testing.assert_close(target.base_model.weight, source.base_model.weight)
    torch.testing.assert_close(target.visual_enhance.weight, new)
    partial = OrderedDict(source.state_dict())
    partial["visual_enhance.weight"] = target.visual_enhance.weight.detach()
    torch.save({"model": partial}, path)
    with pytest.raises(RuntimeError, match="missing"):
        load_checkpoint(path, target, initialize_new_modules=True)
    broken = OrderedDict(source.state_dict())
    del broken["base_model.bias"]
    torch.save({"model": broken}, path)
    with pytest.raises(RuntimeError, match="base_model.bias"):
        load_checkpoint(path, target, initialize_new_modules=True)
    with pytest.raises(ValueError, match="Exact resume"):
        load_checkpoint(path, target, resume=True, initialize_new_modules=True)
