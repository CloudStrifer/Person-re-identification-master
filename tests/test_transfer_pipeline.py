"""Real small-CLIP source training, zero-update transfer, and target fine-tuning.

All annotations and images are explicit temporary fixtures, not benchmarks.
"""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from model.clip_model import CLIP
from processor import baseline
from test import main as evaluation_main
from utils.checkpoint import read_checkpoint
from utils.options import get_args


def make_data(parent, name, train_ids, val_ids=(), test_ids=()):
    folder_name, annotation_name = {
        "Synth_PEDES": ("Synth-Pedes", "synthpedes-dataset.json"),
        "RSTPReid": ("RSTPReid", "data_captions.json"),
        "CUHK-PEDES": ("CUHK-PEDES", "reid_raw.json"),
        "ICFG-PEDES": ("ICFG-PEDES", "ICFG-PEDES.json"),
    }[name]
    folder = parent / folder_name
    image_folder = folder / ("images" if name == "Synth_PEDES" else "imgs")
    image_folder.mkdir(parents=True)
    entries = []
    for split, identities in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for identity in identities:
            for view in range(2):
                filename = f"{split}_{identity}_{view}.png"
                Image.new("RGB", (16, 32), (identity % 255, 40 + view * 30, 70)).save(image_folder / filename)
                key = "img_path" if name == "RSTPReid" else "file_path"
                entry = {"id": identity, key: f"images/{filename}" if name == "Synth_PEDES" else filename,
                         "captions": ["a person in a red coat"]}
                if name != "Synth_PEDES":
                    entry["split"] = split
                entries.append(entry)
    annotation = folder / annotation_name
    annotation.write_text(json.dumps(entries), encoding="utf-8")
    return annotation


@pytest.fixture(scope="module")
def trained_source():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    scratch = Path(__file__).resolve().parents[1] / ".recovery" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            data = root / "data"
            make_data(data, "Synth_PEDES", (1000, 2000))
            make_data(data, "RSTPReid", (10, 40), (70, 80), (90, 100))
            make_data(data, "CUHK-PEDES", (30, 40, 50), (60, 70), (80, 90))
            make_data(data, "ICFG-PEDES", (60, 70, 80), (), (90, 100))
            baseline.set_seed(7)
            clip = CLIP(embed_dim=64, image_resolution=(32, 32), vision_layers=1,
                        vision_width=64, vision_patch_size=16, stride_size=16,
                        context_length=16, vocab_size=49408, transformer_width=64,
                        transformer_heads=1, transformer_layers=1)
            weights = root / "tiny_clip.pt"
            torch.save(clip.state_dict(), weights)
            del clip
            flags = ["--root_dir", str(data), "--clip_weights", str(weights), "--device", "cpu",
                     "--img_size", "32", "16", "--text_length", "16", "--batch_size", "4",
                     "--num_instance", "2", "--test_batch_size", "2", "--num_workers", "0",
                     "--num_epoch", "1", "--warmup_epochs", "0", "--use_nac", "--use_a2dfp", "--use_cdfp",
                     "--generator_depth", "1", "--cmt_depth", "1", "--a2dfp_embed_dim", "32",
                     "--nac_corruption_rate", "1"]
            observed = []
            real_move = baseline.move_batch

            def inspect_synthetic(cpu_batch, device):
                batch = real_move(cpu_batch, device)
                assert {item["caption_source"] for item in batch["metadata"]} == {"synthetic"}
                assert not batch["target_valid_mask"].any() and not batch["corruption_mask"].any()
                observed.extend(item["sample_id"] for item in batch["metadata"])
                return batch

            with patch.object(baseline, "move_batch", side_effect=inspect_synthetic):
                source = baseline.run_training(get_args(flags + ["--dataset_name", "Synth_PEDES", "--mode", "pretrain",
                                                                 "--output_dir", str(root / "source")]))
            assert len(set(observed)) == 4
            assert source["selected_checkpoint"] == source["last_checkpoint"]
            assert not (root / "source" / "best.pth").exists()
            history = json.loads((root / "source" / "history.jsonl").read_text().splitlines()[0])
            assert history["loss_gen"] == 0 and "validation" not in history
            saved = read_checkpoint(source["last_checkpoint"])
            assert saved["selection_rule"] == "fixed_schedule_final_checkpoint"
            assert saved["epoch"] == saved["global_step"] == 1
            assert saved["label_map"] == {"1000": 0, "2000": 1}
            del saved
            weights.rename(root / "clip_unavailable.pt")
            yield root, data, flags, source
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("target", ("RSTPReid", "CUHK-PEDES", "ICFG-PEDES"))
def test_direct_transfer_keeps_source_weights_and_never_trains_or_selects_on_target(trained_source, target):
    root, data, flags, source = trained_source
    checkpoint = Path(source["selected_checkpoint"])
    digest = baseline._sha256(checkpoint)
    output = root / f"direct-{target}.json"
    models = []
    real_build = baseline._build_model

    def capture_model(*args, **kwargs):
        model = real_build(*args, **kwargs)
        models.append(model)
        return model

    with patch.object(baseline, "_build_model", side_effect=capture_model), \
            patch.object(baseline, "build_optimizer", side_effect=AssertionError("direct transfer cannot optimize")), \
            patch.object(baseline, "run_training", side_effect=AssertionError("direct transfer cannot train")):
        assert evaluation_main(["--checkpoint", str(checkpoint), "--dataset_name", target,
                                "--root_dir", str(data), "--device", "cpu", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["source_dataset"] == "Synth_PEDES" and result["dataset"] == target
    assert result["protocol"] == "direct_transfer" and result["source_training_mode"] == "pretrain"
    assert result["source_epoch"] == result["source_global_step"] == 1
    assert result["source_selection_rule"] == "fixed_schedule_final_checkpoint"
    assert result["selection"]["checkpoint_matches_selection"] is True
    assert result["checkpoint_sha256"] == digest == baseline._sha256(checkpoint)
    assert result["evaluation_scope"] == "provided_test_split"
    assert result["evaluation"]["num_test_images"] == result["metrics"]["num_gallery"] == 4
    assert result["evaluation"]["num_test_captions"] == result["metrics"]["num_queries"] == 4
    assert result["metrics"]["num_valid_queries"] == 4
    assert len(models) == 1 and models[0].classifier.weight.shape == (2, 64)
    saved = read_checkpoint(checkpoint)
    for name, value in models[0].state_dict().items():
        torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in models[0].parameters())


@pytest.mark.parametrize("target,num_classes,selection", [
    ("RSTPReid", 2, "official_validation_rank1_earliest_tie"),
    ("ICFG-PEDES", 3, "fixed_schedule_final_checkpoint"),
])
def test_target_finetune_resets_classifier_even_same_size_and_optimizer(trained_source, target, num_classes, selection):
    root, data, flags, source = trained_source
    source_payload = read_checkpoint(source["last_checkpoint"])
    source_classifier = source_payload["model"]["classifier.weight"].clone()
    source_backbone = source_payload["model"]["base_model.visual.conv1.weight"].clone()
    assert source_payload["optimizer"]["state"]
    del source_payload
    optimizer_states, initialized = [], []
    real_optimizer, real_load = baseline.build_optimizer, baseline.load_checkpoint

    def capture_optimizer(args, model):
        optimizer = real_optimizer(args, model)
        assert not optimizer.state
        optimizer_states.append(optimizer)
        return optimizer

    def inspect_initialization(path, model, *args, **kwargs):
        assert kwargs.get("skip_classifier") is True and not kwargs.get("resume", False)
        initial_classifier = model.classifier.weight.detach().clone()
        assert initial_classifier.shape == (num_classes, 64)
        if num_classes == 2:
            assert not torch.equal(initial_classifier, source_classifier)
        assert len(optimizer_states) == 1 and not optimizer_states[0].state
        report = real_load(path, model, *args, **kwargs)
        torch.testing.assert_close(model.classifier.weight, initial_classifier, rtol=0, atol=0)
        torch.testing.assert_close(model.base_model.visual.conv1.weight, source_backbone, rtol=0, atol=0)
        assert report["load_report"]["skipped_keys"] == ["classifier.weight"]
        assert not optimizer_states[0].state
        initialized.append(True)
        return report

    with patch.object(baseline, "build_optimizer", side_effect=capture_optimizer), \
            patch.object(baseline, "load_checkpoint", side_effect=inspect_initialization):
        result = baseline.run_training(get_args(flags + ["--dataset_name", target, "--mode", "finetune",
            "--finetune", source["last_checkpoint"], "--output_dir", str(root / f"target-{target}")]))
    assert initialized == [True]
    saved = read_checkpoint(result["last_checkpoint"])
    assert saved["epoch"] == 1 and saved["global_step"] == len(saved["label_map"]) // 2
    assert saved["model"]["classifier.weight"].shape == (num_classes, 64)
    assert saved["model_config"]["num_classes"] == num_classes
    assert saved["selection_rule"] == selection
    assert saved["initialization"]["kind"] == "retrieval_checkpoint"
    assert saved["initialization"]["checkpoint_sha256"] == baseline._sha256(source["last_checkpoint"])
    assert saved["initialization"]["source_dataset"] == "Synth_PEDES"
    assert saved["initialization"]["classifier_reset"] and saved["initialization"]["optimizer_reset"]
    step_counts = [int(state["step"]) for state in saved["optimizer"]["state"].values() if "step" in state]
    assert step_counts and set(step_counts) == {saved["global_step"]}
    history = json.loads((Path(result["output_dir"]) / "history.jsonl").read_text().splitlines()[0])
    assert ("validation" in history) == (target == "RSTPReid")
    evaluation = baseline.evaluate_checkpoint(result["selected_checkpoint"], device="cpu", batch_size=1)
    assert evaluation["protocol"] == "in_domain" and evaluation["source_dataset"] == target
    assert evaluation["initialization"]["source_dataset"] == "Synth_PEDES"
    assert evaluation["selection"]["checkpoint_matches_selection"] is True
