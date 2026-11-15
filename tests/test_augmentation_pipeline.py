"""Two CPU epochs exercise offline augmentation with the real joint model."""

import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from model.clip_model import CLIP
from processor import baseline
from test import main as evaluate_main
from utils.checkpoint import read_checkpoint
from utils.options import get_args


def _prepare_run(root):
    dataset = root / "data" / "RSTPReid"
    (dataset / "imgs").mkdir(parents=True)
    annotations = []
    for split, identities in (("train", (10, 40)), ("val", (80, 90)), ("test", (100, 110))):
        for identity in identities:
            filename = f"{split}_{identity}.png"
            Image.new("RGB", (16, 32), (identity, 30, 90)).save(dataset / "imgs" / filename)
            annotations.append({"id": identity, "split": split, "img_path": filename,
                                "captions": ["a person in a red coat", "a person with black trousers"]})
    annotation_path = dataset / "data_captions.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

    offline = root / "offline"
    offline.mkdir()
    records = []
    for identity in (10, 40):
        source_id = f"train_{identity}.png"
        generated_id = f"generated_{identity}.png"
        Image.new("RGB", (16, 32), (identity, 80, 120)).save(offline / generated_id)
        records.extend([
            {"kind": "text", "source_image_id": source_id, "source_pid": identity,
             "captions": ["a pedestrian wearing a red jacket", "a person wearing black pants"],
             "provenance": {"model": "fixture"}},
            {"kind": "image", "source_image_id": source_id, "source_pid": identity,
             "image_path": generated_id, "selected": True,
             "captions": ["the generated pedestrian wears a red coat", "the generated person has black trousers"],
             "provenance": {"rank": 1, "score": 0.8, "seed": identity}},
        ])
    manifest = offline / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "dataset": "RSTPReid", "records": records}), encoding="utf-8")
    baseline.set_seed(7)
    clip = CLIP(embed_dim=64, image_resolution=(32, 32), vision_layers=1,
                vision_width=64, vision_patch_size=16, stride_size=16, context_length=16,
                vocab_size=49408, transformer_width=64, transformer_heads=1, transformer_layers=1)
    weights = root / "tiny_clip.pt"
    torch.save(clip.state_dict(), weights)
    output = root / "joint_run"
    flags = ["--root_dir", str(root / "data"), "--clip_weights", str(weights),
             "--device", "cpu", "--img_size", "32", "16", "--text_length", "16",
             "--batch_size", "4", "--num_instance", "2", "--test_batch_size", "2", "--num_workers", "0",
             "--warmup_epochs", "0", "--num_epoch", "2", "--img_aug", "false",
             "--generator_depth", "1", "--cmt_depth", "1", "--a2dfp_embed_dim", "32",
             "--nac_corruption_rate", "1", "--use_nac", "--use_a2dfp", "--use_cdfp",
             "--use_llm_augment", "--use_generated", "--augmentation_manifest", str(manifest),
             "--output_dir", str(output)]
    return flags, output, annotation_path, manifest, weights


def test_joint_augmented_training_resume_signature_and_independent_evaluation(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # These model/optimizer checkpoints are large enough to warrant immediate
        # cleanup, instead of leaving them in pytest's retained temporary roots.
        with tempfile.TemporaryDirectory(dir=tmp_path) as temporary:
            root = Path(temporary)
            flags, output, annotation_path, manifest, weights = _prepare_run(root)
            original_manifest = manifest.read_bytes()
            original_annotations = annotation_path.read_bytes()
            observed_sources = set()
            augmented_samples = set()
            human_noise = []
            real_move = baseline.move_batch
            real_save = baseline.save_checkpoint

            def observe_training_batch(cpu_batch, device):
                batch = real_move(cpu_batch, device)
                for index, metadata in enumerate(batch["metadata"]):
                    source = metadata["caption_source"]
                    observed_sources.add(source)
                    assert metadata["split"] == "train"
                    assert metadata["source_pid"] in (10, 40)
                    assert metadata["source_image_id"] == f"train_{metadata['source_pid']}.png"
                    if source in ("llm", "synthetic"):
                        augmented_samples.add(metadata["sample_id"])
                        assert not batch["target_valid_mask"][index].any()
                        assert not batch["corruption_mask"][index].any()
                        assert torch.equal(batch["caption_ids"][index], batch["clean_ids"][index])
                        if source == "synthetic":
                            assert metadata["generated_image_id"].startswith("generated:")
                            assert metadata["augmentation_provenance"]["rank"] == 1
                    else:
                        assert source == "human" and batch["target_valid_mask"][index].any()
                        human_noise.append(bool(batch["corruption_mask"][index].any()))
                return batch

            def interrupt_at_epoch_boundary(path, *args, **kwargs):
                result = real_save(path, *args, **kwargs)
                if Path(path).name == "last.pth" and kwargs["epoch"] == 1:
                    raise InterruptedError("augmentation epoch boundary")
                return result

            with patch.object(baseline, "move_batch", side_effect=observe_training_batch), \
                    patch.object(baseline, "save_checkpoint", side_effect=interrupt_at_epoch_boundary), \
                    pytest.raises(InterruptedError, match="epoch boundary"):
                baseline.run_training(get_args(flags))
            assert observed_sources == {"human", "llm", "synthetic"}
            assert len(augmented_samples) == 8 and human_noise and all(human_noise)
            checkpoint = output / "last.pth"
            saved = read_checkpoint(checkpoint)
            assert saved["epoch"] == 1 and saved["global_step"] == 3
            assert saved["label_map"] == {"10": 0, "40": 1}
            signature = saved["data_signature"]
            assert signature["annotation_sha256"] == hashlib.sha256(original_annotations).hexdigest()
            assert signature["training_pairs"] == 12
            assert signature["augmentation"] == {
                "version": 1, "sha256": hashlib.sha256(original_manifest).hexdigest(),
                "use_llm_augment": True, "use_generated": True,
                "text_records": 2, "image_records": 2, "added_captions": 8,
            }
            for config in (saved["train_config"], saved["model_config"]):
                assert all(config[name] for name in
                           ("use_nac", "use_a2dfp", "use_cdfp", "use_llm_augment", "use_generated"))
                assert config["sampler"] == "identity"
            for prefix in ("nac.", "visual_enhance.", "textual_enhance."):
                assert any(name.startswith(prefix) for name in saved["model"])
            prior_classifier = saved["model"]["classifier.weight"].clone()
            del saved

            # Provenance changes keep the labels, captions and sample count
            # identical, so rejection must come from the manifest digest.
            changed = json.loads(original_manifest)
            changed["records"][0]["provenance"]["model"] = "changed-fixture"
            manifest.write_text(json.dumps(changed), encoding="utf-8")
            weights.rename(root / "source_clip_unavailable.pt")
            resume_flags = ["--resume", str(checkpoint), "--device", "cpu"]
            with patch.object(baseline, "_build_model", side_effect=AssertionError("data guard must run first")), \
                    pytest.raises(ValueError, match="changed"):
                baseline.run_training(get_args(resume_flags), argv=resume_flags)
            manifest.write_bytes(original_manifest)
            resumed = baseline.run_training(get_args(resume_flags), argv=resume_flags)
            final = read_checkpoint(resumed["last_checkpoint"])
            assert final["epoch"] == 2 and final["global_step"] == 6
            assert final["data_signature"] == signature
            assert not torch.equal(final["model"]["classifier.weight"], prior_classifier)
            assert final["optimizer"]["state"] and final["dataloader_rng_state"].numel() > 0
            history = [json.loads(line) for line in (output / "history.jsonl").read_text(encoding="utf-8").splitlines()]
            assert history[-1]["epoch"] == 2 and history[-1]["loss_gen"] > 0
            assert history[-1]["metric_nac_valid_target_tokens"] > 0
            assert history[-1]["loss_cdfp"] >= 0
            del final

            # The independent test entry point must use only original test
            # images, even though its checkpoint remembers augmentation flags.
            manifest.parent.rename(root / "offline_unavailable")
            assert not manifest.exists() and not weights.exists()
            assert annotation_path.read_bytes() == original_annotations
            metrics_path = root / "evaluation.json"
            assert evaluate_main(["--checkpoint", resumed["selected_checkpoint"], "--device", "cpu",
                                  "--test_batch_size", "1", "--output", str(metrics_path)]) == 0
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert result["dataset"] == "RSTPReid" and result["split"] == "test"
            assert result["metrics"]["num_queries"] == result["metrics"]["num_valid_queries"] == 4
            for metric in ("R1", "R5", "R10", "mAP", "mINP"):
                assert 0 <= result["metrics"][metric] <= 100
    finally:
        torch.set_num_threads(old_threads)
