"""CPU fixture smoke: plan -> SYNTH pretrain -> full finetune -> test -> collect.

No benchmark data, official CLIP, API or diffusion generation is used.
Temporary models are removed on completion; only the validation report remains.
"""
import json
import logging
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    import torch
    from PIL import Image
    from model.clip_model import CLIP
    from tools.experiments import build_plan, load_plan, run_job, write_plan
    from tools.collect_results import collect_results

    torch.set_num_threads(1)
    logging.disable(logging.INFO)
    scratch = PROJECT / ".recovery/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        work = Path(directory)
        root = work / "data"
        for name, source in (("Synth-Pedes", True), ("RSTPReid", False)):
            folder = root / name
            (folder / "imgs").mkdir(parents=True)
            rows = []
            splits = [("train", (10, 40))] if source else [
                ("train", (10, 40)), ("val", (80, 90)), ("test", (100, 110))]
            for split, identities in splits:
                for identity in identities:
                    for view in range(2):
                        filename = f"{split}_{identity}_{view}.png"
                        Image.new("RGB", (16, 32), (identity, 30 + view * 20, 90)).save(folder / "imgs" / filename)
                        rows.append({"id": identity, "split": split,
                            "file_path" if source else "img_path": "imgs/" + filename if source else filename,
                            "captions": ["a person in a red coat", "a person in black trousers"]})
            annotation = "synthpedes-dataset.json" if source else "data_captions.json"
            (folder / annotation).write_text(json.dumps(rows), encoding="utf-8")
        aug = work / "augmentation"
        aug.mkdir()
        records = []
        for identity in (10, 40):
            common = {"source_pid": identity, "source_image_id": f"train_{identity}_0.png",
                      "provenance": {"backend": "explicit-offline-fixture"}}
            records.append({**common, "kind": "text", "captions": [
                "a pedestrian wearing a crimson jacket", "a walker in dark pants"]})
            filename = f"generated_{identity}.png"
            Image.new("RGB", (16, 32), (identity, 50, 90)).save(aug / filename)
            records.append({**common, "kind": "image", "image_path": filename, "selected": True,
                "captions": ["this synthetic person wears a green coat", "this generated view shows gray trousers"]})
        manifest = aug / "manifest.json"
        manifest.write_text(json.dumps({"version": 1, "dataset": "RSTPReid", "records": records}), encoding="utf-8")
        clip = CLIP(embed_dim=64, image_resolution=(32, 32), vision_layers=1, vision_width=64,
            vision_patch_size=16, stride_size=16, context_length=16, vocab_size=49408,
            transformer_width=64, transformer_heads=1, transformer_layers=1)
        weights = work / "tiny_clip.pt"
        torch.save(clip.state_dict(), weights)
        del clip
        paths = {"root_dir": str(root), "bpe_path": str(PROJECT / "data/bpe_simple_vocab_16e6.txt.gz"),
            "clip_weights": str(weights), "augmentation_manifests": {"RSTPReid": {"both": str(manifest)}}}
        overrides = {"device": "cpu", "num_workers": 0, "num_epoch": 1, "warmup_epochs": 0,
            "batch_size": 4, "test_batch_size": 2, "num_instance": 2, "img_size": [32, 16],
            "text_length": 16, "generator_depth": 1, "cmt_depth": 1, "a2dfp_embed_dim": 32}
        output = work / "experiments"
        plan = build_plan(PROJECT / "configs/experiments/paper.json", paths, output,
                          groups=["module_ablation"], overrides=overrides)
        write_plan(plan, output / "experiments.json")
        plan = load_plan(output / "experiments.json")
        run_job(plan, "seed42-synth-baseline")
        weights.rename(work / "source_clip_unavailable.pt")
        run_job(plan, "seed42-rstp-full")
        evaluation = run_job(plan, "seed42-test-rstp-full")
        collected = collect_results(output, plan_path=output / "experiments.json")
        assert collected["report"]["evaluated"] == 1 and collected["report"]["pending"] == 5
        assert evaluation["protocol"] == "in_domain"
        assert evaluation["initialization"]["source_dataset"] == "Synth_PEDES"
        assert evaluation["metrics"]["num_valid_queries"] == 8
        report = {"smoke_test": True, "benchmark_training_completed": False,
            "dataset_kind": "explicit temporary fixtures", "source_and_target_epochs": 1,
            "official_clip_used": False, "stages": ["plan", "source_pretrain",
                "finetune_all_modules_and_both_augmentations", "independent_test", "collect_results"],
            "source_clip_removed_before_finetune": True,
            "num_queries": evaluation["metrics"]["num_queries"], "collector": collected["report"],
            "initialization_source_dataset": evaluation["initialization"]["source_dataset"],
            "real_server_training_completed": False}
    destination = PROJECT / ".recovery/stage5_validation/pipeline_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
