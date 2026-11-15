"""One mixed human/augmented real-RSTP batch; no epoch or benchmark evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augmentation_manifest", required=True)
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--clip_weights", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default=".recovery/stage4_validation/real_cached_text.json")
    options = parser.parse_args(argv)
    if options.threads < 1:
        parser.error("threads must be positive")

    import torch
    from datasets.build import build_loaders, collate
    from model import build_finetune_model
    from processor.baseline import move_batch, resolve_device, set_seed
    from solver import build_optimizer
    from utils.options import get_args

    torch.set_num_threads(options.threads)
    args = get_args(["--root_dir", options.root_dir, "--clip_weights", options.clip_weights,
        "--augmentation_manifest", options.augmentation_manifest, "--device", options.device,
        "--use_llm_augment", "--use_generated", "--use_nac", "--use_a2dfp", "--use_cdfp",
        "--batch_size", "4", "--num_instance", "2", "--nac_corruption_rate", "1"])
    started = time.monotonic()
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = build_loaders(args)
    records = bundle["dataset"].train_records
    human, augmented = {}, {}
    for index, row in enumerate(records):
        key = (row["source_pid"], row["source_image_id"])
        if row["caption_source"] == "human":
            human.setdefault(key, index)
        else:
            augmented.setdefault(key, index)
    selected, identities = [], set()
    for key, index in augmented.items():
        if key in human and key[0] not in identities:
            selected.extend([human[key], index])
            identities.add(key[0])
        if len(identities) == 2:
            break
    if len(identities) != 2:
        raise ValueError("Validation needs augmented records from at least two original training identities")
    batch = move_batch(collate([bundle["train_set"][index] for index in selected]), device)
    generated = torch.tensor([item["caption_source"] != "human" for item in batch["metadata"]], device=device)
    assert not batch["target_valid_mask"][generated].any()
    assert not batch["corruption_mask"][generated].any()
    assert batch["target_valid_mask"][~generated].any()

    model = build_finetune_model(args, bundle["num_classes"]).to(device).train()
    optimizer = build_optimizer(args, model)
    tracked = {"image_patch": model.base_model.visual.conv1.weight,
        "text_projection": model.base_model.text_projection,
        "nac_vocabulary": model.nac.vocab_projection.weight,
        "classifier": model.classifier.weight}
    before = {name: value.detach().clone() for name, value in tracked.items()}
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    gradients = {name: value.grad is not None and bool(torch.isfinite(value.grad).all())
        and bool(value.grad.abs().sum() > 0) for name, value in tracked.items()}
    assert all(gradients.values()), gradients
    optimizer.step()
    updates = {name: not torch.equal(value.detach(), before[name]) for name, value in tracked.items()}
    assert all(updates.values()), updates
    report = {"benchmark_training_completed": False, "real_diffusion_or_qwen_called": False,
        "dataset": "RSTPReid", "device": str(device), "torch": str(torch.__version__),
        "augmentation_signature": bundle["dataset"].augmentation_signature,
        "batch_sources": dict(Counter(item["caption_source"] for item in batch["metadata"])),
        "source_images": [item["source_image_id"] for item in batch["metadata"]],
        "samples_per_identity": dict(Counter(batch["pids"].cpu().tolist())),
        "augmented_target_tokens": int(batch["target_valid_mask"][generated].sum()),
        "human_target_tokens": int(batch["target_valid_mask"][~generated].sum()),
        "losses": {key: float(value.detach()) for key, value in output["losses"].items()},
        "total_loss": float(output["loss"].detach()), "finite_nonzero_gradients": gradients,
        "parameters_updated": updates, "elapsed_seconds": round(time.monotonic() - started, 2)}
    destination = Path(options.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
