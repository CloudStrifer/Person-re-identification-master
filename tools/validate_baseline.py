"""One real-data optimizer step and a validation subset, never a benchmark run."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--dataset_name", default="RSTPReid")
    parser.add_argument("--clip_weights", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default=".recovery/stage1_validation/real_data_step.json")
    options = parser.parse_args(argv)
    import torch
    from torch.utils.data import DataLoader, Subset
    from datasets.build import build_loaders
    from model import build_finetune_model
    from processor.baseline import move_batch, resolve_device, set_seed
    from solver import build_optimizer
    from utils.metrics import Evaluator
    from utils.options import get_args

    if options.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(options.threads)
    args = get_args(["--root_dir", options.root_dir, "--dataset_name", options.dataset_name,
        "--clip_weights", options.clip_weights, "--device", options.device,
        "--batch_size", "2", "--test_batch_size", "2", "--num_workers", "0"])
    started = time.monotonic()
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = build_loaders(args)
    model = build_finetune_model(args, bundle["num_classes"]).to(device)
    optimizer = build_optimizer(args, model)
    model.train()
    batch = move_batch(next(iter(bundle["train_loader"])), device)
    tracked = {"image_patch": model.base_model.visual.conv1.weight,
               "text_projection": model.base_model.text_projection,
               "identity_classifier": model.classifier.weight}
    before = {key: value.detach().clone() for key, value in tracked.items()}
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    if not torch.isfinite(output["loss"]):
        raise FloatingPointError("Non-finite real-data loss")
    output["loss"].backward()
    checks = {key: value.grad is not None and bool(torch.isfinite(value.grad).all()) and bool(value.grad.abs().sum() > 0) for key, value in tracked.items()}
    if not all(checks.values()):
        raise AssertionError(f"Missing or invalid gradients: {checks}")
    optimizer.step()
    updated = {key: not torch.equal(value.detach(), before[key]) for key, value in tracked.items()}
    if not all(updated.values()):
        raise AssertionError(f"Parameters did not update: {updated}")
    model.eval()
    with torch.inference_mode():
        image_features = model.encode_image(batch["images"])
        text_features = model.encode_text(batch["caption_ids"], batch["attention_mask"])
        torch.testing.assert_close(image_features[:1], model.encode_image(batch["images"][:1]), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(text_features[:1], model.encode_text(batch["caption_ids"][:1], batch["attention_mask"][:1]), rtol=1e-4, atol=1e-5)
    metrics = None
    subset_counts = None
    if bundle["val_img_loader"] is not None:
        source = bundle["dataset"].val
        selected = list(dict.fromkeys(source["image_pids"]))[:2]
        def first_indices(values, limit):
            counts = Counter()
            result = []
            for index, identity in enumerate(values):
                if identity in selected and counts[identity] < limit:
                    counts[identity] += 1
                    result.append(index)
            return result
        image_indices = first_indices(source["image_pids"], 2)
        text_indices = first_indices(source["caption_pids"], 2)
        images = DataLoader(Subset(bundle["val_img_loader"].dataset, image_indices), batch_size=2)
        texts = DataLoader(Subset(bundle["val_txt_loader"].dataset, text_indices), batch_size=2)
        metrics = Evaluator(images, texts).evaluate(model)
        subset_counts = {"identities": len(selected), "gallery_images": len(image_indices), "text_queries": len(text_indices)}
    report = {"validation_level": "one real-data CPU/FP32 optimizer step and small official-validation subset" if device.type == "cpu" else "one real-data optimizer step and small official-validation subset",
        "benchmark_training_completed": False, "dataset": args.dataset_name,
        "device": str(device), "torch": torch.__version__, "precision": "fp32",
        "image_shape": list(batch["images"].shape), "text_shape": list(batch["caption_ids"].shape),
        "training_identities": bundle["num_classes"], "training_pairs": len(bundle["train_loader"].dataset),
        "losses": {key: float(value.detach()) for key, value in output["losses"].items()},
        "total_loss": float(output["loss"].detach()), "finite_nonzero_gradients": checks,
        "parameters_updated": updated, "independent_sample_encoding": "passed",
        "validation_subset": subset_counts, "subset_sanity_metrics_not_benchmark": metrics,
        "elapsed_seconds": round(time.monotonic() - started, 2)}
    destination = Path(options.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
