"""Small real-data A2DFP/CDFP checks on CPU or an explicitly selected CUDA build.

Each invocation uses at most one four-sample batch and a tiny official val subset.
No full epoch, test-set selection, or benchmark performance claim is made.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("a2dfp", "cdfp", "joint"), required=True)
    parser.add_argument("--use_nac", action="store_true", help="Also validate training-time NAC")
    parser.add_argument("--nac_corruption_rate", type=float, default=0.15)
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--clip_weights", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default=None)
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
    flags = ["--root_dir", options.root_dir, "--clip_weights", options.clip_weights,
        "--device", options.device, "--batch_size", "4", "--num_instance", "2", "--sampler", "identity",
        "--test_batch_size", "2", "--num_workers", "0", "--use_a2dfp", str(options.variant != "cdfp"),
        "--use_cdfp", str(options.variant != "a2dfp")]
    if options.use_nac:
        flags += ["--use_nac", "--nac_corruption_rate", str(options.nac_corruption_rate)]
    args = get_args(flags)
    start = time.monotonic()
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = build_loaders(args)
    batch = move_batch(next(iter(bundle["train_loader"])), device)
    identity_counts = Counter(batch["pids"].cpu().tolist())
    assert len(identity_counts) == 2 and set(identity_counts.values()) == {2}
    model = build_finetune_model(args, bundle["num_classes"]).to(device)
    optimizer = build_optimizer(args, model)
    head_parameters = {id(p) for group in optimizer.param_groups if group["group_name"] == "heads" for p in group["params"]}
    tracked = {"image_patch": model.base_model.visual.conv1.weight,
               "text_projection": model.base_model.text_projection,
               "identity_classifier": model.classifier.weight}
    if options.use_nac:
        tracked.update(nac_vocabulary=model.nac.vocab_projection.weight,
                       nac_cross_attention=model.nac.cross_attn.in_proj_weight)
    enhancement_parameters = {}
    if args.use_a2dfp:
        enhancement_parameters = {name: value for name, value in model.named_parameters() if name.startswith(("visual_enhance.", "textual_enhance."))}
        assert enhancement_parameters and all(id(p) in head_parameters for p in enhancement_parameters.values())
        for prefix in ("visual_enhance", "textual_enhance"):
            tracked[prefix] = next(p for n, p in enhancement_parameters.items() if n.startswith(prefix))
    before = {key: value.detach().clone() for key, value in tracked.items()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    expected = output["losses"]["sdm"] + output["losses"]["id"]
    if args.use_cdfp:
        expected = expected + args.cdfp_loss_weight * output["losses"]["cdfp"]
    if options.use_nac:
        expected = expected + args.gen_loss_weight * output["losses"]["gen"]
    torch.testing.assert_close(output["loss"], expected)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    gradient_checks = {key: p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0) for key, p in tracked.items()}
    assert all(gradient_checks.values()), gradient_checks
    enhancement_grads = {name: p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0) for name, p in enhancement_parameters.items()}
    assert all(enhancement_grads.values()), enhancement_grads
    optimizer.step()
    updates = {key: not torch.equal(p.detach(), before[key]) for key, p in tracked.items()}
    assert all(updates.values()), updates
    metrics = {key: float(value.detach()) for key, value in output["metrics"].items() if torch.is_tensor(value) and value.numel() == 1}
    model.eval()
    with torch.inference_mode():
        images = model.encode_image(batch["images"])
        texts = model.encode_text(batch["caption_ids"], batch["attention_mask"])
        torch.testing.assert_close(images[:1], model.encode_image(batch["images"][:1]), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(texts[:1], model.encode_text(batch["caption_ids"][:1], batch["attention_mask"][:1]), rtol=1e-4, atol=1e-5)
    source = bundle["dataset"].val
    selected = list(dict.fromkeys(source["image_pids"]))[:2]
    def indices(values):
        counts, result = Counter(), []
        for index, identity in enumerate(values):
            if identity in selected and counts[identity] < 2:
                counts[identity] += 1
                result.append(index)
        return result
    image_indices, text_indices = indices(source["image_pids"]), indices(source["caption_pids"])
    validation = Evaluator(DataLoader(Subset(bundle["val_img_loader"].dataset, image_indices), batch_size=2),
        DataLoader(Subset(bundle["val_txt_loader"].dataset, text_indices), batch_size=2)).evaluate(model)
    report = {"variant": options.variant, "use_nac": options.use_nac,
        "nac_corruption_rate": options.nac_corruption_rate if options.use_nac else None,
        "benchmark_training_completed": False,
        "device": str(device), "precision": "fp32", "torch": str(torch.__version__),
        "dataset": "RSTPReid", "image_shape": list(batch["images"].shape), "text_shape": list(batch["caption_ids"].shape),
        "identities_per_batch": len(identity_counts), "samples_per_identity": sorted(identity_counts.values()),
        "distinct_source_images": len({m["source_image_id"] for m in batch["metadata"]}),
        "losses": {key: float(value.detach()) for key, value in output["losses"].items()}, "total_loss": float(output["loss"].detach()),
        "cdfp_and_other_diagnostics": metrics, "finite_nonzero_gradients": gradient_checks,
        "enhancement_parameters_with_finite_nonzero_gradients": sum(enhancement_grads.values()),
        "enhancement_parameter_tensors": len(enhancement_parameters), "parameters_updated": updates,
        "independent_sample_encoding": "passed", "subset_sanity_metrics_not_benchmark": validation,
        "elapsed_seconds": round(time.monotonic() - start, 2)}
    stage = "stage3" if options.use_nac else "stage2"
    destination = Path(options.output or f".recovery/{stage}_validation/{options.variant}_real_step.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
