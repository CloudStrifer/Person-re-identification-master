"""Shared baseline training/evaluation without implicit downloads."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import hashlib
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch

from datasets.build import build_loaders, canonical_dataset_name
from model import build_finetune_model
from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from utils.metrics import Evaluator
from utils.options import build_parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use --device cpu or install a compatible CUDA PyTorch build.")
    return torch.device(request)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _explicit_destinations(argv):
    actions = build_parser()._option_string_actions
    return {actions[option].dest for token in argv if (option := token.split("=", 1)[0]) in actions}


def _resume_configuration(args, argv, payload):
    if payload.get("format_version") != 1 or not payload.get("train_config"):
        raise ValueError("Exact resume requires a recovered checkpoint with train_config; use --finetune for initialization.")
    saved = payload["train_config"]
    runtime = {"device", "test_batch_size", "log_period", "output_dir", "root_dir", "bpe_path", "augmentation_manifest"}
    explicit = _explicit_destinations(argv)
    for key in explicit - runtime - {"resume", "resume_ckpt_file", "check_resources"}:
        if key in saved:
            current, previous = getattr(args, key), saved[key]
            if key == "sampler" and getattr(args, "_sampler_requested", None) == "auto":
                current = previous
            if isinstance(current, (list, tuple)) and isinstance(previous, (list, tuple)):
                current, previous = list(current), list(previous)
            if current != previous:
                raise ValueError(f"Exact resume cannot change {key}: {previous!r} -> {current!r}. Start a new --finetune run instead.")
    merged = {**vars(args), **saved}
    for key in explicit & runtime:
        merged[key] = getattr(args, key)
    checkpoint_directory = Path(args.resume_ckpt_file).resolve().parent
    if "output_dir" in explicit and Path(args.output_dir).resolve() != checkpoint_directory:
        raise ValueError("Resume must use the checkpoint's run directory (including its best.pth); use --finetune for a new run.")
    merged["output_dir"] = str(checkpoint_directory)
    merged.update(resume=True, resume_ckpt_file=args.resume_ckpt_file, finetune="", initialize_new_modules=False, check_resources=False)
    return argparse.Namespace(**merged)


def _logger(directory, filename):
    logger = logging.getLogger("NCHNet")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / filename, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _close_logger(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _config(args):
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _initialization_metadata(args, resumed, initial):
    if resumed is not None:
        return resumed.get("initialization")
    if args.finetune:
        return {"kind": "retrieval_checkpoint", "checkpoint": str(Path(args.finetune).resolve()),
                "checkpoint_sha256": _sha256(args.finetune),
                "source_dataset": initial.get("model_config", {}).get("dataset_name"),
                "source_epoch": initial.get("epoch"), "source_global_step": initial.get("global_step"),
                "classifier_reset": True, "optimizer_reset": True}
    checkpoint = getattr(args, "clip_checkpoint", "")
    return {"kind": "clip", "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else args.pretrain_choice,
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint and Path(checkpoint).is_file() else None}


def _build_model(args, num_classes, payload=None):
    config = payload.get("model_config", {}) if payload else {}
    if payload and "backbone_config" not in config:
        raise ValueError("Checkpoint has no backbone_config; initialize from local CLIP weights instead.")
    if payload:
        backbone = config["backbone_config"]
        resolution = backbone["image_resolution"]
        resolution = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
        if tuple(args.img_size) != resolution or args.text_length != backbone["context_length"]:
            raise ValueError(f"Checkpoint expects image size {resolution} and text length {backbone['context_length']}; set matching input options.")
    return build_finetune_model(args, num_classes, backbone_cfg=config.get("backbone_config"), load_pretrained=payload is None)


def run_training(args, argv=None):
    """Train one dataset. Validation chooses best; test is never used here."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The recovered trainer currently supports one process/device; launch python train.py rather than torchrun.")
    payload = read_checkpoint(args.resume_ckpt_file) if args.resume else None
    if payload:
        args = _resume_configuration(args, argv or [], payload)
    if args.pretrain:
        if args.mode != "pretrain":
            raise ValueError("--pretrain is a source dataset alias; specify --mode pretrain.")
        args.dataset_name = args.pretrain
    if args.mode == "finetune" and not args.finetune and not args.resume:
        raise ValueError("Finetune mode requires --finetune CHECKPOINT; use baseline to start from CLIP.")
    if args.finetune and args.mode != "finetune":
        raise ValueError("--finetune weight initialization requires --mode finetune.")
    requested = [key for key in ("MLM", "nam", "use_dfp") if getattr(args, key, False)]
    if requested:
        raise NotImplementedError(f"These modules have not reached their recovery stage: {requested}")
    device = resolve_device(args.device)
    if args.precision == "amp" and device.type != "cuda":
        raise ValueError("--precision amp is for CUDA; use fp32 on CPU.")
    set_seed(args.seed)
    bundle = build_loaders(args, training=True)
    if not len(bundle["train_loader"]):
        raise ValueError("The training loader has no batches; check data and sampler settings.")
    label_map = {str(key): value for key, value in bundle["label_map"].items()}
    data_signature = {"annotation_sha256": hashlib.sha256(Path(bundle["dataset"].anno_path).read_bytes()).hexdigest(), "training_pairs": len(bundle["train_loader"].dataset)}
    if getattr(bundle["dataset"], "augmentation_signature", None) is not None:
        data_signature["augmentation"] = bundle["dataset"].augmentation_signature
    if payload and payload.get("label_map") != label_map:
        raise ValueError("Training identities differ from the checkpoint; exact resume would change classifier labels.")
    if payload and payload.get("data_signature") != data_signature:
        raise ValueError("Annotation contents or training sample count changed; start a new run instead of resuming.")
    initial = payload or (read_checkpoint(args.finetune) if args.finetune else None)
    initialization = _initialization_metadata(args, payload, initial)
    model = _build_model(args, bundle["num_classes"], initial).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "amp")
    start_epoch, global_step, best_r1, best_epoch = 1, 0, None, None
    restored = None
    if payload:
        restored = load_checkpoint(args.resume_ckpt_file, model, optimizer, scheduler, resume=True, restore_rng=True)
        start_epoch, global_step = int(restored["epoch"]) + 1, int(restored["global_step"])
        best_r1, best_epoch = restored.get("best_r1"), restored.get("best_epoch")
        if restored.get("scaler"):
            scaler.load_state_dict(restored["scaler"])
        if bundle.get("generator") is not None and restored.get("dataloader_rng_state") is not None:
            bundle["generator"].set_state(restored["dataloader_rng_state"])
    elif args.finetune:
        restored = load_checkpoint(args.finetune, model, skip_classifier=True,
            initialize_new_modules=getattr(args, "initialize_new_modules", False))
    directory = Path(args.output_dir)
    if str(directory) == "logs" and not args.resume:
        directory = directory / args.dataset_name / f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}"
    directory = directory.resolve()
    if not args.resume and (directory / "last.pth").exists():
        raise FileExistsError(f"Run exists: {directory}; choose a new --output_dir or resume it.")
    directory.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(directory)
    args.resolved_device = str(device)
    logger = _logger(directory, "train_log.txt")
    try:
        config = _config(args)
        (directory / "configs.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model_config = {**config, "backbone_config": model.backbone_config, "num_classes": bundle["num_classes"]}
        logger.info("Device=%s precision=%s dataset=%s identities=%d pairs=%d", device, args.precision, args.dataset_name, bundle["num_classes"], len(bundle["train_loader"].dataset))
        if restored:
            logger.info("Loaded %d checkpoint parameters; skipped=%s", len(restored["load_report"]["loaded_keys"]), restored["load_report"]["skipped_keys"])
            if restored["load_report"].get("initialized_keys"):
                logger.info("Explicitly initialized new parameters: %s", restored["load_report"]["initialized_keys"])
        if payload and payload["train_config"].get("resolved_device", str(device)) != str(device):
            logger.info("Resuming on a different device; optimizer state is restored but bitwise numerical equivalence is not expected.")
        validation = Evaluator(bundle["val_img_loader"], bundle["val_txt_loader"]) if bundle["val_img_loader"] is not None else None
        selection_rule = "official_validation_rank1_earliest_tie" if validation else "fixed_schedule_final_checkpoint"
        logger.info("Selection: %s", "official validation Rank-1; earliest tie" if validation else "fixed schedule; final checkpoint")
        if start_epoch > args.num_epoch:
            raise ValueError("Checkpoint already completed its schedule; no epochs remain to resume.")
        for epoch in range(start_epoch, args.num_epoch + 1):
            sampler = bundle["train_loader"].sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch - 1)
            model.train()
            sums, count = {}, 0
            epoch_lr = [group["lr"] for group in optimizer.param_groups]
            for iteration, cpu_batch in enumerate(bundle["train_loader"], start=1):
                batch = move_batch(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast("cuda", dtype=torch.float16) if args.precision == "amp" else nullcontext()
                with context:
                    output = model(batch)
                    loss = output["loss"]
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, iteration {iteration}.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                size = batch["images"].shape[0]
                values = {"loss": loss.detach(), **{f"loss_{key}": value.detach() for key, value in output["losses"].items()},
                    **{f"metric_{key}": value.detach() for key, value in output.get("metrics", {}).items() if torch.is_tensor(value) and value.numel() == 1}}
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.0) + float(value) * size
                count += size
                if iteration % args.log_period == 0:
                    logger.info("Epoch %d/%d batch %d/%d loss=%.6f", epoch, args.num_epoch, iteration, len(bundle["train_loader"]), float(loss.detach()))
            scheduler.step()
            record = {"epoch": epoch, "global_step": global_step, **{key: value / count for key, value in sums.items()}, "learning_rates": epoch_lr}
            improved = False
            if validation and (epoch % args.eval_period == 0 or epoch == args.num_epoch):
                record["validation"] = validation.evaluate(model)
                rank1 = record["validation"]["R1"]
                improved = best_r1 is None or rank1 > best_r1
                if improved:
                    best_r1, best_epoch = rank1, epoch
            generator = bundle.get("generator")
            extras = dict(best_r1=best_r1, best_epoch=best_epoch, scaler=scaler.state_dict(), dataloader_rng_state=generator.get_state() if generator is not None else None, data_signature=data_signature,
                          selection_rule=selection_rule, initialization=initialization)
            checkpoint_args = dict(model_config=model_config, train_config=config, label_map=label_map, epoch=epoch, global_step=global_step, **extras)
            if improved:
                save_checkpoint(directory / "best.pth", model, optimizer, scheduler, **checkpoint_args)
            save_checkpoint(directory / "last.pth", model, optimizer, scheduler, **checkpoint_args)
            with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Epoch %d loss=%.6f validation=%s", epoch, record["loss"], record.get("validation", "not used"))
        selected = directory / ("best.pth" if validation else "last.pth")
        logger.info("Complete. Selected checkpoint: %s. Test evaluation is a separate command.", selected)
        return {"output_dir": str(directory), "last_checkpoint": str(directory / "last.pth"), "selected_checkpoint": str(selected), "best_r1": best_r1, "global_step": global_step}
    finally:
        _close_logger(logger)


def evaluate_checkpoint(checkpoint, *, dataset_name=None, root_dir=None, bpe_path=None, device="auto", batch_size=None, num_workers=None, output_file=None):
    payload = read_checkpoint(checkpoint)
    if payload.get("format_version") != 1 or not payload.get("model_config"):
        raise ValueError("Evaluation requires a recovered checkpoint containing model_config.")
    args = argparse.Namespace(**payload["model_config"])
    source_dataset = canonical_dataset_name(args.dataset_name)
    for key, value in {"dataset_name": dataset_name, "root_dir": root_dir, "bpe_path": bpe_path, "test_batch_size": batch_size, "num_workers": num_workers}.items():
        if value is not None:
            setattr(args, key, value)
    args.dataset_name = canonical_dataset_name(args.dataset_name)
    args.device, args.training = device, False
    target_device = resolve_device(device)
    bundle = build_loaders(args, training=False)
    if bundle["test_img_loader"] is None:
        raise ValueError("The selected dataset has no test split.")
    model = _build_model(args, int(payload["model_config"]["num_classes"]), payload).to(target_device)
    load_checkpoint(checkpoint, model)
    metrics = Evaluator(bundle["test_img_loader"], bundle["test_txt_loader"]).evaluate(model)
    selection_rule = payload.get("selection_rule", "unspecified")
    matches_selection = None
    if selection_rule == "official_validation_rank1_earliest_tie":
        matches_selection = payload.get("epoch") == payload.get("best_epoch")
    elif selection_rule == "fixed_schedule_final_checkpoint":
        matches_selection = payload.get("epoch") == payload.get("train_config", {}).get("num_epoch")
    result = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": _sha256(checkpoint),
              "dataset": args.dataset_name, "split": "test", "metrics": metrics,
              "source_dataset": source_dataset,
              "protocol": "in_domain" if source_dataset == args.dataset_name else "direct_transfer",
              "source_epoch": payload.get("epoch"), "source_global_step": payload.get("global_step"),
              "source_training_mode": payload["model_config"].get("mode"),
              "source_selection_rule": selection_rule,
              "selection": {"training_rule": selection_rule, "best_validation_epoch": payload.get("best_epoch"),
                            "best_validation_r1": payload.get("best_r1"), "checkpoint_matches_selection": matches_selection},
              "initialization": payload.get("initialization"), "seed": payload["model_config"].get("seed"),
              "model_options": {key: bool(payload["model_config"].get(key, False)) for key in
                                ("use_nac", "use_a2dfp", "use_cdfp", "use_llm_augment", "use_generated")},
              "evaluation_scope": "provided_test_split",
              "test_annotation_sha256": _sha256(bundle["dataset"].anno_path),
              "evaluation": {"device": str(target_device), "precision": "fp32",
                             "batch_size": args.test_batch_size, "num_workers": args.num_workers,
                             "num_test_images": len(bundle["dataset"].test["image_pids"]),
                             "num_test_captions": len(bundle["dataset"].test["caption_pids"])}}
    if output_file is not None:
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
