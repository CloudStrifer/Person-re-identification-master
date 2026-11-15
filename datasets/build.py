"""Single-dataset baseline loaders with explicit validation/test separation."""

import random
from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from .bases import ImageDataset, ImageTextDataset, TextDataset
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .sampler import RandomIdentitySampler
from .synthperson import Synth_PEDES
from utils.simple_tokenizer import SimpleTokenizer


DATASETS = {"CUHK-PEDES": CUHKPEDES, "ICFG-PEDES": ICFGPEDES,
            "RSTPReid": RSTPReid, "Synth_PEDES": Synth_PEDES}
_ALIASES = {name.lower().replace("_", "-"): name for name in DATASETS}
_ALIASES.update({"synth-pedes": "Synth_PEDES", "synthpedes": "Synth_PEDES",
                 "rstp-reid": "RSTPReid"})


def canonical_dataset_name(name):
    try:
        return _ALIASES[str(name).lower().replace("_", "-")]
    except KeyError as error:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {list(DATASETS)}") from error


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    # Generation dependencies are deliberately absent from the baseline import path.
    import torchvision.transforms as T
    height, width = img_size
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    transforms = [T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC)]
    if is_train and aug:
        transforms += [T.RandomHorizontalFlip(0.5), T.Pad(10), T.RandomCrop((height, width))]
    transforms += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    if is_train and aug:
        transforms.append(T.RandomErasing(scale=(0.02, 0.4), value=0.0))
    return T.Compose(transforms)


def collate(batch):
    """Stack model inputs while preserving original metadata on the CPU."""
    if not batch:
        raise ValueError("Cannot collate an empty training batch")
    keys = set(batch[0])
    if any(set(item) != keys for item in batch):
        raise ValueError("All samples in a batch must have the same fields")
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        result[key] = values if key == "metadata" else default_collate(values)
    return result


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(args, training=True, tokenizer=None, transforms=None):
    """Build only the selected dataset. Empty official validation means ``None``.

    ``tokenizer``/``transforms`` injection supports small, offline integration tests;
    production defaults always use the actual CLIP BPE and image preprocessing.
    """
    if getattr(args, "distributed", False):
        raise NotImplementedError("Stage 1 supports a single process; distributed sampling is a later stage.")
    enabled = [name for name in ("MLM",) if getattr(args, name, False)]
    if enabled:
        raise NotImplementedError("Stage 1 baseline requires these options disabled: " + ", ".join(enabled))
    if (training and (getattr(args, "use_llm_augment", False) or getattr(args, "use_generated", False))
            and not getattr(args, "augmentation_manifest", None)):
        raise ValueError("--use_llm_augment/--use_generated require --augmentation_manifest "
                         "pointing to a prepared offline manifest")
    name = canonical_dataset_name(args.dataset_name)
    dataset = DATASETS[name](root=args.root_dir, text_length=getattr(args, "text_length", 77))
    if training and (getattr(args, "use_llm_augment", False) or getattr(args, "use_generated", False)):
        from .augmentation import apply_augmentation
        apply_augmentation(dataset, getattr(args, "augmentation_manifest", None),
                           use_llm_augment=getattr(args, "use_llm_augment", False),
                           use_generated=getattr(args, "use_generated", False), dataset_name=name)
    if training and not dataset.train_records:
        raise ValueError(f"No original training captions found in {dataset.anno_path}")
    if not training and not dataset.test["captions"]:
        raise ValueError(f"No test captions found in {dataset.anno_path}")
    tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer(getattr(args, "bpe_path", None))
    image_size = tuple(getattr(args, "img_size", (384, 128)))
    train_transform = (transforms.get("train") if isinstance(transforms, Mapping) else transforms)
    eval_transform = (transforms.get("eval") if isinstance(transforms, Mapping) else transforms)
    if train_transform is None:
        train_transform = build_transforms(image_size, getattr(args, "img_aug", False), True)
    if eval_transform is None:
        eval_transform = build_transforms(image_size, False, False)
    batch_size, test_batch_size = getattr(args, "batch_size", 64), getattr(args, "test_batch_size", 512)
    workers, seed = getattr(args, "num_workers", 0), getattr(args, "seed", 42)
    if batch_size < 1 or test_batch_size < 1 or workers < 0:
        raise ValueError("Batch sizes must be positive and num_workers must be nonnegative")
    generator = torch.Generator().manual_seed(seed)
    common = {"num_workers": workers, "worker_init_fn": seed_worker,
              "pin_memory": str(getattr(args, "device", "cpu")).startswith("cuda")}
    train_set = None
    train_loader = None
    if training:
        train_set = ImageTextDataset(dataset.train_records, train_transform,
                                     text_length=getattr(args, "text_length", 77), tokenizer=tokenizer,
                                     use_nac=getattr(args, "use_nac", False),
                                     nac_corruption_rate=getattr(args, "nac_corruption_rate", 0.15))
        sampler_name = getattr(args, "sampler", "auto")
        if sampler_name == "auto":
            sampler_name = "identity" if getattr(args, "use_cdfp", False) else "random"
        if getattr(args, "use_cdfp", False) and sampler_name != "identity":
            raise ValueError("CDFP requires identity sampling to guarantee other positive and negative samples")
        if sampler_name == "identity":
            sampler = RandomIdentitySampler(dataset.train, batch_size,
                                            getattr(args, "num_instance", 4), seed=seed)
        elif sampler_name == "random":
            sampler = None
        else:
            raise ValueError(f"Unknown sampler {sampler_name!r}; choose random or identity")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=sampler is None,
                                  sampler=sampler, collate_fn=collate, generator=generator,
                                  drop_last=sampler is not None, **common)

    def evaluation_loaders(split, seed_offset):
        if not split["captions"]:
            return None, None
        image_set = ImageDataset(split["image_pids"], split["img_paths"], eval_transform)
        text_set = TextDataset(split["caption_pids"], split["captions"],
                               text_length=getattr(args, "text_length", 77), tokenizer=tokenizer)
        # Even shuffle=False iterators draw a worker/base seed. Give validation
        # and test their own generators so evaluation never advances model RNG.
        image_generator = torch.Generator().manual_seed(seed + seed_offset)
        text_generator = torch.Generator().manual_seed(seed + seed_offset + 1)
        return (DataLoader(image_set, batch_size=test_batch_size, shuffle=False,
                           generator=image_generator, **common),
                DataLoader(text_set, batch_size=test_batch_size, shuffle=False,
                           generator=text_generator, **common))

    val_images, val_texts = evaluation_loaders(dataset.val, 10000)
    test_images, test_texts = evaluation_loaders(dataset.test, 20000)
    return {"train_loader": train_loader, "val_img_loader": val_images, "val_txt_loader": val_texts,
            "test_img_loader": test_images, "test_txt_loader": test_texts,
            "num_classes": len(dataset.label_map), "label_map": dataset.label_map,
            "dataset": dataset, "train_set": train_set, "tokenizer": tokenizer,
            "token_ids": {"sot": tokenizer.encoder["<|startoftext|>"],
                          "eot": tokenizer.encoder["<|endoftext|>"],
                          "mask": tokenizer.encoder.get("<|mask|>")}, "generator": generator}


def build_dataloader(args, tranforms=None):
    """Compatibility tuple API; never substitute test data for validation."""
    training = getattr(args, "training", True)
    loaders = build_loaders(args, training=training, transforms=tranforms)
    if training:
        return (loaders["train_loader"], loaders["val_img_loader"],
                loaders["val_txt_loader"], loaders["num_classes"])
    return loaders["test_img_loader"], loaders["test_txt_loader"], loaders["num_classes"]


def build_zero_shot_loader(*args, **kwargs):
    raise NotImplementedError(
        "The legacy implicit three-target loader has been retired. Use build_loaders "
        "with an explicit source/target dataset; full transfer experiments are stage 5."
    )


def build_filter_loader(*args, **kwargs):
    raise NotImplementedError("Offline filtering is recovery stage 4, separate from baseline training.")
