# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Exact, prefix-safe checkpoints for model initialization and training resume."""
from collections import OrderedDict
from collections.abc import Mapping
import logging
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def _unwrap(model):
    while isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    return model


def _plain(value):
    """Keep checkpoints compatible with torch.load(weights_only=True)."""
    if hasattr(value, "__dict__") and not isinstance(value, torch.Tensor):
        value = vars(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported checkpoint metadata type: {type(value).__name__}")


def capture_rng_state():
    numpy_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3],
                  "cached_gaussian": numpy_state[4]},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available():
            logging.getLogger(__name__).info("CUDA RNG state retained in checkpoint but current device is CPU")
        else:
            if len(state["cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count differs from the resumed checkpoint")
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, model, optimizer=None, scheduler=None, *,
                    model_config=None, train_config=None, label_map=None,
                    epoch=0, global_step=0, **extra):
    path = Path(path)
    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be nonnegative")
    payload = {
        "format_version": 1,
        "model": _unwrap(model).state_dict(),
        "model_config": _plain(model_config or {}),
        "train_config": _plain(train_config or {}),
        "label_map": _plain(label_map or {}),
        "epoch": int(epoch), "global_step": int(global_step),
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    conflicts = set(extra).intersection(payload)
    if conflicts:
        raise ValueError(f"Extra checkpoint fields overwrite reserved keys: {sorted(conflicts)}")
    payload.update({key: _plain(value) for key, value in extra.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def read_checkpoint(path, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must be a model state-dict or a checkpoint dictionary")
    if "model" not in payload:
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            payload = {"model": payload, "format_version": 0}
        else:
            raise ValueError("Checkpoint has no 'model' state-dict")
    if not isinstance(payload["model"], Mapping):
        raise ValueError("Checkpoint 'model' must be a state-dict")
    return dict(payload)


def strip_prefix_if_present(state_dict, prefix="module."):
    """Remove only leading wrappers, preserving keys that have no prefix."""
    result = OrderedDict()
    for key, value in state_dict.items():
        normalized = key
        while normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
        if normalized in result:
            raise ValueError(f"Checkpoint keys collide after prefix removal: {normalized}")
        result[normalized] = value
    return result


def _load_weights(model, loaded_state_dict, allow_classifier_mismatch=False, except_keys=None,
                  skip_classifier=False, initialize_new_modules=False):
    model = _unwrap(model)
    loaded = strip_prefix_if_present(loaded_state_dict)
    current = model.state_dict()
    initialized = set()
    if initialize_new_modules:
        for prefix in ("visual_enhance.", "textual_enhance.", "nac."):
            expected_keys = {key for key in current if key.startswith(prefix)}
            if expected_keys and not any(key.startswith(prefix) for key in loaded):
                initialized.update(expected_keys)
    excluded = tuple(except_keys or ())
    skipped = []
    for key in list(loaded):
        if any(part in key for part in excluded) or (skip_classifier and key.startswith("classifier.")):
            skipped.append(key)
            del loaded[key]
        elif allow_classifier_mismatch and key.startswith("classifier.") and (
                key not in current or loaded[key].shape != current[key].shape):
            skipped.append(key)
            del loaded[key]
    missing = sorted(set(current) - set(loaded))
    unexpected = sorted(set(loaded) - set(current))
    mismatched = [key for key in current.keys() & loaded.keys()
                  if not isinstance(loaded[key], torch.Tensor) or loaded[key].shape != current[key].shape]
    def permitted_missing(key):
        return (key in initialized or ((allow_classifier_mismatch or skip_classifier) and key.startswith("classifier."))
                or any(part in key for part in excluded))
    disallowed_missing = [key for key in missing if not permitted_missing(key)]
    if disallowed_missing or unexpected or mismatched:
        raise RuntimeError("Checkpoint/model mismatch: "
                           f"missing={disallowed_missing}, unexpected={unexpected}, shape_mismatch={sorted(mismatched)}")
    if not set(current).intersection(loaded):
        raise RuntimeError("Checkpoint contains no loadable model parameters")
    model.load_state_dict(loaded, strict=not missing)
    return {"loaded_keys": sorted(set(current).intersection(loaded)),
            "missing_keys": missing, "unexpected_keys": unexpected,
            "skipped_keys": sorted(skipped), "initialized_keys": sorted(initialized)}


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, *,
                    resume=False, restore_rng=False, allow_classifier_mismatch=False,
                    skip_classifier=False, initialize_new_modules=False, map_location="cpu"):
    payload = read_checkpoint(path, map_location)
    if resume:
        if allow_classifier_mismatch or skip_classifier or initialize_new_modules:
            raise ValueError("Exact resume cannot skip classifier parameters or initialize new modules")
        required = {"rng_state", "epoch", "global_step", "train_config", "model_config", "label_map"}
        if optimizer is not None:
            required.add("optimizer")
        if scheduler is not None:
            required.add("scheduler")
        missing = required - payload.keys()
        if payload.get("format_version") != 1 or missing:
            raise ValueError(f"Exact resume requires format_version=1 and complete state; missing={sorted(missing)}")
    if model is not None:
        payload["load_report"] = _load_weights(model, payload["model"], allow_classifier_mismatch,
                                               skip_classifier=skip_classifier,
                                               initialize_new_modules=initialize_new_modules)
    # Construct the scheduler before loading either state; load optimizer last so
    # the current group learning rates are restored exactly.
    if resume:
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        if "rng_state" not in payload:
            raise ValueError("Checkpoint contains no RNG state")
        restore_rng_state(payload["rng_state"])
    return payload


# Compatibility facade for old scripts; matching is now exact, never by suffix.
class Checkpointer:
    def __init__(self, model, optimizer=None, scheduler=None, save_dir="",
                 save_to_disk=None, logger=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.save_dir = save_dir
        self.save_to_disk = True if save_to_disk is None else save_to_disk
        self.logger = logger or logging.getLogger(__name__)

    def save(self, name, **kwargs):
        if not self.save_dir or not self.save_to_disk:
            return None
        return save_checkpoint(Path(self.save_dir) / f"{name}.pth", self.model,
                               self.optimizer, self.scheduler, **kwargs)

    def load(self, f=None):
        if not f:
            return {}
        return load_checkpoint(f, self.model)

    def resume(self, f=None):
        if not f:
            raise FileNotFoundError("An explicit checkpoint is required for resume")
        return load_checkpoint(f, self.model, self.optimizer, self.scheduler,
                               resume=True, restore_rng=True)

    def _load_file(self, path):
        return read_checkpoint(path)

    def _load_model(self, checkpoint, except_keys=None):
        return _load_weights(self.model, checkpoint["model"], except_keys=except_keys)


def check_key(key, except_keys):
    return any(part in key for part in (except_keys or ()))


def load_state_dict(model, loaded_state_dict, except_keys=None):
    return _load_weights(model, loaded_state_dict, except_keys=except_keys)


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    loaded = strip_prefix_if_present(loaded_state_dict)
    missing = [key for key in model_state_dict if key not in loaded and not check_key(key, except_keys)]
    unexpected = set(loaded) - set(model_state_dict)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint keys differ: missing={missing}, unexpected={sorted(unexpected)}")
    for key, value in loaded.items():
        if not check_key(key, except_keys):
            if value.shape != model_state_dict[key].shape:
                raise RuntimeError(f"Checkpoint shape mismatch for {key}")
            model_state_dict[key] = value
