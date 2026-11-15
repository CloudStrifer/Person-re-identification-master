"""Validate and merge a prepared, offline augmentation manifest into training.

This module uses only the standard library. Reading manifests never invokes a
generator, downloads weights, or substitutes an original image's captions for
an uncaptioned generated image.
"""

import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath


_DATASET_NAMES = {"CUHK-PEDES", "ICFG-PEDES", "RSTPReid", "Synth_PEDES"}


def _relative_path(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Augmentation {field} must be a nonempty relative path")
    normalized = value.replace("\\", "/")
    if (Path(normalized).is_absolute() or PureWindowsPath(normalized).drive
            or any(part in ("", ".", "..") for part in normalized.split("/"))
            or ":" in normalized):
        raise ValueError(f"Augmentation {field} must be a contained relative path: {value!r}")
    return normalized


def _caption_key(caption):
    return " ".join(caption.split())


def _captions(record, index):
    captions = record.get("captions")
    if (not isinstance(captions, list) or len(captions) != 2
            or any(not isinstance(caption, str) or not caption.strip() for caption in captions)):
        raise ValueError(f"Augmentation record #{index} requires exactly two nonempty captions; "
                         "generated images need their own captions before training")
    if len({_caption_key(caption) for caption in captions}) != 2:
        raise ValueError(f"Duplicate source-caption in augmentation record #{index}")
    return captions


def _file_key(path):
    """Recognize symlinks, case aliases, and hard links without reading images."""
    path = Path(path).resolve()
    try:
        status = path.stat()
    except FileNotFoundError:
        return ("path", str(path))
    return ("inode", status.st_dev, status.st_ino) if status.st_ino else ("path", str(path))


def apply_augmentation(dataset, manifest_path=None, *, use_llm_augment=False,
                       use_generated=False, dataset_name=None):
    """Append enabled, validated records while preserving the original splits.

    Image records enter training only after offline filtering marks them
    ``selected: true``. Disabled record kinds are left unread so a text-only
    run can use a manifest whose image-generation stages are still incomplete.
    Validation is atomic: a bad manifest cannot partially mutate the dataset.
    """
    if not (use_llm_augment or use_generated):
        return dataset
    if not manifest_path:
        raise ValueError("--use_llm_augment/--use_generated require --augmentation_manifest "
                         "pointing to a prepared offline manifest")
    if getattr(dataset, "augmentation_signature", None) is not None:
        raise ValueError("Augmentation has already been applied to this dataset")
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing --augmentation_manifest: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid augmentation manifest JSON: {manifest_path}") from error
    if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
            or manifest["version"] != 1 or not isinstance(manifest.get("records"), list)):
        raise ValueError("Augmentation manifest requires version: 1 and a records list")
    expected_name = dataset_name or ("Synth_PEDES" if dataset.folder == "Synth-Pedes" else dataset.folder)
    if (not isinstance(manifest.get("dataset"), str) or manifest["dataset"] not in _DATASET_NAMES
            or manifest["dataset"] != expected_name):
        raise ValueError(f"Augmentation manifest dataset must be {expected_name!r}")

    sources = {}
    for source in dataset.train_annos:
        source_id = source["source_image_id"]
        previous = sources.setdefault(source_id, source)
        if previous["source_pid"] != source["source_pid"]:
            raise ValueError(f"Ambiguous original training source: {source_id}")
    source_rows = {row["source_image_id"]: row for row in dataset.train_records}
    all_original_files = {_file_key(record["img_path"])
                          for records in (dataset.train_annos, dataset.val_annos, dataset.test_annos)
                          for record in records}
    seen_captions = {(row["source_image_id"], _caption_key(row["caption"]))
                     for row in dataset.train_records}
    seen_generated = set()
    image_id = max((row["image_id"] for row in dataset.train_records), default=-1) + 1
    additions = []
    text_records = image_records = 0
    for index, record in enumerate(manifest["records"]):
        if not isinstance(record, dict) or record.get("kind") not in ("text", "image"):
            raise ValueError(f"Augmentation record #{index} requires kind 'text' or 'image'")
        kind = record["kind"]
        if ((kind == "text" and not use_llm_augment)
                or (kind == "image" and not use_generated)):
            continue
        if kind == "image":
            selected = record.get("selected", False)
            if type(selected) is not bool:
                raise ValueError(f"Augmentation record #{index} selected must be a boolean")
            if not selected:
                continue
        source_id = _relative_path(record.get("source_image_id"), "source_image_id")
        source_pid = record.get("source_pid")
        if source_id not in sources:
            raise ValueError(f"Augmentation source_image_id must reference an original train image: {source_id!r}")
        source = sources[source_id]
        if type(source_pid) is not int or source_pid != source["source_pid"]:
            raise ValueError(f"Augmentation source_pid must match the original train identity for {source_id!r}")
        if record.get("split", "train") != "train":
            raise ValueError(f"Augmentation record #{index} must belong to train")
        captions = _captions(record, index)
        provenance = record.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError(f"Augmentation record #{index} provenance must be an object")
        metadata = {"source_pid": source_pid, "source_image_id": source_id,
                    "pid": dataset.label_map[source_pid], "split": "train",
                    "caption_source": "llm" if kind == "text" else "synthetic",
                    "augmentation_provenance": copy.deepcopy(provenance)}
        if kind == "text":
            metadata.update(image_id=source_rows[source_id]["image_id"], img_path=source["img_path"])
            for caption in captions:
                pair = (source_id, _caption_key(caption))
                if pair in seen_captions:
                    raise ValueError(f"Duplicate source-caption in augmentation record #{index}: {source_id}")
                seen_captions.add(pair)
            text_records += 1
        else:
            relative_path = _relative_path(record.get("image_path"), "image_path")
            path = (manifest_path.parent / relative_path).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError(f"Augmentation image_path escapes the manifest directory: {relative_path!r}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing generated image for augmentation record #{index}: {path}")
            file_key = _file_key(path)
            if file_key in all_original_files:
                raise ValueError(f"Generated image_path references an original dataset image: {path}")
            if file_key in seen_generated:
                raise ValueError(f"Duplicate generated image_path in augmentation record #{index}: {path}")
            seen_generated.add(file_key)
            generated_id = "generated:" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
            metadata.update(image_id=image_id, img_path=str(path), generated_image_id=generated_id)
            image_id += 1
            image_records += 1
        for caption_index, caption in enumerate(captions):
            additions.append({**metadata, "caption": caption,
                              "sample_id": f"augmentation:{kind}:{index}:{caption_index}"})

    dataset.train_records.extend(additions)
    dataset.train = [(row["pid"], row["image_id"], row["img_path"], row["caption"], None)
                     for row in dataset.train_records]
    dataset.augmentation_signature = {"version": 1, "sha256": hashlib.sha256(raw).hexdigest(),
                                      "use_llm_augment": bool(use_llm_augment),
                                      "use_generated": bool(use_generated),
                                      "text_records": text_records, "image_records": image_records,
                                      "added_captions": len(additions)}
    dataset.logger.info("Offline augmentation: %d text records, %d selected generated images, "
                        "%d added captions; training now contains %d captions",
                        text_records, image_records, len(additions), len(dataset.train_records))
    return dataset
