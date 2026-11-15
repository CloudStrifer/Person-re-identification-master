"""Resumable offline stage-four workflows. Optional models load only on demand."""

from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile

from augmentation.text import (
    AttributeCache, _ERROR_TEXT, generate_caption_record, generation_provenance,
    normalize_source_image_id, validate_attributes, validate_caption_record,
    validate_source,
)


_DATASETS = {"cuhk-pedes": "CUHK-PEDES", "icfg-pedes": "ICFG-PEDES",
             "rstpreid": "RSTPReid", "rstp-reid": "RSTPReid",
             "synth-pedes": "Synth_PEDES", "synthpedes": "Synth_PEDES"}


def _canonical_dataset(value):
    try:
        return _DATASETS[str(value).lower().replace("_", "-")]
    except KeyError:
        raise ValueError("Unknown augmentation dataset: {}".format(value)) from None


def _read(value):
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    with Path(value).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_write_json(output, payload):
    """Commit one complete JSON value in its destination directory."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         prefix=output.name + ".", dir=output.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _header(payload, key):
    if (not isinstance(payload, dict) or type(payload.get("version")) is not int
            or payload["version"] != 1 or not isinstance(payload.get(key), list)):
        raise ValueError("Expected version: 1 and a {} list".format(key))
    dataset = _canonical_dataset(payload.get("dataset"))
    if dataset != payload.get("dataset"):
        raise ValueError("The dataset field must use its canonical name")
    return dataset


def load_catalog(catalog_path):
    """Read a catalog containing only exact original training sources."""
    catalog = _read(catalog_path)
    _header(catalog, "sources")
    seen = set()
    for source in catalog["sources"]:
        source_id, _ = validate_source(source)
        if source_id != source["source_image_id"] or source_id in seen:
            raise ValueError("Catalog source IDs must be unique normalized annotation paths")
        seen.add(source_id)
        path = source.get("image_path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("Catalog image_path must be the absolute original image path")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        captions = source.get("captions")
        if not isinstance(captions, list) or not captions or any(not isinstance(c, str) or not c.strip() for c in captions):
            raise ValueError("Catalog sources must retain their original human captions")
    return catalog


def _sources(catalog, limit=None, source_ids=None):
    sources = catalog["sources"]
    if source_ids is not None:
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        requested = {normalize_source_image_id(value) for value in source_ids}
        missing = requested - {source["source_image_id"] for source in sources}
        if missing:
            raise ValueError("Requested source IDs are absent from the training catalog: {}".format(sorted(missing)))
        sources = [source for source in sources if source["source_image_id"] in requested]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        sources = sources[:limit]
    return sources


def _base_config(stage, catalog, sources, **options):
    return {"stage": stage, "catalog_sha256": _digest(catalog),
            "source_ids": [source["source_image_id"] for source in sources], **options}


def _resume(output, dataset, config, *, key="records"):
    output = Path(output).expanduser().resolve()
    if output.exists():
        payload = _read(output)
        if _header(payload, key) != dataset or payload.get("config") != config:
            raise ValueError("Resume configuration differs; use a new output for changed inputs or options")
        return payload
    payload = {"version": 1, "dataset": dataset, "config": config, key: []}
    atomic_write_json(output, payload)
    return payload


def _identity(record, sources):
    if not isinstance(record, Mapping):
        raise ValueError("An augmentation record must be an object")
    source_id = normalize_source_image_id(record.get("source_image_id"))
    if source_id not in sources:
        raise ValueError("Record must reference an exact original training source: {}".format(source_id))
    source = sources[source_id]
    if type(record.get("source_pid")) is not int or record["source_pid"] != source["source_pid"]:
        raise ValueError("Record source_pid differs from its original training identity")
    if record.get("split", "train") != "train":
        raise ValueError("Only training records may be augmented")
    return source


def _relative_asset(value, directory):
    if not isinstance(value, str) or not value:
        raise ValueError("Generated image_path must be a contained relative path")
    value = value.replace("\\", "/")
    if (Path(value).is_absolute() or PureWindowsPath(value).drive or ":" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("Generated image_path must be a contained relative path")
    directory = Path(directory).resolve()
    path = (directory / value).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Generated image_path escapes its manifest directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _output_relative(path, output):
    try:
        return Path(path).resolve().relative_to(Path(output).resolve().parent).as_posix()
    except ValueError:
        raise ValueError("Generated assets must live inside the output manifest directory; choose a common parent for the output") from None


def export_catalog(dataset_name, root_dir, output, *, limit=None, source_ids=None):
    """Export original train annotations, never an already-augmented loader."""
    from datasets.build import DATASETS
    name = _canonical_dataset(dataset_name)
    dataset = DATASETS[name](root=root_dir, verbose=False)
    sources = {}
    for annotation in dataset.train_annos:
        source_id = normalize_source_image_id(annotation["source_image_id"])
        source = {"source_image_id": source_id, "source_pid": annotation["source_pid"],
                  "image_path": str(Path(annotation["img_path"]).resolve()), "split": "train",
                  "captions": list(annotation["captions"])}
        if source_id in sources:
            previous = sources[source_id]
            if any(previous[key] != source[key] for key in ("source_pid", "image_path")):
                raise ValueError("Ambiguous original training source: {}".format(source_id))
            for caption in source["captions"]:
                if caption not in previous["captions"]:
                    previous["captions"].append(caption)
        else:
            sources[source_id] = source
    payload = {"version": 1, "dataset": name, "sources": list(sources.values())}
    payload["sources"] = _sources(payload, limit, source_ids)
    load_catalog(payload)
    if Path(output).exists() and _read(output) != payload:
        raise ValueError("Catalog output differs; choose a new output for changed annotations or scope")
    atomic_write_json(output, payload)
    return payload


def _provider(provider, attributes):
    if provider is not None and attributes is not None:
        raise ValueError("Choose a provider object or an attribute cache, not both")
    if provider is None:
        if attributes is None:
            raise ValueError("The default attribute-cache backend requires --attributes")
        provider = AttributeCache(attributes)
    backend = getattr(provider, "backend", "custom")
    model = getattr(provider, "model", type(provider).__name__)
    configuration = {"backend": backend, "model": model}
    for name in ("base_url", "api_key_env", "timeout", "max_image_bytes"):
        if hasattr(provider, name):
            configuration[name] = getattr(provider, name)
    if isinstance(provider, AttributeCache):
        configuration["attributes_sha256"] = _digest(provider._attributes)
    return provider, configuration


def _extract(provider, image_path, key):
    if isinstance(provider, AttributeCache):
        return provider.extract_attributes(image_path, source_image_id=key)
    return validate_attributes(provider.extract_attributes(image_path))


def _caption_options(provider_config, seed):
    return {"backend": provider_config["backend"], "model": provider_config["model"], "seed": seed}


def augment_text(catalog_path, output, *, provider=None, attributes=None, seed=0,
                 limit=None, source_ids=None):
    catalog = load_catalog(catalog_path)
    sources = _sources(catalog, limit, source_ids)
    provider, provider_config = _provider(provider, attributes)
    options = _caption_options(provider_config, seed)
    config = _base_config("text", catalog, sources, provider=provider_config,
                          generation=generation_provenance(**options))
    payload = _resume(output, catalog["dataset"], config)
    by_source = {source["source_image_id"]: source for source in sources}
    done = {}
    for record in payload["records"]:
        source = _identity(record, by_source)
        key = source["source_image_id"]
        if key in done:
            raise ValueError("Duplicate source record in saved text manifest")
        attrs = _extract(provider, source["image_path"], key) if isinstance(provider, AttributeCache) else None
        validate_caption_record(record, source, attrs=attrs, **options)
        validate_caption_strings(record["captions"], source["captions"])
        done[key] = record
    for source in sources:
        key = source["source_image_id"]
        if key in done:
            continue
        record = generate_caption_record(source, _extract(provider, source["image_path"], key), **options)
        validate_caption_strings(record["captions"], source["captions"])
        payload["records"].append(record)
        atomic_write_json(output, payload)
    return payload


def validate_caption_strings(captions, human_captions=()):
    if (not isinstance(captions, list) or len(captions) != 2
            or any(not isinstance(value, str) or not value.strip() for value in captions)):
        raise ValueError("Exactly two nonempty caption strings are required")
    normalized = [" ".join(value.split()) for value in captions]
    if len(set(normalized)) != 2:
        raise ValueError("The two captions are duplicates")
    human = {" ".join(value.split()) for value in human_captions}
    if any(value in human for value in normalized):
        raise ValueError("A generated caption duplicates an original human caption")
    if any(_ERROR_TEXT.search(value) or "{" in value or "}" in value for value in captions):
        raise ValueError("API error text or unresolved placeholders cannot become captions")
    return list(captions)


def import_captions(catalog_path, captions_path, output, *, limit=None,
                    source_ids=None, strict=False):
    """Import exact-ID legacy caption pairs; report unusable sources explicitly."""
    catalog = load_catalog(catalog_path)
    sources = _sources(catalog, limit, source_ids)
    legacy = _read(captions_path)
    if not isinstance(legacy, dict):
        raise ValueError("Legacy captions must map exact source IDs to two strings")
    normalized = {}
    invalid_keys = 0
    for key, value in legacy.items():
        try:
            key = normalize_source_image_id(key)
        except ValueError:
            invalid_keys += 1
            continue  # Nontraining/absolute keys cannot be guessed into the train catalog.
        if key in normalized:
            raise ValueError("Duplicate normalized source IDs in the legacy cache")
        normalized[key] = value
    source_hash = _digest(legacy)
    config = _base_config("import-captions", catalog, sources, source_sha256=source_hash, strict=bool(strict))
    payload = _resume(output, catalog["dataset"], config)
    expected = []
    invalid = []
    for source in sources:
        key = source["source_image_id"]
        try:
            captions = validate_caption_strings(normalized.get(key), source["captions"])
        except ValueError as error:
            if strict:
                raise ValueError("Unusable legacy captions for {}: {}".format(key, error)) from None
            invalid.append({"source_image_id": key, "reason": str(error)})
            continue
        expected.append({"kind": "text", "source_image_id": key, "source_pid": source["source_pid"],
                         "captions": captions, "provenance": {
                             "backend": "legacy-caption-cache", "model": "legacy-unspecified",
                             "source_sha256": source_hash}})
    if payload["records"] != expected[:len(payload["records"])]:
        raise ValueError("Saved imported records differ from the validated legacy cache")
    # A legacy import is cheap and fully local. Validate all rows, then commit
    # once rather than rewriting a growing 18k-row manifest after every row.
    payload["records"] = expected
    training_ids = {source["source_image_id"] for source in catalog["sources"]}
    payload["report"] = {"imported_sources": len(expected), "skipped_invalid_sources": invalid,
                         "ignored_nontraining_keys": invalid_keys + sum(key not in training_ids for key in normalized)}
    atomic_write_json(output, payload)
    return payload


def _subject_mapping(value):
    mapping = {} if value is None else _read(value)
    if not isinstance(mapping, dict) or any(not isinstance(key, str) or not isinstance(subject, str)
                                            for key, subject in mapping.items()):
        raise ValueError("subject_map must map exact original PID strings to checkpoint subject strings")
    return mapping


def generate_images(catalog_path, output, *, pipeline=None, metadata=None,
                    local_model_dir=None, checkpoint_dir=None, subject_map=None,
                    output_dir=None, num_candidates=20, seed=0, num_inference_steps=30,
                    guidance_scale=7.5, height=512, width=256, negative_prompt=None,
                    device="cpu", dtype="float32", network_alpha=None,
                    limit=None, source_ids=None):
    from augmentation.images import create_pipeline, generate_candidates, load_legacy_bundle, resolve_identity_binding
    catalog = load_catalog(catalog_path)
    sources = _sources(catalog, limit, source_ids)
    subject_map = _subject_mapping(subject_map)
    if metadata is not None and pipeline is None:
        raise ValueError("Injected identity metadata requires an injected pipeline; local generation reads its checkpoint metadata")
    if metadata is None:
        if checkpoint_dir is None:
            raise ValueError("Image generation requires checkpoint_dir or injected checkpoint metadata")
        bundle = load_legacy_bundle(checkpoint_dir)
        metadata = bundle.metadata
    bindings = {source["source_image_id"]: resolve_identity_binding(
        metadata, source["source_pid"], subject_id=subject_map.get(str(source["source_pid"]))) for source in sources}
    subjects = {}
    for binding in bindings.values():
        previous = subjects.setdefault(binding.subject_id, binding.source_pid)
        if previous != binding.source_pid:
            raise ValueError("Distinct source PIDs cannot share a checkpoint subject binding")
    directory = Path(output_dir or Path(output).resolve().parent / "generated").resolve()
    if not directory.is_relative_to(Path(output).resolve().parent):
        raise ValueError("Generated images must live inside the candidate catalog directory")
    if pipeline is None:
        if local_model_dir is None or checkpoint_dir is None:
            raise ValueError("Image generation requires an explicit local_model_dir and checkpoint_dir")
        pipeline = create_pipeline(local_model_dir, checkpoint_dir, device=device, dtype=dtype,
                                   expected_subjects=[binding.subject_id for binding in bindings.values()],
                                   network_alpha=network_alpha)
    config = _base_config("images", catalog, sources, metadata_sha256=_digest(metadata), subject_map=subject_map,
                          output_dir=str(directory), num_candidates=num_candidates, seed=seed,
                          num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                          height=height, width=width, negative_prompt=negative_prompt,
                          pipeline=getattr(pipeline, "_offline_augmentation_provenance", {}),
                          source_images_sha256={source["source_image_id"]: _file_digest(source["image_path"]) for source in sources})
    payload = _resume(output, catalog["dataset"], config, key="candidates")
    by_source = {source["source_image_id"]: source for source in sources}
    saved = {}
    for record in payload["candidates"]:
        _identity(record, by_source)
        path = str(Path(record["image_path"]).resolve())
        if path in saved or record.get("kind") != "image_candidate" or record.get("selected") is not False:
            raise ValueError("Invalid or duplicate saved image candidate")
        saved[path] = record

    def save_candidate(record):
        path = str(Path(record["image_path"]).resolve())
        if path in saved:
            if saved[path] != record:
                raise ValueError("Saved candidate differs from its image provenance sidecar")
        else:
            saved[path] = record
            payload["candidates"].append(record)
            atomic_write_json(output, payload)

    expected = set()
    for source in sources:
        rows = generate_candidates(source, pipeline, bindings[source["source_image_id"]], directory,
                                   num_candidates=num_candidates, seed=seed,
                                   num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                   height=height, width=width, negative_prompt=negative_prompt,
                                   on_candidate=save_candidate)
        expected.update(str(Path(row["image_path"]).resolve()) for row in rows)
    if set(saved) != expected:
        raise ValueError("Saved candidate catalog includes unexpected image records")
    return payload


def create_clip_encoder(clip_weights, *, device="cpu"):
    """Load the project's fixed ViT-B/16 image encoder and evaluation transform."""
    import torch
    from PIL import Image
    from datasets.build import build_transforms
    from model.clip_model import build_CLIP_from_openai_pretrained
    path = Path(clip_weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Filtering requires an explicit local CLIP checkpoint: {}".format(path))
    model, configuration = build_CLIP_from_openai_pretrained(str(path), (384, 128), 16, allow_download=False)
    if configuration["vision_patch_size"] != 16 or configuration["vision_width"] != 768 or configuration["vision_layers"] != 12:
        raise ValueError("Stage-four filtering requires the fixed CLIP ViT-B/16 checkpoint")
    model = model.float().to(device).eval()
    transform = build_transforms((384, 128), aug=False, is_train=False)

    def encode(paths):
        tensors = []
        for image_path in paths:
            with Image.open(image_path) as image:
                tensors.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            sequence = model.encode_image(torch.stack(tensors).to(device))
        if sequence.ndim != 3:
            raise ValueError("Project CLIP must return its image token sequence")
        return sequence[:, 0].float().cpu()

    encode.provenance = {"backend": "project-clip-vit-b-16", "weights_sha256": _file_digest(path),
                         "image_size": [384, 128], "pooling": "class_token", "device": str(device)}
    return encode


def filter_images(catalog_path, candidates_path, output, *, feature_encoder=None,
                  clip_weights=None, device="cpu", keep=10, batch_size=8,
                  limit=None, source_ids=None):
    from augmentation.images import select_candidates, selected_image_records
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    catalog = load_catalog(catalog_path)
    sources = _sources(catalog, limit, source_ids)
    source_map = {source["source_image_id"]: source for source in catalog["sources"]}
    candidates = _read(candidates_path)
    if _header(candidates, "candidates") != catalog["dataset"]:
        raise ValueError("Candidate catalog dataset differs from the original catalog")
    groups = {source["source_image_id"]: [] for source in sources}
    seen = set()
    seen_files = set()
    originals = {_file_key(source["image_path"]) for source in catalog["sources"]}
    for candidate in candidates["candidates"]:
        source = _identity(candidate, source_map)
        if candidate.get("kind") != "image_candidate" or candidate.get("selected") is not False:
            raise ValueError("Filtering expects unselected image candidates")
        path = Path(candidate["image_path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Candidate image_path must be an existing absolute path")
        path = path.resolve()
        file_key = _file_key(path)
        if path in seen or file_key in seen_files or file_key in originals:
            raise ValueError("Duplicate candidate image or an original image used as a generated artifact")
        seen.add(path)
        seen_files.add(file_key)
        if source["source_image_id"] in groups:
            _output_relative(path, output)
            groups[source["source_image_id"]].append(candidate)
    # A full source catalog may accompany a small, deliberately scoped run.
    # Only original images with actual candidates need feature extraction.
    sources = [source for source in sources if groups[source["source_image_id"]]]
    if not sources:
        raise ValueError("No candidates exist for the selected training sources")
    for source in sources:
        count = len(groups[source["source_image_id"]])
        if count < keep:
            raise ValueError("Source {} has {} candidates, fewer than keep={}; finish generation or lower --keep".format(
                source["source_image_id"], count, keep))
    if feature_encoder is None:
        if clip_weights is None:
            raise ValueError("Filtering requires --clip_weights or an injected feature encoder")
        feature_encoder = create_clip_encoder(clip_weights, device=device)
    config = _base_config("filter", catalog, sources, candidates_sha256=_digest(candidates), keep=keep,
                          batch_size=batch_size, encoder=getattr(feature_encoder, "provenance", {"backend": "injected"}),
                          source_images_sha256={source["source_image_id"]: _file_digest(source["image_path"]) for source in sources},
                          candidate_images_sha256={str(path): _file_digest(path) for path in sorted(seen)})
    payload = _resume(output, catalog["dataset"], config)
    saved = {}
    for record in payload["records"]:
        source = _identity(record, source_map)
        if source["source_image_id"] not in groups or record.get("kind") != "image" or record.get("selected") is not True:
            raise ValueError("Invalid selected image in saved filtering manifest")
        path = _relative_asset(record.get("image_path"), Path(output).resolve().parent)
        if str(path) in saved:
            raise ValueError("Duplicate saved selected image")
        saved[str(path)] = record
    expected = set()
    for source in sources:
        group = groups[source["source_image_id"]]
        rankings = select_candidates(source["image_path"], [row["image_path"] for row in group],
                                     feature_encoder, k=keep, batch_size=batch_size)
        records = selected_image_records(source, group, rankings, Path(output).resolve().parent)
        for record in records:
            key = str(_relative_asset(record["image_path"], Path(output).resolve().parent))
            expected.add(key)
            if key in saved:
                if saved[key] != record:
                    raise ValueError("Saved selection differs from current deterministic ranking")
            else:
                saved[key] = record
                payload["records"].append(record)
                atomic_write_json(output, payload)
    if set(saved) != expected:
        raise ValueError("Saved filtering manifest has unexpected selected records")
    return payload


def caption_images(catalog_path, selected_path, output, *, provider=None,
                   attributes=None, seed=0, limit=None, source_ids=None):
    catalog = load_catalog(catalog_path)
    sources = _sources(catalog, limit, source_ids)
    source_map = {source["source_image_id"]: source for source in catalog["sources"]}
    selected = _read(selected_path)
    if _header(selected, "records") != catalog["dataset"]:
        raise ValueError("Selected image manifest dataset differs from the original catalog")
    selected_dir = Path(selected_path).resolve().parent
    wanted = {source["source_image_id"] for source in sources}
    items = []
    seen = set()
    for record in selected["records"]:
        source = _identity(record, source_map)
        if record.get("kind") != "image" or type(record.get("selected")) is not bool:
            raise ValueError("Captioning expects filtered image records with a boolean selected field")
        if not isinstance(record.get("provenance", {}), dict):
            raise ValueError("Selected image provenance must be an object")
        if not record["selected"] or source["source_image_id"] not in wanted:
            continue
        path = _relative_asset(record.get("image_path"), selected_dir)
        relative = _output_relative(path, output)
        if path in seen:
            raise ValueError("Duplicate retained generated image")
        seen.add(path)
        items.append((record, source, path, relative))
    provider, provider_config = _provider(provider, attributes)
    options = _caption_options(provider_config, seed)
    config = _base_config("caption-images", catalog, sources, selected_sha256=_digest(selected),
                          provider=provider_config, generation=generation_provenance(**options),
                          images_sha256={record["image_path"]: _file_digest(path) for record, _, path, _ in items})
    payload = _resume(output, catalog["dataset"], config)
    done = {}
    for record in payload["records"]:
        _identity(record, source_map)
        key = record.get("image_path")
        if key in done:
            raise ValueError("Duplicate generated image in saved caption manifest")
        done[key] = record
    expected_keys = set()
    for selected_record, original_source, path, relative in items:
        # This key intentionally names the GENERATED artifact in the selected
        # manifest, never its source ID or a basename guessed from the source.
        cache_key = selected_record["image_path"].replace("\\", "/")
        temporary_source = {**original_source, "source_image_id": cache_key, "image_path": str(path)}
        existing = done.get(relative)
        if existing is not None and not isinstance(provider, AttributeCache):
            attrs = validate_attributes(existing.get("attributes"))
        else:
            attrs = _extract(provider, path, cache_key)
        caption_record = generate_caption_record(temporary_source, attrs, **options)
        validate_caption_strings(caption_record["captions"])
        record = {**selected_record, "image_path": relative, "captions": caption_record["captions"],
                  "attributes": caption_record["attributes"], "provenance": {
                      **selected_record.get("provenance", {}), "captioning": caption_record["provenance"],
                      "caption_image_path": cache_key}}
        expected_keys.add(relative)
        if existing is not None:
            if existing != record:
                raise ValueError("Saved generated-image captions differ from current attributes or provenance")
        else:
            payload["records"].append(record)
            atomic_write_json(output, payload)
    if set(done) - expected_keys:
        raise ValueError("Saved generated-image caption manifest contains unexpected records")
    return payload


def _file_key(path):
    status = Path(path).stat()
    return (status.st_dev, status.st_ino) if status.st_ino else str(Path(path).resolve())


def merge_manifests(inputs, output, *, catalog_path=None):
    """Merge ready records and rebase contained generated files for the loader."""
    if not inputs:
        raise ValueError("At least one input augmentation manifest is required")
    catalog = load_catalog(catalog_path) if catalog_path is not None else None
    source_map = {source["source_image_id"]: source for source in catalog["sources"]} if catalog else None
    original_files = {_file_key(source["image_path"]) for source in catalog["sources"]} if catalog else set()
    records = []
    dataset = catalog["dataset"] if catalog else None
    seen_text = set()
    seen_pairs = set()
    seen_images = set()
    inputs_config = []
    skipped = 0
    for input_path in inputs:
        payload = _read(input_path)
        name = _header(payload, "records")
        dataset = dataset or name
        if name != dataset:
            raise ValueError("Cannot merge manifests from different datasets")
        inputs_config.append({"path": str(Path(input_path).resolve()), "sha256": _digest(payload)})
        for original in payload["records"]:
            if not isinstance(original, Mapping) or original.get("kind") not in {"text", "image"}:
                raise ValueError("Only ready text/image records can enter a training manifest")
            record = copy.deepcopy(original)
            kind = record["kind"]
            if kind == "image":
                if type(record.get("selected")) is not bool:
                    raise ValueError("Generated image records require selected: true or false")
                if not record["selected"]:
                    skipped += 1
                    continue
            if source_map is not None:
                source = _identity(record, source_map)
            else:
                validate_source({**record, "split": record.get("split", "train")})
                source = None
            if not isinstance(record.get("provenance", {}), dict):
                raise ValueError("Record provenance must be an object")
            captions = validate_caption_strings(record.get("captions"), source["captions"] if source and kind == "text" else ())
            source_id = record["source_image_id"]
            if kind == "text":
                if source_id in seen_text:
                    raise ValueError("Duplicate text augmentation source across input manifests")
                seen_text.add(source_id)
                for caption in captions:
                    pair = (source_id, " ".join(caption.split()))
                    if pair in seen_pairs:
                        raise ValueError("Duplicate source-caption across input manifests")
                    seen_pairs.add(pair)
            else:
                path = _relative_asset(record.get("image_path"), Path(input_path).resolve().parent)
                key = _file_key(path)
                if key in seen_images or key in original_files:
                    raise ValueError("Duplicate generated image or an original image used as a generated artifact")
                seen_images.add(key)
                record["image_path"] = _output_relative(path, output)
            records.append(record)
    config = {"stage": "merge", "inputs": inputs_config,
              "catalog_sha256": _digest(catalog) if catalog else None}
    payload = {"version": 1, "dataset": dataset, "config": config, "records": records,
               "report": {"text_records": sum(row["kind"] == "text" for row in records),
                          "image_records": sum(row["kind"] == "image" for row in records),
                          "skipped_unselected": skipped}}
    if Path(output).exists() and _read(output) != payload:
        raise ValueError("Merged output differs; use a new output for changed manifests")
    atomic_write_json(output, payload)
    return payload
