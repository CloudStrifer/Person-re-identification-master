"""Offline stage-4 image generation and per-original CLIP selection.

The diffusion dependency is imported only by :func:`create_pipeline`. Legacy
DreamBooth checkpoints are the three files written by the repository's
multi-subject script, not a PEFT or modern diffusers adapter directory. LoRA
updates are folded into ordinary attention projections, so inference does not
depend on the removed ``LoRAAttnProcessor`` classes.

Candidate catalogs deliberately have no captions. Selected records must acquire
two captions describing the *generated* image before entering a training
manifest; copying the original image's captions would give incorrect labels.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from PIL import Image
import torch
from torch import nn


_PROJECTIONS = {"to_q": "to_q", "to_k": "to_k", "to_v": "to_v", "to_out": "to_out.0"}
_LORA_KEYS = {f"{name}_lora.{direction}.weight"
              for name in _PROJECTIONS for direction in ("down", "up")}
_VIEWS = (
    "standing, front view", "walking, side view", "standing, rear view",
    "walking, three-quarter view", "standing, side view",
    "walking, front view", "standing, three-quarter rear view",
    "walking, rear view", "standing, three-quarter view", "walking, opposite side view",
)


@dataclass(frozen=True)
class IdentityBinding:
    source_pid: str
    subject_id: str
    token: str
    random_id: str


@dataclass(frozen=True)
class LegacyAdapterBundle:
    lora_state: Mapping[str, Mapping[str, torch.Tensor]]
    embeddings: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def identity_for(self, source_pid: Any, *, subject_id: str | None = None) -> IdentityBinding:
        return resolve_identity_binding(self.metadata, source_pid, subject_id=subject_id)


def _identifier(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty string or integer")
    return str(value)


def validate_identity_metadata(metadata: Mapping[str, Any],
                               expected_subjects: Sequence[Any] | None = None) -> None:
    """Reject absent/ambiguous identity bindings instead of inventing a token."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    tokens = metadata.get("placeholder_tokens")
    random_ids = metadata.get("subject_random_ids")
    if not isinstance(tokens, Mapping) or not tokens:
        raise ValueError("metadata.placeholder_tokens must be a nonempty subject-to-token mapping")
    if not isinstance(random_ids, Mapping) or set(tokens) != set(random_ids):
        raise ValueError("metadata.subject_random_ids must match placeholder_tokens subjects exactly")
    for subject, token in tokens.items():
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("metadata subject keys must be nonempty strings")
        if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
            raise ValueError(f"Invalid learned token for subject {subject!r}")
        random_id = random_ids[subject]
        if not isinstance(random_id, str) or not random_id.strip() or any(c.isspace() for c in random_id):
            raise ValueError(f"Invalid random identifier for subject {subject!r}")
    if len(set(tokens.values())) != len(tokens):
        raise ValueError("Different subjects cannot share a learned identity token")
    if len(set(random_ids.values())) != len(random_ids):
        raise ValueError("Different subjects cannot share a training random identifier")
    if "subject_ids" in metadata:
        subjects = metadata["subject_ids"]
        if (not isinstance(subjects, list) or any(not isinstance(s, str) for s in subjects)
                or len(subjects) != len(set(subjects)) or set(subjects) != set(tokens)):
            raise ValueError("metadata.subject_ids does not match placeholder_tokens")
    if expected_subjects is not None:
        missing = {_identifier(pid, "subject") for pid in expected_subjects} - set(tokens)
        if missing:
            raise ValueError(f"Checkpoint has no identity binding for subjects: {sorted(missing)}")


def resolve_identity_binding(metadata: Mapping[str, Any], source_pid: Any, *,
                             subject_id: str | None = None) -> IdentityBinding:
    """Use exact PID keys, or an explicit caller-supplied subject mapping.

    In particular, ``1`` is not silently guessed to mean folder ``0001``.
    """
    validate_identity_metadata(metadata)
    pid = _identifier(source_pid, "source_pid")
    subject = pid if subject_id is None else _identifier(subject_id, "subject_id")
    if subject not in metadata["placeholder_tokens"]:
        raise ValueError(f"Checkpoint has no identity binding for subject {subject!r} (PID {pid!r})")
    return IdentityBinding(pid, subject, metadata["placeholder_tokens"][subject],
                           metadata["subject_random_ids"][subject])


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_embedding_vectors(embeddings: Mapping[str, torch.Tensor],
                                metadata: Mapping[str, Any], width: int | None = None) -> None:
    expected = set(metadata["placeholder_tokens"].values())
    if not isinstance(embeddings, Mapping) or set(embeddings) != expected:
        actual = set(embeddings) if isinstance(embeddings, Mapping) else set()
        raise ValueError(f"Learned embeddings do not match identity tokens; missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")
    dimensions = set()
    for token, vector in embeddings.items():
        if (not isinstance(vector, torch.Tensor) or vector.ndim != 1
                or not vector.is_floating_point() or not torch.isfinite(vector).all().item()):
            raise ValueError(f"Embedding for {token!r} must be one finite floating-point vector")
        dimensions.add(vector.numel())
        if vector.numel() == 0 or (width is not None and vector.numel() != width):
            raise ValueError(f"Embedding width mismatch for {token!r}: expected {width}, got {vector.numel()}")
    if len(dimensions) != 1:
        raise ValueError("Learned identity embeddings have inconsistent widths")


def load_legacy_bundle(checkpoint_dir: str | Path, *,
                       expected_subjects: Sequence[Any] | None = None) -> LegacyAdapterBundle:
    directory = Path(checkpoint_dir).expanduser().resolve()
    paths = {name: directory / name for name in ("unet_lora.pt", "embeddings.pt", "metadata.json")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing legacy adapter file: {path}")
    metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
    validate_identity_metadata(metadata, expected_subjects)
    lora = torch.load(paths["unet_lora.pt"], map_location="cpu", weights_only=True)
    embeddings = torch.load(paths["embeddings.pt"], map_location="cpu", weights_only=True)
    if not isinstance(lora, Mapping) or not lora:
        raise ValueError("unet_lora.pt has no attention processor weights")
    _validate_embedding_vectors(embeddings, metadata)
    provenance = {"format": "legacy_nested_attention_lora", "checkpoint_dir": str(directory),
                  "sha256": {name: _file_hash(path) for name, path in paths.items()}}
    return LegacyAdapterBundle(lora, embeddings, metadata, provenance)


def fold_legacy_lora(unet: nn.Module, state: Mapping[str, Mapping[str, torch.Tensor]], *,
                     network_alpha: float | None = None) -> dict[str, Any]:
    """Fold ``scale * up @ down`` into q/k/v/out projection weights.

    All processor names, all eight keys, every rank and matrix shape are checked
    before any model weight changes. The old trainer constructs processors with
    ``network_alpha=None``: its multiplier is **1**, not ``1 / rank``. An explicit
    alpha uses the historical ``alpha / rank`` convention. Call only once on a
    freshly loaded base UNet; a second fold would double the adapter effect.
    """
    if getattr(unet, "_offline_legacy_lora_folded", False):
        raise ValueError("Legacy LoRA has already been folded into this UNet")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Legacy LoRA state must be a nonempty nested mapping")
    if network_alpha is not None and (isinstance(network_alpha, bool)
                                      or not math.isfinite(float(network_alpha))):
        raise ValueError("network_alpha must be finite or None")
    if hasattr(unet, "attn_processors"):
        expected = set(unet.attn_processors)
        if set(state) != expected:
            raise ValueError(f"Attention processor mismatch; missing={sorted(expected - set(state))}, "
                             f"unexpected={sorted(set(state) - expected)}")
    plan = []
    seen_weights = set()
    ranks = {}
    for processor_name, weights in state.items():
        if not isinstance(processor_name, str) or not processor_name.endswith(".processor"):
            raise ValueError(f"Invalid legacy attention processor name: {processor_name!r}")
        if not isinstance(weights, Mapping) or set(weights) != _LORA_KEYS:
            actual = set(weights) if isinstance(weights, Mapping) else set()
            raise ValueError(f"Invalid LoRA keys for {processor_name}; missing={sorted(_LORA_KEYS - actual)}, "
                             f"unexpected={sorted(actual - _LORA_KEYS)}")
        attention_name = processor_name.removesuffix(".processor")
        processor_ranks = set()
        for legacy_name, projection_name in _PROJECTIONS.items():
            module_name = f"{attention_name}.{projection_name}"
            try:
                projection = unet.get_submodule(module_name)
            except (AttributeError, KeyError) as error:
                raise ValueError(f"UNet is missing attention projection {module_name}") from error
            weight = getattr(projection, "weight", None)
            down = weights[f"{legacy_name}_lora.down.weight"]
            up = weights[f"{legacy_name}_lora.up.weight"]
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
                raise ValueError(f"{module_name} must have a floating-point matrix weight")
            if (not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor)
                    or down.ndim != 2 or up.ndim != 2
                    or not down.is_floating_point() or not up.is_floating_point()):
                raise ValueError(f"{processor_name}.{legacy_name} LoRA weights must be floating-point matrices")
            rank = down.shape[0]
            if rank <= 0 or tuple(down.shape) != (rank, weight.shape[1]) or tuple(up.shape) != (weight.shape[0], rank):
                raise ValueError(f"LoRA shape mismatch for {module_name}: base={tuple(weight.shape)}, "
                                 f"down={tuple(down.shape)}, up={tuple(up.shape)}")
            if not torch.isfinite(down).all().item() or not torch.isfinite(up).all().item():
                raise ValueError(f"Non-finite LoRA weight for {module_name}")
            if id(weight) in seen_weights:
                raise ValueError(f"Multiple LoRA entries target the same projection {module_name}")
            seen_weights.add(id(weight))
            processor_ranks.add(rank)
            scale = 1.0 if network_alpha is None else float(network_alpha) / rank
            plan.append((weight, down, up, scale))
        if len(processor_ranks) != 1:
            raise ValueError(f"Inconsistent ranks in legacy processor {processor_name}")
        ranks[processor_name] = processor_ranks.pop()
    with torch.no_grad():
        for weight, down, up, scale in plan:
            delta = up.to(device=weight.device, dtype=torch.float32) @ down.to(device=weight.device, dtype=torch.float32)
            weight.add_((delta * scale).to(dtype=weight.dtype))
    unet._offline_legacy_lora_folded = True
    return {"method": "fold_up_matmul_down", "network_alpha": network_alpha,
            "scale_convention": "1" if network_alpha is None else "network_alpha / rank",
            "ranks": ranks, "projection_count": len(plan)}


def install_learned_embeddings(tokenizer: Any, text_encoder: nn.Module,
                                embeddings: Mapping[str, torch.Tensor],
                                metadata: Mapping[str, Any]) -> dict[str, int]:
    """Install vectors by token text, independent of historical token IDs."""
    validate_identity_metadata(metadata)
    input_weight = text_encoder.get_input_embeddings().weight
    _validate_embedding_vectors(embeddings, metadata, input_weight.shape[1])
    # Validate vectors before modifying the vocabulary or resizing any weights.
    tokens = list(metadata["placeholder_tokens"].values())
    vocabulary = tokenizer.get_vocab()
    for token in tokens:
        if token not in vocabulary:
            if tokenizer.add_tokens(token) != 1:
                raise ValueError(f"Cannot add learned identity token {token!r}")
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in tokens}
    if len(set(token_ids.values())) != len(tokens):
        raise ValueError("Learned identity tokens resolve to duplicate tokenizer IDs")
    for token, token_id in token_ids.items():
        if (not isinstance(token_id, int) or token_id < 0 or token_id >= len(tokenizer)
                or tokenizer.encode(token, add_special_tokens=False) != [token_id]):
            raise ValueError(f"Learned identity token {token!r} does not encode to its single vocabulary ID")
    text_encoder.resize_token_embeddings(len(tokenizer))
    input_weight = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        for token, token_id in token_ids.items():
            input_weight[token_id].copy_(embeddings[token].to(device=input_weight.device, dtype=input_weight.dtype))
    return token_ids


def create_pipeline(local_model_dir: str | Path, checkpoint_dir: str | Path, *,
                    device: str = "cpu", dtype: str | torch.dtype = torch.float32,
                    expected_subjects: Sequence[Any] | None = None,
                    network_alpha: float | None = None) -> Any:
    """Build a local SD1.5 pipeline; never resolve a remote model identifier."""
    model_dir = Path(local_model_dir).expanduser().resolve()
    if not model_dir.is_dir() or not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(f"Expected a local Stable Diffusion pipeline directory: {model_dir}")
    bundle = load_legacy_bundle(checkpoint_dir, expected_subjects=expected_subjects)
    if isinstance(dtype, str):
        choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in choices:
            raise ValueError(f"Unsupported diffusion dtype: {dtype}")
        dtype = choices[dtype]
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Diffusion dtype must be float32, float16, or bfloat16")
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as error:
        raise ImportError("Image generation requires the optional diffusers and transformers dependencies") from error
    pipeline = StableDiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=dtype, local_files_only=True)
    fold_report = fold_legacy_lora(pipeline.unet, bundle.lora_state, network_alpha=network_alpha)
    install_learned_embeddings(pipeline.tokenizer, pipeline.text_encoder, bundle.embeddings, bundle.metadata)
    pipeline.to(device)
    for method in ("enable_attention_slicing", "enable_vae_slicing"):
        if hasattr(pipeline, method):
            getattr(pipeline, method)()
    pipeline._offline_augmentation_provenance = {
        **bundle.provenance, "base_model_dir": str(model_dir), "dtype": str(dtype),
        "device": str(device), "lora": fold_report,
    }
    return pipeline


def candidate_seed(source_image_id: Any, candidate_index: int, base_seed: int = 0) -> int:
    source_id = _identifier(source_image_id, "source_image_id")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer")
    digest = hashlib.sha256(json.dumps(["stage4-image-v1", base_seed, source_id, candidate_index],
                                       ensure_ascii=False).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def build_identity_prompt(binding: IdentityBinding, candidate_index: int) -> str:
    if not isinstance(binding, IdentityBinding):
        raise TypeError("binding must be an IdentityBinding resolved from checkpoint metadata")
    return f"a {binding.random_id} full body photo of a {binding.token} person, {_VIEWS[candidate_index % len(_VIEWS)]}"


def _verify_image(path: Path) -> None:
    with Image.open(path) as opened:
        opened.verify()


def generate_candidates(source: Mapping[str, Any], pipeline: Any, binding: IdentityBinding,
                         output_dir: str | Path, *, num_candidates: int = 20,
                         seed: int = 0, num_inference_steps: int = 30,
                         guidance_scale: float = 7.5, height: int = 512, width: int = 256,
                         negative_prompt: str | None = None,
                         on_candidate: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Generate one image at a time with independent, stable per-original seeds.

    Completed candidates are verified and reused. A matching sidecar with a
    missing image can resume interrupted work. Existing images are never
    overwritten, and stale settings/identity bindings fail explicitly.
    ``pipeline`` is injectable for lightweight offline tests; production uses
    :func:`create_pipeline`. ``on_candidate(record)`` runs immediately after
    every saved or verified resumed image, allowing incremental catalog writes.
    """
    source_id = _identifier(source.get("source_image_id"), "source_image_id")
    source_pid = _identifier(source.get("source_pid"), "source_pid")
    if source.get("split") != "train":
        raise ValueError("Only original training images may seed image augmentation")
    if not isinstance(binding, IdentityBinding) or binding.source_pid != source_pid:
        raise ValueError("Source PID does not match the learned identity binding")
    for name, value in (("num_candidates", num_candidates), ("num_inference_steps", num_inference_steps),
                        ("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if height % 8 or width % 8:
        raise ValueError("Diffusion height and width must be divisible by 8")
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    source_path = Path(source["image_path"]).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    _verify_image(source_path)
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", source_id)[:40] or "source"
    directory = Path(output_dir).expanduser().resolve() / f"{slug}-{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    source_hash = _file_hash(source_path)
    records = []
    for index in range(num_candidates):
        path = directory / f"candidate_{index:03d}.png"
        sidecar = path.with_suffix(".json")
        per_image_seed = candidate_seed(source_id, index, seed)
        prompt = build_identity_prompt(binding, index)
        record = {"kind": "image_candidate", "selected": False,
                  "source_image_id": source["source_image_id"], "source_pid": source["source_pid"],
                  "image_path": str(path), "provenance": {
                      "candidate_index": index, "seed": per_image_seed, "prompt": prompt,
                      "subject_id": binding.subject_id, "identity_token": binding.token,
                      "subject_random_id": binding.random_id, "source_sha256": source_hash,
                      "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale,
                      "height": height, "width": width, "negative_prompt": negative_prompt,
                      "pipeline": getattr(pipeline, "_offline_augmentation_provenance", {}),
                  }}
        if sidecar.exists():
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
            if existing != record:
                raise ValueError(f"Candidate resume metadata differs: {sidecar}")
        elif path.exists():
            raise FileExistsError(f"Candidate has no provenance sidecar; refusing to overwrite: {path}")
        if path.exists():
            _verify_image(path)
            records.append(record)
            if on_candidate is not None:
                on_candidate(record)
            continue
        generator = torch.Generator(device="cpu").manual_seed(per_image_seed)
        with torch.inference_mode():
            result = pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator,
                              num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                              height=height, width=width, num_images_per_prompt=1)
        images = getattr(result, "images", None)
        flags = getattr(result, "nsfw_content_detected", None)
        if flags is not None and any(flags):
            raise ValueError(f"Diffusion pipeline rejected candidate {index} for source {source_id}")
        if not images or len(images) != 1 or not isinstance(images[0], Image.Image):
            raise ValueError("Diffusion pipeline must return exactly one PIL image")
        # Metadata first permits an interrupted save to resume with the same seed.
        if not sidecar.exists():
            with sidecar.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
        with path.open("xb") as stream:
            images[0].convert("RGB").save(stream, format="PNG")
        records.append(record)
        if on_candidate is not None:
            on_candidate(record)
    return records


def _normalized_features(encoder: Callable, paths: Sequence[Path]) -> torch.Tensor:
    with torch.inference_mode():
        features = torch.as_tensor(encoder(paths)).detach().to(device="cpu", dtype=torch.float32)
    if features.ndim != 2 or features.shape[0] != len(paths) or features.shape[1] == 0:
        raise ValueError("feature_encoder must return one feature vector per image, shape [N, D]")
    norms = features.norm(dim=1, keepdim=True)
    if not torch.isfinite(features).all().item() or (norms <= 0).any().item():
        raise ValueError("feature_encoder returned non-finite or zero image features")
    return features / norms


def select_candidates(source_path: str | Path, candidate_paths: Sequence[str | Path],
                       feature_encoder: Callable[[Sequence[Path]], torch.Tensor], *,
                       k: int = 10, batch_size: int = 8) -> list[dict[str, Any]]:
    """Rank cosine similarity to this individual original, never an identity mean.

    ``feature_encoder`` must be the fixed project CLIP ViT-B/16 image encoder
    (same weights, preprocessing and pooled feature for original and candidates).
    It receives paths and returns [N,D] unnormalized or normalized features.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or k > len(candidate_paths):
        raise ValueError("k must be positive and no larger than the number of candidates")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    original = Path(source_path).expanduser().resolve()
    paths = [Path(path).expanduser().resolve() for path in candidate_paths]
    if len(set(paths)) != len(paths) or original in paths:
        raise ValueError("Candidate paths must be unique and different from the source image")
    for path in [original, *paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = _normalized_features(feature_encoder, [original])
    scores = []
    for start in range(0, len(paths), batch_size):
        features = _normalized_features(feature_encoder, paths[start:start + batch_size])
        if features.shape[1] != reference.shape[1]:
            raise ValueError("Original and candidate CLIP feature widths differ")
        scores.extend((features @ reference[0]).clamp(-1, 1).tolist())
    order = sorted(range(len(paths)), key=lambda index: (-scores[index], index))[:k]
    return [{"image_path": str(paths[index]), "rank": rank, "cosine": scores[index]}
            for rank, index in enumerate(order, 1)]


def selected_image_records(source: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                            rankings: Sequence[Mapping[str, Any]],
                            manifest_dir: str | Path) -> list[dict[str, Any]]:
    """Prepare retained artifacts for captioning, with manifest-relative paths.

    These records are intentionally incomplete training records until the caller
    adds two captions obtained from each generated image's own attributes.
    """
    directory = Path(manifest_dir).expanduser().resolve()
    by_path = {Path(candidate["image_path"]).resolve(): candidate for candidate in candidates}
    records = []
    seen = set()
    for ranking in rankings:
        path = Path(ranking["image_path"]).resolve()
        if path in seen or path not in by_path:
            raise ValueError("Rankings must reference unique candidates from this source")
        seen.add(path)
        candidate = by_path[path]
        for field in ("source_image_id", "source_pid"):
            if candidate[field] != source[field]:
                raise ValueError(f"Candidate {field} differs from its original source")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(directory).as_posix()
        except ValueError as error:
            raise ValueError("Selected images must live inside the manifest directory") from error
        records.append({"kind": "image", "selected": True,
                        "source_image_id": source["source_image_id"], "source_pid": source["source_pid"],
                        "image_path": relative,
                        "provenance": {**candidate["provenance"], "rank": ranking["rank"],
                                       "cosine": ranking["cosine"],
                                       "selection_reference": "individual_original_image"}})
    return records
