"""Offline workflows: resumable cache text and complete fake-image preparation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

import augmentation.workflow as workflow_module
from augmentation.text import AttributeCache
from augmentation.workflow import (
    atomic_write_json, augment_text, caption_images, export_catalog, filter_images,
    generate_images, import_captions, load_catalog, merge_manifests,
)
from tools.augment import build_parser, main


@pytest.fixture
def original_data(tmp_path):
    folder = tmp_path / "data" / "RSTPReid"
    images = folder / "imgs"
    images.mkdir(parents=True)
    entries = [
        {"id": 37, "split": "train", "img_path": "a.png", "captions": ["a red original", "the original red shirt"]},
        {"id": 37, "split": "train", "img_path": "b.png", "captions": ["a blue original", "the original blue shirt"]},
        {"id": 80, "split": "val", "img_path": "val.png", "captions": ["a validation image"]},
        {"id": 81, "split": "test", "img_path": "test.png", "captions": ["a test image"]},
    ]
    for index, entry in enumerate(entries):
        Image.new("RGB", (8, 16), (20 + index, 50, 90)).save(images / entry["img_path"])
    (folder / "data_captions.json").write_text(json.dumps(entries), encoding="utf-8")
    return folder.parent


@pytest.fixture
def catalog_file(original_data, tmp_path):
    path = tmp_path / "offline" / "catalog.json"
    export_catalog("RSTPReid", original_data, path)
    return path


ATTRS = {"a.png": {"upper_clothing": "a red checked shirt"},
         "b.png": {"upper_clothing": "a blue striped shirt"}}


def test_catalog_exports_only_original_train_sources_and_preserves_original_ids(catalog_file):
    catalog = load_catalog(catalog_file)
    assert catalog["version"] == 1 and catalog["dataset"] == "RSTPReid"
    assert [source["source_image_id"] for source in catalog["sources"]] == ["a.png", "b.png"]
    assert all(source["source_pid"] == 37 and source["split"] == "train" for source in catalog["sources"])
    assert all(Path(source["image_path"]).is_absolute() for source in catalog["sources"])
    assert catalog["sources"][0]["captions"] == ["a red original", "the original red shirt"]


def test_cached_text_is_deterministic_and_resume_does_not_rewrite(catalog_file, tmp_path):
    output = tmp_path / "text.json"
    first = augment_text(catalog_file, output, attributes=ATTRS, seed=44)
    before = output.read_bytes(), output.stat().st_mtime_ns
    assert augment_text(catalog_file, output, attributes=copy.deepcopy(ATTRS), seed=44) == first
    assert before == (output.read_bytes(), output.stat().st_mtime_ns)
    assert len(first["records"]) == 2
    assert first["records"][0]["attributes"] == ATTRS["a.png"]
    assert all(len(set(row["captions"])) == 2 for row in first["records"])
    for changed in [{"seed": 45}, {"attributes": {**ATTRS, "a.png": {"upper_clothing": "yellow shirt"}}}]:
        options = dict(attributes=ATTRS, seed=44)
        options.update(changed)
        with pytest.raises(ValueError, match="Resume configuration differs"):
            augment_text(catalog_file, output, **options)


def test_paid_provider_resume_uses_saved_valid_attributes_without_second_request(catalog_file, tmp_path):
    class FakeProvider:
        backend = "qwen-vl"
        model = "offline-provider"
        calls = []

        def extract_attributes(self, path):
            self.calls.append(path)
            return {"upper_clothing": "a green shirt"}

    provider = FakeProvider()
    output = tmp_path / "provider-text.json"
    first = augment_text(catalog_file, output, provider=provider, limit=1)
    assert len(provider.calls) == 1
    assert augment_text(catalog_file, output, provider=provider, limit=1) == first
    assert len(provider.calls) == 1
    damaged = copy.deepcopy(first)
    damaged["records"][0]["captions"][0] = "a corrupted sentence"
    atomic_write_json(output, damaged)
    with pytest.raises(ValueError, match="Saved caption record differs"):
        augment_text(catalog_file, output, provider=provider, limit=1)
    assert len(provider.calls) == 1


def test_text_progress_survives_provider_interruption(catalog_file, tmp_path):
    class Interruptible:
        backend = "fake"
        model = "test"
        calls = []
        fail = True

        def extract_attributes(self, path):
            self.calls.append(Path(path).name)
            if Path(path).name == "b.png" and self.fail:
                raise RuntimeError("interrupted fake provider")
            return ATTRS[Path(path).name]

    provider = Interruptible()
    output = tmp_path / "interrupted.json"
    with pytest.raises(RuntimeError, match="interrupted"):
        augment_text(catalog_file, output, provider=provider)
    assert len(json.loads(output.read_text(encoding="utf-8"))["records"]) == 1
    provider.fail = False
    result = augment_text(catalog_file, output, provider=provider)
    assert len(result["records"]) == 2
    assert provider.calls == ["a.png", "b.png", "b.png"]


def test_import_reports_duplicate_error_and_nontraining_sources_without_basename_guess(catalog_file, tmp_path):
    legacy = {"a.png": ["a red original", "new second caption"],
              "b.png": ["same caption", "same   caption"],
              "val.png": ["val one", "val two"],
              "directory/a.png": ["a guessed one", "a guessed two"]}
    result = import_captions(catalog_file, legacy, tmp_path / "imported.json")
    assert result["records"] == []
    assert len(result["report"]["skipped_invalid_sources"]) == 2
    assert result["report"]["ignored_nontraining_keys"] == 2
    with pytest.raises(ValueError, match="Unusable legacy captions"):
        import_captions(catalog_file, legacy, tmp_path / "strict.json", strict=True)
    legacy["a.png"] = ["API call failed", "not a description"]
    legacy["b.png"] = ["a generated blue outfit", "the pictured blue top"]
    result = import_captions(catalog_file, legacy, tmp_path / "good-import.json")
    assert [row["source_image_id"] for row in result["records"]] == ["b.png"]
    assert result["report"]["imported_sources"] == 1
    assert "error" in result["report"]["skipped_invalid_sources"][0]["reason"].lower()


def test_large_local_import_commits_once_after_validation(catalog_file, tmp_path, monkeypatch):
    catalog = load_catalog(catalog_file)
    source = catalog["sources"][0]
    catalog["sources"] = [{**source, "source_image_id": "source-{}.png".format(index)} for index in range(50)]
    legacy = {row["source_image_id"]: ["a new generated caption", "a second generated description"] for row in catalog["sources"]}
    writes = []
    original_write = workflow_module.atomic_write_json

    def record_write(output, payload):
        writes.append(len(payload["records"]))
        original_write(output, payload)

    monkeypatch.setattr(workflow_module, "atomic_write_json", record_write)
    result = import_captions(catalog, legacy, tmp_path / "bulk-import.json")
    assert len(result["records"]) == 50
    assert writes == [0, 50]


class FakePipeline:
    _offline_augmentation_provenance = {"backend": "offline-fixture"}

    def __init__(self):
        self.calls = []
        self.fail_at = None

    def __call__(self, **kwargs):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("fake generation interrupted")
        self.calls.append(kwargs["generator"].initial_seed())
        value = self.calls[-1] % 255
        return SimpleNamespace(images=[Image.new("RGB", (8, 16), (value, 50, 90))], nsfw_content_detected=None)


METADATA = {"placeholder_tokens": {"0037": "<ped37>"}, "subject_random_ids": {"0037": "rid37"}}


def fake_encoder(paths):
    result = []
    for path in paths:
        if path.name == "a.png":
            result.append([1., 0.])
        elif path.name == "b.png":
            result.append([0., 1.])
        else:
            index = int(path.stem.split("_")[-1])
            result.append([1., 0.] if index % 2 == 0 else [0., 1.])
    return torch.tensor(result)


def test_end_to_end_fake_images_filter_generated_own_captions_and_merge(catalog_file, original_data, tmp_path):
    folder = tmp_path / "offline"
    candidates_path = folder / "candidates.json"
    pipeline = FakePipeline()
    candidates = generate_images(catalog_file, candidates_path, pipeline=pipeline, metadata=METADATA,
                                 subject_map={"37": "0037"}, height=16, width=8)
    assert len(candidates["candidates"]) == len(pipeline.calls) == 40
    assert all("captions" not in row for row in candidates["candidates"])
    assert generate_images(catalog_file, candidates_path, pipeline=pipeline, metadata=METADATA,
                           subject_map={"37": "0037"}, height=16, width=8) == candidates
    assert len(pipeline.calls) == 40
    selected_path = folder / "selected.json"
    selected = filter_images(catalog_file, candidates_path, selected_path, feature_encoder=fake_encoder)
    assert len(selected["records"]) == 20
    for record in selected["records"]:
        index = record["provenance"]["candidate_index"]
        assert index % 2 == (record["source_image_id"] == "b.png")
        assert record["selected"] is True and "captions" not in record
        assert not Path(record["image_path"]).is_absolute()
    own_attributes = {row["image_path"]: {"upper_clothing": "a generated emerald shirt",
                                         "carried_items": "a generated gray bag"}
                      for row in selected["records"]}
    image_text_path = folder / "image-text.json"
    image_text = caption_images(catalog_file, selected_path, image_text_path, attributes=own_attributes, seed=9)
    assert len(image_text["records"]) == 20
    for row in image_text["records"]:
        assert row["source_pid"] == 37 and row["kind"] == "image" and row["selected"]
        assert row["attributes"] == own_attributes[row["image_path"]]
        assert all("a generated emerald shirt" in caption and "a generated gray bag" in caption for caption in row["captions"])
        assert row["provenance"]["caption_image_path"] == row["image_path"]
    pairs = {tuple(row["provenance"]["captioning"]["template_ids"]) for row in image_text["records"]}
    assert len(pairs) > 1  # Seeded by each GENERATED artifact, not its source ID.
    assert caption_images(catalog_file, selected_path, image_text_path, attributes=own_attributes, seed=9) == image_text
    text_path = folder / "text.json"
    augment_text(catalog_file, text_path, attributes=ATTRS)
    merged = merge_manifests([text_path, image_text_path], tmp_path / "ready.json", catalog_path=catalog_file)
    assert merged["report"] == {"text_records": 2, "image_records": 20, "skipped_unselected": 0}
    assert len(merged["records"]) == 22
    assert all((tmp_path / row["image_path"]).is_file() for row in merged["records"] if row["kind"] == "image")
    assert all(row["image_path"].startswith("offline/") for row in merged["records"] if row["kind"] == "image")
    from datasets.rstpreid import RSTPReid
    original = RSTPReid(root=original_data, verbose=False)
    augmented = RSTPReid(root=original_data, augmentation_manifest=tmp_path / "ready.json",
                         use_llm_augment=True, use_generated=True, verbose=False)
    assert len(augmented.train_records) == 48  # 4 human + 4 generated text + 40 generated-image captions.
    assert augmented.label_map == {37: 0}
    assert augmented.val == original.val and augmented.test == original.test


def test_filter_handles_a_small_generation_run_with_the_full_source_catalog(catalog_file, tmp_path):
    folder = tmp_path / "offline"
    candidates_path = folder / "small-candidates.json"
    generate_images(catalog_file, candidates_path, pipeline=FakePipeline(), metadata=METADATA,
                    subject_map={"37": "0037"}, limit=1, height=16, width=8)
    selected_path = folder / "small-selected.json"
    selected = filter_images(catalog_file, candidates_path, selected_path, feature_encoder=fake_encoder)
    assert len(selected["records"]) == 10
    assert {row["source_image_id"] for row in selected["records"]} == {"a.png"}
    assert filter_images(catalog_file, candidates_path, selected_path, feature_encoder=fake_encoder) == selected
    with pytest.raises(ValueError, match="fewer than keep"):
        filter_images(catalog_file, candidates_path, folder / "too-many.json", feature_encoder=fake_encoder, keep=21)
    with pytest.raises(ValueError, match="No candidates"):
        filter_images(catalog_file, candidates_path, folder / "empty.json", feature_encoder=fake_encoder,
                      source_ids=["b.png"])


def test_candidate_manifest_commits_each_image_before_interruption(catalog_file, tmp_path):
    pipeline = FakePipeline()
    pipeline.fail_at = 2
    output = tmp_path / "progress" / "candidates.json"
    options = dict(pipeline=pipeline, metadata=METADATA, subject_map={"37": "0037"},
                   num_candidates=3, limit=1, height=16, width=8)
    with pytest.raises(RuntimeError, match="interrupted"):
        generate_images(catalog_file, output, **options)
    assert len(json.loads(output.read_text(encoding="utf-8"))["candidates"]) == 2
    pipeline.fail_at = None
    result = generate_images(catalog_file, output, **options)
    assert len(result["candidates"]) == len(pipeline.calls) == 3
    with pytest.raises(ValueError, match="Resume configuration differs"):
        generate_images(catalog_file, output, **dict(options, seed=14))


def test_generation_rejects_ambiguous_injected_identity_bindings(catalog_file, tmp_path):
    with pytest.raises(ValueError, match="injected pipeline"):
        generate_images(catalog_file, tmp_path / "bad.json", metadata=METADATA)
    catalog = load_catalog(catalog_file)
    catalog["sources"][1]["source_pid"] = 38
    with pytest.raises(ValueError, match="Distinct source PIDs"):
        generate_images(catalog, tmp_path / "bad-map.json", pipeline=FakePipeline(), metadata=METADATA,
                        subject_map={"37": "0037", "38": "0037"})


def test_caption_images_requires_generated_path_cache_keys_and_contained_output(catalog_file, tmp_path):
    folder = tmp_path / "offline"
    candidate_path = folder / "candidates.json"
    generate_images(catalog_file, candidate_path, pipeline=FakePipeline(), metadata=METADATA,
                    subject_map={"37": "0037"}, num_candidates=2, limit=1, height=16, width=8)
    selected_path = folder / "selected.json"
    filter_images(catalog_file, candidate_path, selected_path, feature_encoder=fake_encoder, keep=1, limit=1)
    with pytest.raises(ValueError, match="No cached visual attributes"):
        caption_images(catalog_file, selected_path, folder / "wrong-cache.json", attributes=ATTRS, limit=1)
    with pytest.raises(ValueError, match="inside the output manifest"):
        caption_images(catalog_file, selected_path, folder / "elsewhere" / "captions.json", attributes=ATTRS)


def test_merge_rejects_duplicate_and_incomplete_active_records(catalog_file, tmp_path):
    text = tmp_path / "text.json"
    augment_text(catalog_file, text, attributes=ATTRS)
    with pytest.raises(ValueError, match="Duplicate text"):
        merge_manifests([text, text], tmp_path / "duplicate.json", catalog_path=catalog_file)
    selected = {"version": 1, "dataset": "RSTPReid", "records": [{"kind": "image", "selected": True,
        "source_image_id": "a.png", "source_pid": 37, "image_path": "not-generated.png"}]}
    path = tmp_path / "incomplete.json"
    atomic_write_json(path, selected)
    with pytest.raises(ValueError, match="two nonempty"):
        merge_manifests([path], tmp_path / "bad.json", catalog_path=catalog_file)
    selected["records"][0]["selected"] = False
    atomic_write_json(path, selected)
    merged = merge_manifests([path], tmp_path / "unselected.json", catalog_path=catalog_file)
    assert merged["records"] == [] and merged["report"]["skipped_unselected"] == 1


def test_cli_catalog_text_and_exact_source_limit(original_data, tmp_path, capsys):
    catalog_path = tmp_path / "cli-catalog.json"
    main(["catalog", "--root_dir", str(original_data), "--output", str(catalog_path)])
    attrs = tmp_path / "attrs.json"
    atomic_write_json(attrs, ATTRS)
    output = tmp_path / "cli-text.json"
    result = main(["text", "--catalog", str(catalog_path), "--attributes", str(attrs),
                   "--source_id", "b.png", "--limit", "1", "--output", str(output)])
    assert [row["source_image_id"] for row in result["records"]] == ["b.png"]
    assert "text: 1 records" in capsys.readouterr().out
    options = build_parser().parse_args(["images", "--catalog", "a", "--output", "b",
                                        "--local_model_dir", "c", "--checkpoint_dir", "d"])
    assert options.num_candidates == 20
    options = build_parser().parse_args(["filter", "--catalog", "a", "--output", "b",
                                        "--candidates", "c", "--clip_weights", "d"])
    assert options.keep == 10
