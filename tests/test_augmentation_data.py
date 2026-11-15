"""Offline augmentation respects source identities and evaluation boundaries."""

import copy
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from datasets.augmentation import apply_augmentation
from datasets.bases import tokenize
from datasets.build import build_loaders
from datasets.rstpreid import RSTPReid
from utils.simple_tokenizer import SimpleTokenizer


class TinyTokenizer:
    encoder = {"<|startoftext|>": 98, "<|endoftext|>": 99}

    def encode(self, caption):
        return [3 + len(word) for word in caption.split()]


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "RSTPReid"
    image_dir = folder / "imgs"
    image_dir.mkdir(parents=True)
    entries = [
        {"id": 42, "split": "train", "img_path": "a.png", "captions": ["red coat", "red jacket"]},
        {"id": 7, "split": "train", "img_path": "b.png", "captions": ["blue coat", "blue jacket"]},
        {"id": 91, "split": "val", "img_path": "val.png", "captions": ["white coat", "white jacket"]},
        {"id": 92, "split": "test", "img_path": "test.png", "captions": ["black coat", "black jacket"]},
    ]
    for index, entry in enumerate(entries):
        Image.new("RGB", (8, 16), (index * 10, 20, 40)).save(image_dir / entry["img_path"])
    (folder / "data_captions.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


def make_manifest(root, records=None, **overrides):
    folder = root / "offline"
    folder.mkdir(exist_ok=True)
    Image.new("RGB", (8, 16), (100, 80, 40)).save(folder / "generated.png")
    if records is None:
        records = [text_record(), image_record()]
    manifest = {"version": 1, "dataset": "RSTPReid", "records": records, **overrides}
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def text_record(**changes):
    return {"kind": "text", "source_image_id": "a.png", "source_pid": 42,
            "captions": ["a pedestrian in a red coat", "a person wearing a red jacket"],
            "provenance": {"model": "offline-fixture"}, **changes}


def image_record(**changes):
    return {"kind": "image", "source_image_id": "a.png", "source_pid": 42,
            "image_path": "generated.png", "selected": True,
            "captions": ["a red coat on this generated person", "this person has a red jacket"],
            "provenance": {"rank": 1, "score": 0.8, "seed": 42}, **changes}


def options(root, **changes):
    return SimpleNamespace(dataset_name="RSTPReid", root_dir=str(root), batch_size=4,
                           test_batch_size=2, num_workers=0, device="cpu", text_length=16,
                           **changes)


def load(root, **changes):
    return build_loaders(options(root, **changes), tokenizer=TinyTokenizer(),
                         transforms=lambda image: torch.zeros(3, 16, 8))


@pytest.mark.parametrize("text_enabled,image_enabled,total", [(True, False, 6), (False, True, 6), (True, True, 8)])
def test_merge_flags_preserve_originals_labels_image_ids_and_eval(source, text_enabled, image_enabled, total):
    baseline = RSTPReid(root=source, verbose=False)
    before = copy.deepcopy(baseline.train_records)
    manifest = make_manifest(source)
    bundle = load(source, augmentation_manifest=str(manifest),
                  use_llm_augment=text_enabled, use_generated=image_enabled)
    dataset = bundle["dataset"]
    assert dataset.train_records[:4] == before
    assert len(dataset.train_records) == total
    assert dataset.label_map == baseline.label_map == {7: 0, 42: 1}
    assert dataset.train_id_container == {0, 1}
    assert dataset.val == baseline.val and dataset.test == baseline.test
    assert dataset.train_annos == baseline.train_annos
    assert dataset.train[0][1] == dataset.train[1][1] == 0
    assert dataset.train[2][1] == dataset.train[3][1] == 1
    for row in dataset.train_records[4:]:
        assert row["pid"] == 1 and row["source_pid"] == 42 and row["source_image_id"] == "a.png"
        assert row["split"] == "train"
        if row["caption_source"] == "llm":
            assert row["image_id"] == 0 and row["img_path"] == before[0]["img_path"]
        else:
            assert row["caption_source"] == "synthetic"
            assert row["image_id"] == 2
            assert row["img_path"] == str((manifest.parent / "generated.png").resolve())
            assert row["generated_image_id"].startswith("generated:")
            assert row["augmentation_provenance"]["seed"] == 42
    signature = dataset.augmentation_signature
    assert signature["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert signature["text_records"] == int(text_enabled)
    assert signature["image_records"] == int(image_enabled)
    assert signature["added_captions"] == total - 4


def test_generated_ids_are_stable_across_text_flag_and_have_distinct_captions(source):
    manifest = make_manifest(source)
    first = load(source, augmentation_manifest=manifest, use_generated=True)["dataset"]
    both = load(source, augmentation_manifest=manifest, use_llm_augment=True, use_generated=True)["dataset"]
    assert [(row["image_id"], row["generated_image_id"], row["caption"])
            for row in first.train_records if "generated_image_id" in row] == [
                (row["image_id"], row["generated_image_id"], row["caption"])
                for row in both.train_records if "generated_image_id" in row]
    assert {row["caption"] for row in first.train_records[4:]} == set(image_record()["captions"])


def test_augmented_captions_have_no_fabricated_nac_targets_or_corruption(source):
    manifest = make_manifest(source)
    tokenizer = SimpleTokenizer()
    bundle = build_loaders(options(source, augmentation_manifest=manifest, use_llm_augment=True,
                                   use_generated=True, use_nac=True, nac_corruption_rate=1),
                           tokenizer=tokenizer, transforms=lambda image: torch.zeros(3, 16, 8))
    for index in range(4, 8):
        item = bundle["train_set"][index]
        caption = bundle["dataset"].train_records[index]["caption"]
        expected = tokenize(caption, tokenizer, text_length=16)
        assert torch.equal(item["caption_ids"], expected)
        assert torch.equal(item["clean_ids"], expected)
        assert not item["target_valid_mask"].any() and not item["corruption_mask"].any()
        assert item["metadata"]["source_image_id"] == "a.png"
    assert bundle["train_set"][6]["metadata"]["generated_image_id"].startswith("generated:")
    assert bundle["train_set"][6]["metadata"]["augmentation_provenance"]["rank"] == 1


def test_evaluation_completely_ignores_saved_augmentation_flags_and_paths(source):
    args = options(source, augmentation_manifest=source / "missing.json", use_llm_augment=True, use_generated=True)
    bundle = build_loaders(args, training=False, tokenizer=TinyTokenizer(), transforms=lambda image: torch.zeros(3, 16, 8))
    assert bundle["train_loader"] is None
    assert bundle["dataset"].augmentation_signature is None
    assert len(bundle["dataset"].train) == 4
    assert next(iter(bundle["test_txt_loader"]))[0].tolist() == [92, 92]


def test_disabled_flags_do_not_read_the_manifest(source):
    bundle = load(source, augmentation_manifest=source / "does-not-exist.json")
    assert len(bundle["dataset"].train) == 4 and bundle["dataset"].augmentation_signature is None


@pytest.mark.parametrize("flag", ["use_llm_augment", "use_generated"])
def test_enabled_flags_require_an_explicit_manifest(source, flag):
    with pytest.raises(ValueError, match="--augmentation_manifest"):
        load(source, **{flag: True})


def test_direct_annotations_supports_flags_and_merge_is_atomic(source):
    manifest = make_manifest(source)
    direct = RSTPReid(root=source, verbose=False, augmentation_manifest=manifest, use_llm_augment=True)
    assert len(direct.train) == 6
    clean = RSTPReid(root=source, verbose=False)
    original = copy.deepcopy(clean.train_records)
    bad = make_manifest(source, [text_record(), text_record(captions=["third caption", "a pedestrian in a red coat"])])
    with pytest.raises(ValueError, match="Duplicate source-caption"):
        apply_augmentation(clean, bad, use_llm_augment=True)
    assert clean.train_records == original and clean.augmentation_signature is None


@pytest.mark.parametrize("overrides,message", [
    ({"source_pid": 7}, "source_pid"),
    ({"source_pid": "42"}, "source_pid"),
    ({"source_pid": True}, "source_pid"),
    ({"source_image_id": "test.png", "source_pid": 92}, "original train image"),
    ({"source_image_id": "val.png", "source_pid": 91}, "original train image"),
    ({"source_image_id": "missing.png"}, "original train image"),
    ({"source_image_id": "../a.png"}, "relative path"),
    ({"source_image_id": "./a.png"}, "relative path"),
    ({"split": "test"}, "must belong to train"),
    ({"captions": ["only one"]}, "exactly two"),
    ({"captions": [" ", "valid"]}, "exactly two"),
    ({"captions": ["same", "same"]}, "Duplicate source-caption"),
    ({"captions": ["red coat", "another caption"]}, "Duplicate source-caption"),
    ({"provenance": []}, "provenance"),
])
def test_invalid_text_source_caption_and_schema_fail(source, overrides, message):
    path = make_manifest(source, [text_record(**overrides)])
    with pytest.raises(ValueError, match=message):
        load(source, augmentation_manifest=path, use_llm_augment=True)


@pytest.mark.parametrize("overrides,message", [
    ({"version": 2}, "version"), ({"version": True}, "version"),
    ({"dataset": "CUHK-PEDES"}, "dataset"), ({"dataset": []}, "dataset"),
    ({"records": {}}, "records list"),
    ({"records": [{"kind": "unknown"}]}, "kind"),
])
def test_invalid_manifest_header_fails(source, overrides, message):
    path = make_manifest(source, **overrides)
    with pytest.raises(ValueError, match=message):
        load(source, augmentation_manifest=path, use_llm_augment=True)


@pytest.mark.parametrize("image_path", ["../outside.png", "/outside.png", "C:\\outside.png", "./generated.png"])
def test_generated_paths_must_be_relative_and_contained(source, image_path):
    path = make_manifest(source, [image_record(image_path=image_path)])
    with pytest.raises(ValueError, match="relative path"):
        load(source, augmentation_manifest=path, use_generated=True)


def test_missing_generated_image_and_missing_own_captions_fail(source):
    path = make_manifest(source, [image_record(image_path="missing.png")])
    with pytest.raises(FileNotFoundError, match="Missing generated image"):
        load(source, augmentation_manifest=path, use_generated=True)
    record = image_record()
    del record["captions"]
    path = make_manifest(source, [record])
    with pytest.raises(ValueError, match="own captions"):
        load(source, augmentation_manifest=path, use_generated=True)


def test_unselected_images_and_disabled_kinds_can_be_incomplete(source):
    record = {"kind": "image", "selected": False}
    path = make_manifest(source, [text_record(), record])
    assert len(load(source, augmentation_manifest=path, use_llm_augment=True)["dataset"].train) == 6
    assert len(load(source, augmentation_manifest=path, use_generated=True)["dataset"].train) == 4
    del record["selected"]
    path = make_manifest(source, [record])
    assert len(load(source, augmentation_manifest=path, use_generated=True)["dataset"].train) == 4


def test_duplicate_generated_paths_and_original_files_are_rejected(source):
    path = make_manifest(source, [image_record(), image_record()])
    with pytest.raises(ValueError, match="Duplicate generated image_path"):
        load(source, augmentation_manifest=path, use_generated=True)
    # Windows case aliases expose one file under different path spellings.
    alias_name = "GENERATED.png"
    if os.name != "nt":
        alias_name = "alias.png"
        os.link(path.parent / "generated.png", path.parent / alias_name)
    path = make_manifest(source, [image_record(), image_record(image_path=alias_name)])
    with pytest.raises(ValueError, match="Duplicate generated image_path"):
        load(source, augmentation_manifest=path, use_generated=True)
    path = source / "RSTPReid" / "imgs" / "manifest.json"
    path.write_text(json.dumps({"version": 1, "dataset": "RSTPReid",
                                "records": [image_record(image_path="a.png")]}), encoding="utf-8")
    with pytest.raises(ValueError, match="original dataset image"):
        load(source, augmentation_manifest=path, use_generated=True)
