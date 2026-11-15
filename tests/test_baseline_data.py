"""Data contract checks with tiny original-format annotations, no external data."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from datasets.bases import ImageTextMLMDataset, tokenize
from datasets.build import build_loaders, canonical_dataset_name
from datasets.rstpreid import RSTPReid
from datasets.sampler import RandomIdentitySampler
from datasets.synthperson import Synth_PEDES
from utils.simple_tokenizer import SimpleTokenizer


class TinyTokenizer:
    encoder = {"<|mask|>": 90, "<|startoftext|>": 98, "<|endoftext|>": 99}

    def encode(self, caption):
        # Include ordinary byte token ID 0 to check padding is not inferred from ID.
        return [0] + [3 + len(word) for word in caption.split()]


def write_annotations(root, folder, annotation_file, entries, path_key="file_path", image_folder="imgs"):
    data_dir = root / folder
    image_dir = data_dir / image_folder
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries):
        path = image_dir / entry[path_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 16), (index * 17 % 255, 20, 40)).save(path)
    (data_dir / annotation_file).write_text(json.dumps(entries), encoding="utf-8")
    return data_dir


def args_for(root, name="RSTPReid", **overrides):
    values = dict(dataset_name=name, root_dir=str(root), text_length=12,
                  img_size=(16, 8), batch_size=4, test_batch_size=2,
                  img_aug=False, num_workers=0, sampler="random", seed=42,
                  device="cpu", use_llm_augment=False, use_generated=False, MLM=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def rstp_entries(with_val=True):
    entries = [
        {"id": 42, "split": "train", "img_path": "a.jpg", "captions": ["red coat", "red jacket"]},
        {"id": 7, "split": "train", "img_path": "b.jpg", "captions": ["blue coat", "blue shirt"]},
        {"id": 92, "split": "test", "img_path": "test.jpg", "captions": ["black coat"]},
    ]
    if with_val:
        entries.append({"id": 91, "split": "val", "img_path": "val.jpg", "captions": ["white coat"]})
    return entries


def test_selected_dataset_clean_tokens_contiguous_labels_and_metadata(tmp_path):
    write_annotations(tmp_path, "RSTPReid", "data_captions.json", rstp_entries(), path_key="img_path")
    bundle = build_loaders(args_for(tmp_path), tokenizer=TinyTokenizer())
    assert bundle["label_map"] == {7: 0, 42: 1}
    assert bundle["num_classes"] == 2  # four captions, two identities
    dataset = bundle["dataset"]
    assert dataset.train[0][1] == dataset.train[1][1]
    assert dataset.train[0][1] != dataset.train[2][1]
    train_set = bundle["train_set"]
    first = train_set[0]
    expected, mask = tokenize("red coat", TinyTokenizer(), 12, return_attention_mask=True)
    assert torch.equal(first["caption_ids"], expected)
    assert torch.equal(first["attention_mask"], mask)
    assert first["caption_ids"][1].item() == 0 and first["attention_mask"][1]
    assert first["metadata"]["source_pid"] == 42
    assert first["metadata"]["caption_source"] == "human"
    assert torch.equal(train_set[0]["caption_ids"], expected)
    batch = next(iter(bundle["train_loader"]))
    assert batch["images"].shape == (4, 3, 16, 8)
    assert isinstance(batch["metadata"], list)
    assert Counter(batch["pids"].tolist()) == {0: 2, 1: 2}
    assert next(iter(bundle["val_txt_loader"]))[0].tolist() == [91]
    assert next(iter(bundle["test_txt_loader"]))[0].tolist() == [92]


def test_no_validation_is_not_replaced_by_test(tmp_path):
    write_annotations(tmp_path, "RSTPReid", "data_captions.json", rstp_entries(False), path_key="img_path")
    bundle = build_loaders(args_for(tmp_path), tokenizer=TinyTokenizer())
    assert bundle["val_img_loader"] is None and bundle["val_txt_loader"] is None
    assert bundle["test_txt_loader"] is not None
    evaluation = build_loaders(args_for(tmp_path), training=False, tokenizer=TinyTokenizer())
    assert evaluation["train_loader"] is None
    assert next(iter(evaluation["test_txt_loader"]))[0].tolist() == [92]


def test_full_synth_preserves_image_ids_and_source_provenance(tmp_path):
    entries = [{"id": 900, "file_path": "p/a.jpg", "captions": ["red coat", "red shirt"]},
               {"id": 7, "file_path": "p/b.jpg", "captions": ["blue coat"]}]
    write_annotations(tmp_path, "Synth-Pedes", "synthpedes-dataset.json", entries, image_folder="")
    dataset = Synth_PEDES(root=tmp_path, verbose=False)
    assert dataset.label_map == {7: 0, 900: 1}
    assert len(dataset.train) == 3 and len(dataset.train_id_container) == 2
    assert dataset.train[0][1] == dataset.train[1][1]
    assert all(record["caption_source"] == "synthetic" for record in dataset.train_records)
    assert not dataset.test["captions"] and not dataset.val["captions"]
    with pytest.raises(ValueError, match="full SYNTH-PEDES"):
        Synth_PEDES(root=tmp_path, train_split=0.8)


@pytest.mark.parametrize("name,folder,annotation_file", [
    ("CUHK-PEDES", "CUHK-PEDES", "reid_raw.json"),
    ("ICFG-PEDES", "ICFG-PEDES", "ICFG-PEDES.json"),
])
def test_other_original_formats_keep_raw_eval_identities(tmp_path, name, folder, annotation_file):
    entries = [{"id": 101, "split": "train", "file_path": "a.jpg", "captions": ["red coat"]},
               {"id": 205, "split": "test", "file_path": "b.jpg", "captions": ["blue coat"]}]
    write_annotations(tmp_path, folder, annotation_file, entries)
    bundle = build_loaders(args_for(tmp_path, name), tokenizer=TinyTokenizer())
    assert bundle["label_map"] == {101: 0}
    assert bundle["dataset"].test["image_pids"] == [205]
    assert bundle["val_txt_loader"] is None


def test_identity_sampler_has_valid_positive_negative_groups_and_exact_length():
    rows = [(pid, image, f"{pid}_{image}.jpg", "caption", None)
            for pid in (7, 42, 99) for image in (0, 1, 2)]
    sampler = RandomIdentitySampler(rows, batch_size=4, num_instances=2, seed=11)
    indices = list(sampler)
    assert len(indices) == len(sampler) and len(indices) % 4 == 0
    for start in range(0, len(indices), 4):
        counts = Counter(rows[index][0] for index in indices[start:start + 4])
        assert len(counts) == 2 and set(counts.values()) == {2}
    sampler.set_epoch(0)
    assert list(sampler) == indices
    with pytest.raises(ValueError, match="two identities"):
        RandomIdentitySampler(rows, batch_size=2, num_instances=2)


def test_legacy_mlm_does_not_mutate_shared_numpy_source():
    dataset = ImageTextMLMDataset.__new__(ImageTextMLMDataset)
    dataset.tokenizer = TinyTokenizer()
    original = torch.tensor([98, 3, 4, 99, 0, 0])
    before = original.clone()
    dataset._build_random_masked_tokens_and_labels(original.numpy())
    assert torch.equal(original, before)


def test_missing_resources_and_unsupported_augmentation_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="original annotations"):
        build_loaders(args_for(tmp_path), tokenizer=TinyTokenizer())
    with pytest.raises(FileNotFoundError, match="--bpe_path"):
        SimpleTokenizer(tmp_path / "missing.gz")
    with pytest.raises(ValueError, match="augmentation_manifest"):
        build_loaders(args_for(tmp_path, use_llm_augment=True), tokenizer=TinyTokenizer())
    assert canonical_dataset_name("SYNTH-PEDES") == "Synth_PEDES"


def test_source_image_cannot_cross_split_boundary(tmp_path):
    entries = [{"id": 7, "split": "train", "img_path": "a.jpg", "captions": ["red coat"]},
               {"id": 7, "split": "test", "img_path": "a.jpg", "captions": ["red coat"]}]
    write_annotations(tmp_path, "RSTPReid", "data_captions.json", entries, path_key="img_path")
    with pytest.raises(ValueError, match="conflicting split/identity"):
        RSTPReid(root=tmp_path)


@pytest.mark.parametrize("splits", [("train", "val"), ("train", "test"), ("val", "test")])
def test_benchmark_identity_cannot_cross_splits_on_distinct_images(tmp_path, splits):
    entries = [{"id": 7, "split": split, "img_path": f"{split}.jpg", "captions": ["red coat"]}
               for split in splits]
    write_annotations(tmp_path, "RSTPReid", "data_captions.json", entries, path_key="img_path")
    with pytest.raises(ValueError, match="identity-disjoint splits"):
        RSTPReid(root=tmp_path)


def test_synth_respects_explicit_external_manifest(tmp_path):
    entries = [{"id": 7, "split": split, "file_path": f"{split}.jpg", "captions": ["red coat"]}
               for split in ("train", "test")]
    write_annotations(tmp_path, "Synth-Pedes", "synthpedes-dataset.json", entries, image_folder="")
    dataset = Synth_PEDES(root=tmp_path, verbose=False)
    assert len(dataset.train) == 1 and dataset.test["image_pids"] == [7]


def test_evaluation_iterators_leave_global_and_training_rng_unchanged(tmp_path):
    write_annotations(tmp_path, "RSTPReid", "data_captions.json", rstp_entries(), path_key="img_path")
    bundle = build_loaders(args_for(tmp_path), tokenizer=TinyTokenizer())
    before = torch.get_rng_state().clone()
    train_before = bundle["generator"].get_state().clone()
    for key in ("val_img_loader", "val_txt_loader", "test_img_loader", "test_txt_loader"):
        list(bundle[key])
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(bundle["generator"].get_state(), train_before)


def test_truncation_preserves_eot_and_attention_length():
    ids, mask = tokenize("one two three four five", TinyTokenizer(), 4, return_attention_mask=True)
    assert ids[-1] == 99 and mask.tolist() == [True] * 4


def test_official_bpe_preserves_last_ordinary_token_and_control_ids():
    tokenizer = SimpleTokenizer()
    assert len(tokenizer.encoder) == 49408
    assert tokenizer.encoder["jekyll</w>"] == 49405
    assert tokenizer.encode("jekyll") == [49405]
    assert tokenizer.encoder["<|startoftext|>"] == 49406
    assert tokenizer.encoder["<|endoftext|>"] == 49407
    assert "<|mask|>" not in tokenizer.encoder
    # Official CLIP has no mask control token; a literal mask-like string must
    # remain encodable as ordinary text, rather than hitting a missing cache key.
    tokens = tokenizer.encode("a <|mask|> jacket worn by jekyll")
    assert tokens and all(0 <= token < 49406 for token in tokens)
    assert tokens[-1] == 49405


def test_legacy_mask_vocabulary_requires_explicit_opt_in():
    tokenizer = SimpleTokenizer(legacy_mask_token=True)
    assert len(tokenizer.encoder) == 49408
    assert tokenizer.encoder["<|mask|>"] == 49405
    assert tokenizer.encode("<|mask|>") == [49405]
    assert tokenizer.encoder["<|startoftext|>"] == 49406
    assert tokenizer.encoder["<|endoftext|>"] == 49407
