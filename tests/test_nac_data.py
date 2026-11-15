"""Controlled NAC supervision uses actual CLIP BPE without external resources."""

import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from datasets.bases import ImageTextDataset, TextDataset, tokenize
from datasets.build import build_loaders
from datasets.nac_augment import ControlledNACCorruptor, ordinary_token_mask
from utils.simple_tokenizer import SimpleTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return SimpleTokenizer()


def apply_noise(caption, tokenizer, rate=1, groups=None, source="human", length=77):
    clean, attention = tokenize(caption, tokenizer, length, return_attention_mask=True)
    options = {} if groups is None else {"attribute_groups": groups}
    result = ControlledNACCorruptor(tokenizer, rate, **options)(caption, clean, attention, source)
    return clean, attention, result


def test_actual_bpe_color_replacement_keeps_alignment_and_original_tensors(tokenizer):
    caption = "A person wears a RED coat."
    clean, attention = tokenize(caption, tokenizer, return_attention_mask=True)
    original_ids, original_attention = clean.clone(), attention.clone()
    corruptor = ControlledNACCorruptor(tokenizer, 1, (("red", "blue"),))
    result = corruptor(caption, clean, attention)
    expected = tokenize("a person wears a blue coat.", tokenizer)
    assert torch.equal(result["caption_ids"], expected)
    assert torch.equal(result["clean_ids"], original_ids)
    assert torch.equal(clean, original_ids) and torch.equal(attention, original_attention)
    assert result["corruption_mask"].sum() == 1
    assert torch.equal(result["corruption_mask"], expected != original_ids)
    assert torch.equal(result["target_valid_mask"], attention & (clean != 49406) & (clean != 49407))
    assert not result["target_valid_mask"][~attention].any()
    assert not result["corruption_mask"][[0, int(attention.sum()) - 1]].any()
    assert all(token < 49406 for token in expected[result["corruption_mask"]].tolist())


@pytest.mark.parametrize("caption", ["a hundred coating red_coat", "a person walks quickly"])
def test_whole_words_and_no_eligible_attributes_stay_clean(tokenizer, caption):
    clean, _, result = apply_noise(caption, tokenizer, groups=(("red", "blue"), ("coat", "jacket")))
    assert torch.equal(result["caption_ids"], clean)
    assert not result["corruption_mask"].any()
    assert result["target_valid_mask"].any()


def test_unsupported_bpe_lengths_and_truncated_spans_are_skipped(tokenizer):
    assert len(tokenizer.encode("red")) != len(tokenizer.encode("turtleneck"))
    clean, _, result = apply_noise("red", tokenizer, groups=(("red", "turtleneck"),))
    assert torch.equal(result["caption_ids"], clean)
    assert not result["corruption_mask"].any()
    assert len(tokenizer.encode("turtleneck")) == len(tokenizer.encode("multicolored")) == 3
    clean, attention, result = apply_noise("a turtleneck", tokenizer, length=4,
                                         groups=(("turtleneck", "multicolored"),))
    assert torch.equal(result["caption_ids"], clean)
    assert clean[-1] == tokenizer.encoder["<|endoftext|>"] and attention.all()
    assert not result["corruption_mask"].any()


def test_ratio_zero_keeps_clean_supervision_and_does_not_advance_random(tokenizer):
    before = random.getstate()
    clean, _, result = apply_noise("red jacket", tokenizer, rate=0)
    assert random.getstate() == before
    assert torch.equal(result["caption_ids"], clean)
    assert result["target_valid_mask"].sum() == 2
    assert not result["corruption_mask"].any()


@pytest.mark.parametrize("source", ["synthetic", "llm", "unknown", None])
def test_unaligned_sources_do_not_fabricate_supervision_or_noise(tokenizer, source):
    before = random.getstate()
    clean, _, result = apply_noise("red jacket", tokenizer, source=source)
    assert random.getstate() == before
    assert torch.equal(result["caption_ids"], clean)
    assert not result["target_valid_mask"].any()
    assert not result["corruption_mask"].any()


def test_control_and_padding_exclusion_still_accepts_ordinary_token_zero(tokenizer):
    # Token zero can be an attended ordinary CLIP byte, while token 49405 is
    # the ordinary word jekyll in the current pretrained vocabulary.
    ids = torch.tensor([49406, 0, 49405, 49407, 736, 0])
    attention = torch.tensor([True, True, True, True, True, False])
    assert ordinary_token_mask(ids, attention, tokenizer).tolist() == [False, True, True, False, False, False]
    legacy = SimpleTokenizer(legacy_mask_token=True)
    ids = torch.tensor([49406, legacy.encoder["<|mask|>"], 736, 49407, 0])
    mask = ordinary_token_mask(ids, torch.tensor([True] * 4 + [False]), legacy)
    assert mask.tolist() == [False, False, True, False, False]


def test_global_random_state_restores_online_corruption_sequence(tokenizer):
    saved_original = random.getstate()
    try:
        random.seed(27)
        caption = "red coat blue pants black hat green backpack"
        clean, attention = tokenize(caption, tokenizer, return_attention_mask=True)
        corruptor = ControlledNACCorruptor(tokenizer, 0.45)
        corruptor(caption, clean, attention)
        checkpoint_state = random.getstate()
        expected = [corruptor(caption, clean, attention)["caption_ids"] for _ in range(5)]
        expected_state = random.getstate()
        random.setstate(checkpoint_state)
        resumed = ControlledNACCorruptor(tokenizer, 0.45)
        actual = [resumed(caption, clean, attention)["caption_ids"] for _ in range(5)]
        assert all(torch.equal(left, right) for left, right in zip(expected, actual))
        assert random.getstate() == expected_state
        assert any(not torch.equal(left, right) for left, right in zip(actual, actual[1:]))
    finally:
        random.setstate(saved_original)


def test_dataset_preserves_annotations_and_baseline_api(tmp_path, tokenizer):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 16)).save(image)
    row = {"pid": 0, "image_id": 1, "img_path": str(image), "caption": "red jacket",
           "source_pid": 10, "caption_source": "human", "split": "train"}
    before = copy.deepcopy(row)
    baseline = ImageTextDataset([row], tokenizer=tokenizer)[0]
    sample = ImageTextDataset([row], tokenizer=tokenizer, use_nac=True, nac_corruption_rate=1)[0]
    assert row == before
    assert not {"clean_ids", "target_valid_mask", "corruption_mask"} & baseline.keys()
    assert torch.equal(baseline["caption_ids"], sample["clean_ids"])
    assert torch.equal(baseline["attention_mask"], sample["attention_mask"])
    assert sample["corruption_mask"].any()
    assert torch.equal(TextDataset([10], [row["caption"]], tokenizer=tokenizer)[0][1], sample["clean_ids"])
    row["caption_source"] = "llm"
    unaligned = ImageTextDataset([row], tokenizer=tokenizer, use_nac=True, nac_corruption_rate=1)[0]
    assert not unaligned["target_valid_mask"].any()
    assert torch.equal(unaligned["caption_ids"], baseline["caption_ids"])


def test_loader_adds_training_targets_but_keeps_evaluation_clean(tmp_path, tokenizer):
    folder = tmp_path / "RSTPReid"
    (folder / "imgs").mkdir(parents=True)
    entries = []
    for identity, split in enumerate(("train", "val", "test")):
        filename = f"{split}.png"
        Image.new("RGB", (8, 16)).save(folder / "imgs" / filename)
        entries.append({"id": identity, "split": split, "img_path": filename, "captions": ["red coat"]})
    (folder / "data_captions.json").write_text(json.dumps(entries), encoding="utf-8")
    args = SimpleNamespace(dataset_name="RSTPReid", root_dir=str(tmp_path), batch_size=1,
                           test_batch_size=1, num_workers=0, device="cpu", use_nac=True,
                           nac_corruption_rate=1)
    transform = lambda image: torch.zeros(3, 16, 8)
    bundle = build_loaders(args, tokenizer=tokenizer, transforms=transform)
    batch = next(iter(bundle["train_loader"]))
    for key in ("caption_ids", "clean_ids", "target_valid_mask", "corruption_mask", "attention_mask"):
        assert batch[key].shape == (1, 77)
    assert batch["target_valid_mask"].dtype == batch["corruption_mask"].dtype == torch.bool
    assert batch["corruption_mask"].any()
    for key in ("val_txt_loader", "test_txt_loader"):
        assert torch.equal(next(iter(bundle[key]))[1][0], batch["clean_ids"][0])


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_corruption_rates_fail_clearly(tokenizer, rate):
    with pytest.raises(ValueError, match="nac_corruption_rate"):
        ControlledNACCorruptor(tokenizer, rate)
