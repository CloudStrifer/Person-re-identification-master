"""Stage-four captions use real attribute values and never contact paid APIs."""

import copy
import io
import json
import random
from urllib.error import HTTPError, URLError

import pytest

import augmentation.text as text_module
from augmentation.templates import ATTRIBUTE_FIELDS, TEMPLATES
from augmentation.text import (
    AttributeCache, AttributeExtractionError, QwenVLAttributeProvider,
    generate_caption_record, validate_attributes, validate_caption_record,
)


SOURCE = {"source_image_id": "imgs/pedestrian_01.jpg", "source_pid": 87,
          "split": "train", "image_path": "unused.jpg", "captions": ["a human caption"]}
ATTRIBUTES = {"upper_clothing": "a red checked jacket", "lower_clothing": "blue jeans",
              "footwear": "white trainers", "accessories": ["dark glasses"],
              "carried_items": "a small black shoulder bag", "action": "walking"}


def _response(content, **choice_fields):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop", **choice_fields}]}


def test_thirty_explicit_templates_have_every_property_slot_and_no_invented_details():
    assert len(TEMPLATES) == len({template.text for template in TEMPLATES}) == 30
    assert len({template.template_id for template in TEMPLATES}) == 30
    for template in TEMPLATES:
        assert all("{" + field + "}" in template.text for field in ATTRIBUTE_FIELDS)
        rendered = template.render({"upper_clothing": "a red shirt"})
        assert "a red shirt" in rendered
        assert not any(token in rendered for token in ["{", "}", "; ;", ": ;"])
        for invented in ["boots", "skirt", "walking", "standing", "young", "man", "woman", " his ", " her ", "street", "park"]:
            assert invented not in rendered


def test_records_have_two_distinct_templates_real_attributes_and_stable_original_identity():
    source = copy.deepcopy(SOURCE)
    attrs = copy.deepcopy(ATTRIBUTES)
    record = generate_caption_record(source, attrs, seed=73)
    assert record["kind"] == "text"
    assert record["source_image_id"] == SOURCE["source_image_id"]
    assert record["source_pid"] == 87
    assert record["attributes"] == ATTRIBUTES
    assert len(record["captions"]) == len(set(record["captions"])) == 2
    assert len(set(record["provenance"]["template_ids"])) == 2
    for caption in record["captions"]:
        assert caption.strip() and "{" not in caption and "}" not in caption
        for value in ["a red checked jacket", "blue jeans", "white trainers", "dark glasses", "a small black shoulder bag", "walking"]:
            assert value in caption
    assert source == SOURCE and attrs == ATTRIBUTES


def test_unknown_values_are_omitted_without_phantom_attributes():
    attrs = {"upper_clothing": "a red shirt", "gender": "unknown", "age": None,
             "footwear": "not visible", "hair": "", "accessories": ["unknown", "n/a"]}
    record = generate_caption_record(SOURCE, attrs)
    assert record["attributes"] == {"upper_clothing": "a red shirt"}
    for caption in record["captions"]:
        for absent in ["unknown", "None", "footwear", "gender", "age:", "hair", "accessories", "walking", "boots", "his", "her"]:
            assert absent not in caption


def test_deterministic_seed_uses_source_id_and_preserves_global_random_state():
    before = random.getstate()
    original = generate_caption_record(SOURCE, ATTRIBUTES, seed=5)
    assert original == generate_caption_record(SOURCE, ATTRIBUTES, seed=5)
    assert random.getstate() == before
    pairs = {tuple(generate_caption_record(SOURCE, ATTRIBUTES, seed=seed)["provenance"]["template_ids"])
             for seed in range(8)}
    assert len(pairs) > 1
    slashes = dict(SOURCE, source_image_id="imgs\\pedestrian_01.jpg")
    assert generate_caption_record(slashes, ATTRIBUTES, seed=5) == original


@pytest.mark.parametrize("change", [
    {"split": "val"}, {"split": "test"}, {"split": None},
    {"source_pid": True}, {"source_pid": -1}, {"source_pid": "87"},
    {"source_image_id": "../test.jpg"}, {"source_image_id": "/test.jpg"},
    {"source_image_id": "D:/test.jpg"}, {"source_image_id": ""},
])
def test_only_valid_training_source_identity_is_accepted(change):
    with pytest.raises(ValueError):
        generate_caption_record(dict(SOURCE, **change), ATTRIBUTES)


@pytest.mark.parametrize("attrs", [
    {}, {"upper_clothing": "unknown"}, {"error": "server failure"},
    {"upper_clothing": "API call failed: quota exceeded"},
    {"upper_clothing": "Error: missing caption"}, {"upper_clothing": "❌ API调用失败"},
    {"upper_clothing": "[clothing type]"}, {"upper_clothing": "{clothing_type}"},
    {"upper_clothing": "No pedestrian detected"}, {"pedestrian_present": False},
    {"upper_clothing": 12}, {"accessories": ["glasses", 5]},
    {"upper_clothing": "<html>server error</html>"},
])
def test_missing_attributes_and_error_text_never_become_captions(attrs):
    with pytest.raises(ValueError):
        generate_caption_record(SOURCE, attrs)


def test_original_script_aliases_and_nested_provider_schema_are_supported():
    assert validate_attributes({"pedestrian_present": True, "attributes": {
        "clothing type": "a coat", "Pedestrian actions": "walking", "shoes": "black shoes",
    }}) == {"footwear": "black shoes", "clothing_type": "a coat", "action": "walking"}
    with pytest.raises(ValueError, match="Conflicting"):
        validate_attributes({"footwear": "red shoes", "shoes": "black shoes"})


def test_offline_cache_uses_exact_ids_and_copies_values_without_network(tmp_path, monkeypatch):
    def fail_network(*args, **kwargs):
        pytest.fail("The offline cache must never access the network")

    monkeypatch.setattr(text_module, "urlopen", fail_network)
    cache_path = tmp_path / "attributes.json"
    cache_path.write_text(json.dumps({SOURCE["source_image_id"]: ATTRIBUTES}), encoding="utf-8")
    cache = AttributeCache(cache_path)
    attrs = cache.extract_attributes("nonexistent-image.jpg", source_image_id=SOURCE["source_image_id"])
    assert attrs == ATTRIBUTES
    attrs["accessories"].append("fabricated")
    assert cache.extract_attributes(source_image_id=SOURCE["source_image_id"]) == ATTRIBUTES
    with pytest.raises(ValueError, match="No cached"):
        cache.extract_attributes(source_image_id="another/pedestrian_01.jpg")
    with pytest.raises(ValueError, match="exact source_image_id"):
        cache.extract_attributes("pedestrian_01.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        AttributeCache({"imgs/a.jpg": ATTRIBUTES, "imgs\\a.jpg": ATTRIBUTES})


def test_resume_reproduces_captions_and_rejects_config_identity_and_cache_edits():
    record = generate_caption_record(SOURCE, ATTRIBUTES, seed=11)
    assert validate_caption_record(record, SOURCE, seed=11, attrs=ATTRIBUTES) == record
    for options in [{"seed": 12}, {"backend": "qwen-vl"}, {"model": "other"},
                    {"attrs": dict(ATTRIBUTES, upper_clothing="a yellow shirt")}]:
        with pytest.raises(ValueError, match="Saved caption record differs"):
            validate_caption_record(record, SOURCE, **dict({"seed": 11}, **options))
    for key, bad in [("captions", ["API error", "another"]), ("source_pid", 88),
                     ("kind", "image"), ("attributes", {"upper_clothing": "red shirt"})]:
        edited = copy.deepcopy(record)
        edited[key] = bad
        with pytest.raises(ValueError):
            validate_caption_record(edited, SOURCE, seed=11)
    edited = copy.deepcopy(record)
    edited["provenance"]["template_digest"] = "old-template-bank"
    with pytest.raises(ValueError, match="provenance"):
        validate_caption_record(edited, SOURCE, seed=11)


@pytest.mark.parametrize("content", [
    json.dumps({"pedestrian_present": True, **ATTRIBUTES}),
    "```json\n" + json.dumps(ATTRIBUTES) + "\n```",
    [{"type": "text", "text": json.dumps(ATTRIBUTES)}],
])
def test_provider_parses_actual_attribute_json_without_fabricated_captions(content):
    payload = json.dumps(_response(content)).encode("utf-8")
    assert QwenVLAttributeProvider.parse_response(payload) == ATTRIBUTES


@pytest.mark.parametrize("payload", [
    b"not JSON", {"error": {"message": "secret diagnostic"}}, {"choices": []},
    _response("API Error: quota exceeded"), _response('{"pedestrian_present": false}'),
    _response('{}'), _response('{"upper_clothing": "API request failed"}'),
    _response('{"upper_clothing": "red shirt"}', finish_reason="length"),
    _response('{"upper_clothing": "red shirt"}', finish_reason="content_filter"),
    _response([{"type": "image_url", "image_url": "should-not-be-returned"}]),
])
def test_provider_failed_or_incomplete_response_never_returns_descriptions(payload):
    with pytest.raises(AttributeExtractionError, match="invalid or unsuccessful"):
        QwenVLAttributeProvider.parse_response(payload)


def test_provider_is_lazy_and_one_explicit_mocked_request_uses_env_only(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    calls = []

    def fake_transport(request, *, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(_response(json.dumps(ATTRIBUTES))).encode("utf-8"))

    provider = QwenVLAttributeProvider(transport=fake_transport)
    assert calls == []
    with pytest.raises(AttributeExtractionError, match="DASHSCOPE_API_KEY"):
        provider.extract_attributes(tmp_path / "unread.jpg")
    assert calls == []
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-credential")
    image = tmp_path / "person.jpg"
    image.write_bytes(b"offline-fixture-image")
    assert provider.extract_attributes(image) == ATTRIBUTES
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url.endswith("/chat/completions")
    assert request.get_header("Authorization") == "Bearer offline-test-credential"
    body = json.loads(request.data)
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert timeout == 60
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", [
    URLError("contains secret data and image payload"),
    HTTPError("https://example.invalid", 429, "quota and private data", {}, None),
    TimeoutError("private timeout data"),
])
def test_provider_errors_are_sanitized_and_never_retried(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-credential")
    image = tmp_path / "person.png"
    image.write_bytes(b"offline-fixture-image")
    calls = []

    def fail_once(request, *, timeout):
        calls.append(request)
        raise failure

    with pytest.raises(AttributeExtractionError) as caught:
        QwenVLAttributeProvider(transport=fail_once).extract_attributes(image)
    assert len(calls) == 1
    assert not any(secret in str(caught.value) for secret in ["private", "payload", "credential", "quota"])


def test_provider_validates_size_and_credential_before_any_request(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Invalid local input must not consume a provider request")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-credential")
    image = tmp_path / "person.png"
    image.write_bytes(b"too large")
    with pytest.raises(AttributeExtractionError, match="byte limit"):
        QwenVLAttributeProvider(transport=no_network, max_image_bytes=2).extract_attributes(image)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "private\ncredential")
    with pytest.raises(AttributeExtractionError, match="invalid whitespace"):
        QwenVLAttributeProvider(transport=no_network).extract_attributes(image)
