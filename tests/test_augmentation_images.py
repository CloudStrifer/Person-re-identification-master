"""Small offline fixtures, not Stable Diffusion samples or benchmark results.

The LoRA and embedding checks use real torch modules. The generation callback
only creates tiny seeded PIL fixtures to test prompts, resumption and lineage.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from augmentation.images import (
    candidate_seed, create_pipeline, fold_legacy_lora, generate_candidates,
    install_learned_embeddings, load_legacy_bundle, resolve_identity_binding,
    select_candidates, selected_image_records, validate_identity_metadata,
)


def metadata():
    return {"subject_ids": ["10", "40"],
            "placeholder_tokens": {"10": "<10>", "40": "<40>"},
            "subject_random_ids": {"10": "random10", "40": "random40"}}


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(3, 3, bias=False)
        self.to_k = nn.Linear(2, 3, bias=False)
        self.to_v = nn.Linear(2, 3, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(3, 3, bias=False), nn.Identity()])


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.attn = TinyAttention()

    @property
    def attn_processors(self):
        return {"block.attn.processor": object()}


def adapter_state(unet):
    state = {}
    for index, name in enumerate(("to_q", "to_k", "to_v", "to_out")):
        projection = unet.get_submodule(f"block.attn.{name}" + (".0" if name == "to_out" else ""))
        down = torch.arange(1, 2 * projection.in_features + 1).reshape(2, projection.in_features).float() / 10
        up = torch.arange(1, projection.out_features * 2 + 1).reshape(projection.out_features, 2).float() / 20
        state[f"{name}_lora.down.weight"] = down + index / 10
        state[f"{name}_lora.up.weight"] = up
    return {"block.attn.processor": state}


class TinyTokenizer:
    def __init__(self):
        # An already-installed learned token must keep its current ID.
        self.vocabulary = {"base": 0, "<40>": 1}

    def get_vocab(self):
        return dict(self.vocabulary)

    def add_tokens(self, token):
        if token in self.vocabulary:
            return 0
        self.vocabulary[token] = len(self.vocabulary)
        return 1

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.get(token, -1)

    def encode(self, token, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token)]

    def __len__(self):
        return len(self.vocabulary)


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2, 3)

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, length):
        previous = self.embedding.weight.detach().clone()
        self.embedding = nn.Embedding(length, previous.shape[1])
        with torch.no_grad():
            self.embedding.weight[:previous.shape[0]].copy_(previous)
        return self.embedding


class FixturePipeline:
    """Only a tiny PIL test fixture; does not simulate diffusion quality."""
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        color = tuple(torch.randint(0, 256, (3,), generator=kwargs["generator"]).tolist())
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), color)],
                               nsfw_content_detected=[False])


class AugmentationImageTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".recovery" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _checkpoint(self):
        directory = self.root / "adapter"
        directory.mkdir()
        torch.save(adapter_state(TinyUNet()), directory / "unet_lora.pt")
        torch.save({"<10>": torch.tensor([1., 2., 3.]), "<40>": torch.tensor([4., 5., 6.])},
                   directory / "embeddings.pt")
        (directory / "metadata.json").write_text(json.dumps(metadata()), encoding="utf-8")
        return directory

    def _source(self, identifier="train/10/first", pid=10):
        path = self.root / f"source_{len(list(self.root.glob('source_*.png')))}.png"
        Image.new("RGB", (8, 16), (70, 80, 90)).save(path)
        return {"source_image_id": identifier, "source_pid": pid, "image_path": str(path),
                "split": "train", "captions": ["original caption one", "original caption two"]}

    def test_lora_fold_matches_actual_low_rank_linear_outputs_and_alpha_convention(self):
        for alpha, multiplier in ((None, 1.0), (3., 1.5)):
            with self.subTest(alpha=alpha):
                unet = TinyUNet()
                state = adapter_state(unet)
                previous = copy.deepcopy(unet.state_dict())
                report = fold_legacy_lora(unet, state, network_alpha=alpha)
                self.assertEqual(report["projection_count"], 4)
                self.assertEqual(report["network_alpha"], alpha)
                for name in ("to_q", "to_k", "to_v", "to_out"):
                    module_path = f"block.attn.{name}" + (".0" if name == "to_out" else "")
                    module = unet.get_submodule(module_path)
                    values = torch.arange(2 * module.in_features).reshape(2, module.in_features).float() / 4
                    down = state["block.attn.processor"][f"{name}_lora.down.weight"]
                    up = state["block.attn.processor"][f"{name}_lora.up.weight"]
                    expected = F.linear(values, previous[f"{module_path}.weight"]) + multiplier * F.linear(F.linear(values, down), up)
                    torch.testing.assert_close(module(values), expected)
                with self.assertRaisesRegex(ValueError, "already been folded"):
                    fold_legacy_lora(unet, state)

    def test_all_lora_validation_precedes_any_weight_mutation(self):
        for defect in ("missing_key", "bad_shape", "nonfinite", "processor", "rank"):
            with self.subTest(defect=defect):
                unet = TinyUNet()
                state = adapter_state(unet)
                weights = state["block.attn.processor"]
                if defect == "missing_key":
                    del weights["to_out_lora.up.weight"]
                elif defect == "bad_shape":
                    weights["to_out_lora.up.weight"] = torch.ones(4, 2)
                elif defect == "nonfinite":
                    weights["to_out_lora.up.weight"][0, 0] = float("nan")
                elif defect == "processor":
                    state["missing.attn.processor"] = state.pop("block.attn.processor")
                elif defect == "rank":
                    weights["to_out_lora.up.weight"] = torch.ones(3, 1)
                    weights["to_out_lora.down.weight"] = torch.ones(1, 3)
                before = copy.deepcopy(unet.state_dict())
                with self.assertRaises(ValueError):
                    fold_legacy_lora(unet, state)
                for key, tensor in unet.state_dict().items():
                    torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)

    def test_missing_processor_and_empty_saved_adapter_are_errors(self):
        unet = TinyUNet()
        with self.assertRaises(ValueError):
            fold_legacy_lora(unet, {})
        state = adapter_state(unet)
        unet.other = TinyAttention()
        with patch.object(TinyUNet, "attn_processors", property(lambda self: {
                "block.attn.processor": object(), "other.processor": object()})):
            with self.assertRaisesRegex(ValueError, "missing=.*other.processor"):
                fold_legacy_lora(unet, state)

    def test_embeddings_restore_by_token_text_and_preserve_base_vocabulary(self):
        tokenizer, encoder = TinyTokenizer(), TinyTextEncoder()
        base = encoder.embedding.weight[0].detach().clone()
        vectors = {"<10>": torch.tensor([1., 2., 3.]), "<40>": torch.tensor([4., 5., 6.])}
        ids = install_learned_embeddings(tokenizer, encoder, vectors, metadata())
        self.assertEqual(ids, {"<10>": 2, "<40>": 1})
        torch.testing.assert_close(encoder.embedding.weight[0], base)
        for token in vectors:
            torch.testing.assert_close(encoder.embedding.weight[ids[token]], vectors[token])
        # Installing a complete checkpoint again is safe and does not grow vocabulary.
        install_learned_embeddings(tokenizer, encoder, vectors, metadata())
        self.assertEqual(len(tokenizer), 3)

    def test_missing_or_wrong_embedding_is_rejected_before_tokenizer_mutation(self):
        for vectors in ({"<10>": torch.ones(3)}, {"<10>": torch.ones(4), "<40>": torch.ones(4)},
                        {"<10>": torch.ones(1, 3), "<40>": torch.ones(3)}):
            tokenizer, encoder = TinyTokenizer(), TinyTextEncoder()
            before = encoder.embedding.weight.detach().clone()
            with self.assertRaises(ValueError):
                install_learned_embeddings(tokenizer, encoder, vectors, metadata())
            self.assertEqual(len(tokenizer), 2)
            torch.testing.assert_close(encoder.embedding.weight, before)

    def test_metadata_identity_binding_is_exact_and_never_guessed(self):
        binding = resolve_identity_binding(metadata(), 10)
        self.assertEqual((binding.source_pid, binding.token, binding.random_id), ("10", "<10>", "random10"))
        with self.assertRaisesRegex(ValueError, "no identity binding"):
            resolve_identity_binding(metadata(), "010")
        mapped = resolve_identity_binding(metadata(), "010", subject_id="10")
        self.assertEqual(mapped.source_pid, "010")
        for defect in ("missing_random", "duplicate_token", "wrong_subject_ids", "duplicate_random"):
            invalid = metadata()
            if defect == "missing_random":
                del invalid["subject_random_ids"]["40"]
            elif defect == "duplicate_token":
                invalid["placeholder_tokens"]["40"] = "<10>"
            elif defect == "wrong_subject_ids":
                invalid["subject_ids"] = ["10", "10"]
            else:
                invalid["subject_random_ids"]["40"] = "random10"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_identity_metadata(invalid)

    def test_legacy_bundle_loads_all_three_files_on_cpu_and_fingerprints_them(self):
        checkpoint = self._checkpoint()
        bundle = load_legacy_bundle(checkpoint, expected_subjects=[10, 40])
        self.assertEqual(bundle.embeddings["<10>"].device.type, "cpu")
        self.assertEqual(bundle.identity_for(40).token, "<40>")
        self.assertEqual(set(bundle.provenance["sha256"]), {"unet_lora.pt", "embeddings.pt", "metadata.json"})
        with self.assertRaisesRegex(ValueError, "no identity binding"):
            load_legacy_bundle(checkpoint, expected_subjects=[99])
        (checkpoint / "embeddings.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "embeddings.pt"):
            load_legacy_bundle(checkpoint)

    def test_pipeline_factory_is_lazy_local_only_and_folds_current_projection_weights(self):
        checkpoint = self._checkpoint()
        model = self.root / "sd15"
        model.mkdir()
        (model / "model_index.json").write_text("{}", encoding="utf-8")
        calls = []

        class LocalPipeline:
            def __init__(self):
                self.unet = TinyUNet()
                self.tokenizer, self.text_encoder = TinyTokenizer(), TinyTextEncoder()

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append((path, kwargs))
                return cls()

            def to(self, device):
                self.device = device
                return self

        with patch.dict(sys.modules, {"diffusers": SimpleNamespace(StableDiffusionPipeline=LocalPipeline)}):
            pipeline = create_pipeline(model, checkpoint, device="cpu", dtype="float32", expected_subjects=[10])
        self.assertEqual(calls, [(str(model.resolve()), {"torch_dtype": torch.float32, "local_files_only": True})])
        self.assertTrue(pipeline.unet._offline_legacy_lora_folded)
        self.assertEqual(pipeline._offline_augmentation_provenance["lora"]["scale_convention"], "1")
        self.assertEqual(pipeline.device, "cpu")
        with self.assertRaises(FileNotFoundError):
            create_pipeline("not-a-local-model-hub-id", checkpoint)

    def test_twenty_sequential_candidates_bind_identity_and_resume_without_overwrite(self):
        source = self._source()
        binding = resolve_identity_binding(metadata(), source["source_pid"])
        pipeline = FixturePipeline()
        candidates = generate_candidates(source, pipeline, binding, self.root / "images", height=16, width=8)
        self.assertEqual(len(candidates), 20)
        self.assertEqual(len(pipeline.calls), 20)
        self.assertEqual(len({record["provenance"]["seed"] for record in candidates}), 20)
        self.assertTrue(all("<10>" in call["prompt"] and "random10" in call["prompt"] for call in pipeline.calls))
        self.assertTrue(all("<40>" not in call["prompt"] for call in pipeline.calls))
        self.assertGreater(len({call["prompt"] for call in pipeline.calls}), 1)
        first = Path(candidates[0]["image_path"])
        previous = first.read_bytes()
        timestamps = [(Path(c["image_path"]).stat().st_mtime_ns, Path(c["image_path"]).with_suffix(".json").stat().st_mtime_ns)
                      for c in candidates]
        resumed = generate_candidates(source, pipeline, binding, self.root / "images", height=16, width=8)
        self.assertEqual(resumed, candidates)
        self.assertEqual(len(pipeline.calls), 20)
        self.assertEqual(first.read_bytes(), previous)
        self.assertEqual(timestamps, [(Path(c["image_path"]).stat().st_mtime_ns,
                                      Path(c["image_path"]).with_suffix(".json").stat().st_mtime_ns) for c in resumed])
        with self.assertRaisesRegex(ValueError, "resume metadata differs"):
            generate_candidates(source, pipeline, binding, self.root / "images", seed=17, height=16, width=8)
        first.unlink()
        generate_candidates(source, pipeline, binding, self.root / "images", height=16, width=8)
        self.assertEqual(len(pipeline.calls), 21)
        self.assertEqual(first.read_bytes(), previous)
        self.assertNotIn("captions", candidates[0])

    def test_seed_and_paths_depend_on_original_id_and_generation_order_does_not_matter(self):
        source_a = self._source("first")
        source_b = self._source("second")
        self.assertEqual(candidate_seed("first", 2, 7), candidate_seed("first", 2, 7))
        self.assertNotEqual(candidate_seed("first", 2, 7), candidate_seed("second", 2, 7))
        self.assertNotEqual(candidate_seed("first", 2, 7), candidate_seed("first", 1, 7))
        binding = resolve_identity_binding(metadata(), 10)
        pipeline = FixturePipeline()
        forward = [generate_candidates(s, pipeline, binding, self.root / "forward", num_candidates=2, seed=7, height=16, width=8)
                   for s in (source_a, source_b)]
        reverse = [generate_candidates(s, pipeline, binding, self.root / "reverse", num_candidates=2, seed=7, height=16, width=8)
                   for s in (source_b, source_a)]
        for generated_a, generated_b in zip(forward, reversed(reverse)):
            for first, second in zip(generated_a, generated_b):
                self.assertEqual(first["provenance"], second["provenance"])
                self.assertEqual(Path(first["image_path"]).read_bytes(), Path(second["image_path"]).read_bytes())
        self.assertNotEqual(forward[0][0]["image_path"], forward[1][0]["image_path"])

    def test_progress_callback_runs_after_each_saved_and_verified_resumed_image(self):
        source, pipeline = self._source(), FixturePipeline()
        binding = resolve_identity_binding(metadata(), 10)
        progress = []

        def persist(record):
            image_path = Path(record["image_path"])
            self.assertTrue(image_path.is_file())
            self.assertTrue(image_path.with_suffix(".json").is_file())
            with Image.open(image_path) as opened:
                opened.verify()
            progress.append((record["provenance"]["candidate_index"], len(pipeline.calls)))

        generate_candidates(source, pipeline, binding, self.root / "images", num_candidates=3,
                            height=16, width=8, on_candidate=persist)
        self.assertEqual(progress, [(0, 1), (1, 2), (2, 3)])
        progress.clear()
        generate_candidates(source, pipeline, binding, self.root / "images", num_candidates=3,
                            height=16, width=8, on_candidate=persist)
        self.assertEqual(progress, [(0, 3), (1, 3), (2, 3)])
        self.assertEqual(len(pipeline.calls), 3)

    def test_generation_rejects_nontrain_source_wrong_identity_and_untracked_existing_image(self):
        source, pipeline = self._source(), FixturePipeline()
        with self.assertRaisesRegex(ValueError, "PID"):
            generate_candidates(source, pipeline, resolve_identity_binding(metadata(), 40), self.root / "images")
        invalid = {**source, "split": "test"}
        with self.assertRaisesRegex(ValueError, "training"):
            generate_candidates(invalid, pipeline, resolve_identity_binding(metadata(), 10), self.root / "images")
        self.assertFalse(pipeline.calls)
        binding = resolve_identity_binding(metadata(), 10)
        generated = generate_candidates(source, pipeline, binding, self.root / "images", num_candidates=1, height=16, width=8)
        image = Path(generated[0]["image_path"])
        image.with_suffix(".json").unlink()
        with self.assertRaisesRegex(FileExistsError, "no provenance"):
            generate_candidates(source, pipeline, binding, self.root / "images", num_candidates=1, height=16, width=8)

    def test_cosine_selection_uses_each_original_instead_of_identity_average(self):
        source_a, source_b = self._source("a"), self._source("b")
        candidates = [self.root / f"candidate_{i}.png" for i in range(3)]
        for path in candidates:
            Image.new("RGB", (8, 16)).save(path)
        features = {Path(source_a["image_path"]): [3., 0.], Path(source_b["image_path"]): [0., 5.],
                    candidates[0]: [6., 0.], candidates[1]: [0., 7.], candidates[2]: [1., 1.]}
        calls = []

        def encode(paths):
            calls.append(list(paths))
            return torch.tensor([features[path] for path in paths])

        ranks_a = select_candidates(source_a["image_path"], candidates, encode, k=2, batch_size=1)
        ranks_b = select_candidates(source_b["image_path"], candidates, encode, k=2, batch_size=2)
        self.assertEqual([Path(r["image_path"]) for r in ranks_a], [candidates[0], candidates[2]])
        self.assertEqual([Path(r["image_path"]) for r in ranks_b], [candidates[1], candidates[2]])
        self.assertAlmostEqual(ranks_a[0]["cosine"], 1., places=6)
        self.assertAlmostEqual(ranks_a[1]["cosine"], 2**-0.5, places=6)
        self.assertEqual([r["rank"] for r in ranks_a], [1, 2])
        self.assertEqual(calls[0], [Path(source_a["image_path"])])
        # An identity-average reference would favor candidate 2 for both sources.
        mean = F.normalize(torch.tensor([1., 1.]), dim=0)
        scores_to_mean = F.normalize(torch.tensor([features[path] for path in candidates]), dim=1) @ mean
        self.assertEqual(int(scores_to_mean.argmax()), 2)

    def test_candidate_selection_rejects_bad_features_and_uses_stable_tie_order(self):
        source = self._source()
        candidates = [self._source(f"c{i}")["image_path"] for i in range(3)]
        ranks = select_candidates(source["image_path"], candidates, lambda paths: torch.ones(len(paths), 2), k=2)
        self.assertEqual([r["image_path"] for r in ranks], candidates[:2])
        for encoder in (lambda paths: torch.zeros(len(paths), 2),
                        lambda paths: torch.full((len(paths), 2), float("nan")),
                        lambda paths: torch.ones(2)):
            with self.assertRaises(ValueError):
                select_candidates(source["image_path"], candidates, encoder, k=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            select_candidates(source["image_path"], [candidates[0], candidates[0]], lambda _: None, k=1)

    def test_selected_records_are_relative_and_await_their_own_captions(self):
        source = self._source()
        generated = generate_candidates(source, FixturePipeline(), resolve_identity_binding(metadata(), 10),
                                        self.root / "images", num_candidates=2, height=16, width=8)
        rankings = select_candidates(source["image_path"], [c["image_path"] for c in generated],
                                     lambda paths: torch.ones(len(paths), 2), k=1)
        records = selected_image_records(source, generated, rankings, self.root)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["kind"], "image")
        self.assertTrue(record["selected"])
        self.assertFalse(Path(record["image_path"]).is_absolute())
        self.assertTrue((self.root / record["image_path"]).is_file())
        self.assertEqual(record["source_image_id"], source["source_image_id"])
        self.assertEqual(record["provenance"]["selection_reference"], "individual_original_image")
        self.assertNotIn("captions", record)
        with self.assertRaisesRegex(ValueError, "inside the manifest"):
            selected_image_records(source, generated, rankings, self.root / "elsewhere")


if __name__ == "__main__":
    unittest.main()
