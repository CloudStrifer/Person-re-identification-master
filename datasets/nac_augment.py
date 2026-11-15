"""Position-aligned, ordinary attribute replacements for NAC training (D02).

Each eligible attribute span is selected independently with ``corruption_rate``.
This is an online training convention, not an attempt to align LLM rewrites.
All random draws use Python's global RNG, which the training checkpoint and
DataLoader worker initialization already manage.
"""

import math
import random

import regex as re
import torch

from utils.simple_tokenizer import basic_clean, whitespace_clean


# Keep alternatives within explicit semantic groups. Equal BPE length and actual
# context alignment are checked with the supplied tokenizer, never assumed.
ATTRIBUTE_GROUPS = (
    ("black", "white", "red", "blue", "green", "yellow", "brown", "gray",
     "grey", "pink", "purple", "orange"),
    ("shirt", "jacket", "coat", "sweater", "hoodie", "blouse", "vest"),
    ("pants", "trousers", "jeans", "shorts", "skirt"),
    ("shoes", "boots", "sneakers", "sandals"),
    ("hat", "cap", "beanie"),
    ("bag", "backpack", "handbag", "purse"),
)


def ordinary_token_mask(token_ids, attention_mask, tokenizer):
    """Identify ordinary positions, including the valid CLIP byte token ID zero."""
    mask = attention_mask.to(dtype=torch.bool).clone()
    if mask.shape != token_ids.shape or token_ids.ndim != 1:
        raise ValueError("NAC data inputs must be matching one-dimensional token/mask tensors")
    mask[0] = False
    eot = tokenizer.encoder["<|endoftext|>"]
    # Literal control tokens inside an annotation must not create supervision
    # beyond the first sentence terminator, even when length-mask entries exist.
    mask &= (token_ids == eot).cumsum(dim=0) == 0
    for token in ("<|startoftext|>", "<|endoftext|>", "<|mask|>"):
        token_id = tokenizer.encoder.get(token)
        if token_id is not None:
            mask &= token_ids != token_id
    return mask


class ControlledNACCorruptor:
    """Construct clean targets and optional aligned noise without changing inputs.

    Only explicit ``human`` annotations are treated as known clean text. A
    synthetic or LLM caption can still participate in retrieval/correction, but
    has no fabricated token targets and receives no controlled corruption.
    """

    def __init__(self, tokenizer, corruption_rate=0.15, attribute_groups=ATTRIBUTE_GROUPS):
        self.tokenizer = tokenizer
        self.corruption_rate = float(corruption_rate)
        if not math.isfinite(self.corruption_rate) or not 0 <= self.corruption_rate <= 1:
            raise ValueError("nac_corruption_rate must be between 0 and 1")
        self.control_ids = {tokenizer.encoder[token] for token in
                            ("<|startoftext|>", "<|endoftext|>", "<|mask|>")
                            if token in tokenizer.encoder}
        self.replacements = {}
        for group in attribute_groups:
            encoded = {word: tuple(tokenizer.encode(word)) for word in group}
            for word, ids in encoded.items():
                if not self._ordinary_ids(ids):
                    continue
                alternatives = [(other_word, other) for other_word, other in encoded.items()
                                if other != ids and len(other) == len(ids)
                                and self._ordinary_ids(other)]
                if alternatives:
                    self.replacements[word] = (ids, list(dict.fromkeys(alternatives)))
        # Unicode letter/number boundaries also avoid partial matches such as
        # red in hundred, coat in coating, or attributes joined by underscores.
        self.pattern = (re.compile(r"(?<![\p{L}\p{N}_])(?:" +
                        "|".join(re.escape(word) for word in
                                 sorted(self.replacements, key=len, reverse=True)) +
                        r")(?![\p{L}\p{N}_])") if self.replacements else None)

    def _ordinary_ids(self, ids):
        return bool(ids) and all(token >= 0 and token not in self.control_ids for token in ids)

    def __call__(self, caption, clean_ids, attention_mask, caption_source="human"):
        clean_ids = clean_ids.clone()
        noisy_ids = clean_ids.clone()
        target_valid = ordinary_token_mask(clean_ids, attention_mask, self.tokenizer)
        if str(caption_source).strip().lower() != "human":
            target_valid.zero_()
        elif self.corruption_rate and self.pattern is not None:
            normalized = whitespace_clean(basic_clean(caption)).lower()
            full_ids = self.tokenizer.encode(normalized)
            for match in self.pattern.finditer(normalized):
                source_ids, alternatives = self.replacements[match.group()]
                prefix = self.tokenizer.encode(normalized[:match.start()])
                suffix = self.tokenizer.encode(normalized[match.end():])
                # Token offsets must agree with the actual whole caption; this
                # also makes unfamiliar injected tokenizers safely skip spans.
                if prefix + list(source_ids) + suffix != full_ids:
                    continue
                start, end = 1 + len(prefix), 1 + len(prefix) + len(source_ids)
                if end > len(clean_ids) or not target_valid[start:end].all():
                    continue
                if clean_ids[start:end].tolist() != list(source_ids):
                    continue
                alternatives = [ids for word, ids in alternatives
                                if self.tokenizer.encode(normalized[:match.start()] + word +
                                                         normalized[match.end():]) ==
                                prefix + list(ids) + suffix]
                if not alternatives:
                    continue
                if random.random() < self.corruption_rate:
                    replacement = random.choice(alternatives)
                    noisy_ids[start:end] = torch.tensor(replacement, dtype=clean_ids.dtype,
                                                        device=clean_ids.device)
        return {"caption_ids": noisy_ids, "clean_ids": clean_ids,
                "target_valid_mask": target_valid,
                "corruption_mask": (noisy_ids != clean_ids) & target_valid}
