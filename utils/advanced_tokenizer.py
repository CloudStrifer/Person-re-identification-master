import gzip
import html
import os
import random
from functools import lru_cache
from typing import List, Tuple, Dict, Set

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class AdvancedTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe(),
                 attr_dict_path: str = './data/RSTPReid/phrases.txt',
                 mask_prob: float = 0.15,
                 phrase_min_len: int = 2,
                 phrase_max_len: int = 5):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))

        vocab.pop(-1)  # 保持词汇表大小不变
        vocab.extend(['<|mask|>', '<|startoftext|>', '<|endoftext|>'])  # 词汇表大小 49408
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|mask|>': '<|mask|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

        # 短语掩码相关参数
        self.mask_token = self.encoder['<|mask|>']
        self.mask_prob = mask_prob  # 总体掩码概率
        self.phrase_min_len = phrase_min_len  # 短语最小长度
        self.phrase_max_len = phrase_max_len  # 短语最大长度
        self.single_token_mask_prob = 0.3  # 单个token掩码占总掩码的比例
        self.phrase_mask_prob = 1 - self.single_token_mask_prob  # 短语掩码占总掩码的比例

        # 外部知识：属性词典
        self.attribute_phrases = self._load_attribute_dictionary(attr_dict_path) if attr_dict_path else set()
        self.attribute_token_sequences = self._preprocess_attribute_phrases()

    def _load_attribute_dictionary(self, path: str) -> Set[str]:
        """加载属性词典，格式为每行一个属性短语"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"属性词典文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            phrases = {line.strip().lower() for line in f if line.strip()}

        return phrases

    def _preprocess_attribute_phrases(self) -> List[List[int]]:
        """将属性短语转换为token序列，便于后续识别"""
        return [self.encode(phrase) for phrase in self.attribute_phrases if phrase]

    def _find_attribute_phrases(self, tokens: List[int]) -> List[Tuple[int, int]]:
        """在token序列中找到属性短语的位置（起始和结束索引）"""
        attribute_positions = []

        # 检查每个可能的属性短语
        for attr_tokens in self.attribute_token_sequences:
            attr_len = len(attr_tokens)
            token_len = len(tokens)

            for i in range(token_len - attr_len + 1):
                if tokens[i:i + attr_len] == attr_tokens:
                    attribute_positions.append((i, i + attr_len - 1))

        return attribute_positions

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def random_mask(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """
        对token序列进行随机掩码，包括单个token和连续短语

        参数:
            tokens: 原始token序列

        返回:
            masked_tokens: 掩码后的token序列
            mask_labels: 掩码位置标记（1表示被掩码，0表示未被掩码）
        """
        masked_tokens = tokens.copy()
        mask_labels = [0] * len(tokens)
        n_tokens = len(tokens)
        if n_tokens == 0:
            return masked_tokens, mask_labels

        # 先标记已处理的位置，避免重复掩码
        processed = [False] * n_tokens

        # 1. 优先对属性短语进行掩码（提高属性短语的掩码概率）
        attribute_positions = self._find_attribute_phrases(tokens)
        for start, end in attribute_positions:
            # 属性短语有更高的掩码概率
            if random.random() < self.mask_prob * 1.5 and not any(processed[start:end + 1]):
                for i in range(start, end + 1):
                    masked_tokens[i] = self.mask_token
                    mask_labels[i] = 1
                    processed[i] = True

        # 2. 随机选择连续短语进行掩码
        i = 0
        while i < n_tokens:
            if processed[i]:
                i += 1
                continue

            # 决定是否进行短语掩码
            if random.random() < self.mask_prob * self.phrase_mask_prob:
                # 随机选择短语长度
                max_possible_len = min(self.phrase_max_len, n_tokens - i)
                if max_possible_len < self.phrase_min_len:
                    i += 1
                    continue

                phrase_len = random.randint(self.phrase_min_len, max_possible_len)
                end = i + phrase_len - 1

                # 检查这段是否有已处理的token
                if any(processed[i:end + 1]):
                    i += 1
                    continue

                # 进行短语掩码
                for j in range(i, end + 1):
                    masked_tokens[j] = self.mask_token
                    mask_labels[j] = 1
                    processed[j] = True

                i = end + 1
            else:
                i += 1

        # 3. 对剩余未处理的token进行单个token掩码
        for i in range(n_tokens):
            if not processed[i] and random.random() < self.mask_prob * self.single_token_mask_prob:
                masked_tokens[i] = self.mask_token
                mask_labels[i] = 1
                processed[i] = True

        return masked_tokens, mask_labels
