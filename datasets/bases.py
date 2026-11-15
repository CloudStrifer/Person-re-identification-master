from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from PIL import Image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import numpy as np


def read_image(img_path):
    """Read once and surface invalid or missing images instead of retrying forever."""
    try:
        with Image.open(img_path) as image:
            return image.convert('RGB')
    except (OSError, ValueError) as error:
        raise OSError(f"Cannot read pedestrian image: {img_path}") from error

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")
    #这个修改能统计加入的生成图像数据
    def show_dataset_info(self):
        """
        统计数据集信息
        train 是列表格式: [(pid, image_id, img_path, caption, sim), ...]
        test/val 是 dict 格式: {"image_pids": [...], "img_paths": [...], ...}
        """

        # ========== 统计 TRAIN 集 ==========
        # train 是列表，每个元素是 (pid, image_id, img_path, caption, sim)
        num_train_pids = len(self.train_id_container)

        # 统计唯一的image_id数量 = 图像数
        train_image_ids = set()
        train_captions_count = 0
        for item in self.train:
            train_image_ids.add(item[1])  # image_id 在索引1
            train_captions_count += 1

        num_train_imgs = len(train_image_ids)
        num_train_captions = train_captions_count

        # ========== 统计 TEST 集 ==========
        # test 是 dict 格式
        num_test_pids = len(self.test_id_container)
        num_test_imgs = len(self.test['image_pids'])
        num_test_captions = len(self.test['captions'])

        # ========== 统计 VAL 集 ==========
        # val 是 dict 格式
        num_val_pids = len(self.val_id_container)
        num_val_imgs = len(self.val['image_pids'])
        num_val_captions = len(self.val['captions'])

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))

    # def show_dataset_info(self):
    #     num_train_pids, num_train_imgs, num_train_captions = len(
    #         self.train_id_container), len(self.train_annos), len(self.train)
    #     num_test_pids, num_test_imgs, num_test_captions = len(
    #         self.test_id_container), len(self.test_annos), len(
    #             self.test['captions'])
    #     num_val_pids, num_val_imgs, num_val_captions = len(
    #         self.val_id_container), len(self.val_annos), len(
    #             self.val['captions'])
    #
    #     # TODO use prettytable print comand line table
    #
    #     self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
    #     table = PrettyTable(['subset', 'ids', 'images', 'captions'])
    #     table.add_row(
    #         ['train', num_train_pids, num_train_imgs, num_train_captions])
    #     table.add_row(
    #         ['test', num_test_pids, num_test_imgs, num_test_captions])
    #     table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
    #     self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True,
             return_attention_mask=False):
    if text_length < 2:
        raise ValueError("text_length must allow both start and end tokens")
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    if return_attention_mask:
        # Token ID zero is also a valid CLIP byte token, so mask by length.
        attention_mask = torch.arange(text_length) < len(tokens)
        return result, attention_mask
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 tokenizer=None,
                 bpe_path=None,
                 use_nac=False,
                 nac_corruption_rate=0.15):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer(bpe_path)
        self.nac_corruptor = None
        if use_nac:
            # Keep NAC's training-only rule construction outside evaluation and
            # the unchanged baseline data path.
            from .nac_augment import ControlledNACCorruptor
            self.nac_corruptor = ControlledNACCorruptor(self.tokenizer, nac_corruption_rate)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, dict):
            pid, image_id, img_path, caption = (item[k] for k in ('pid', 'image_id', 'img_path', 'caption'))
            metadata = {k: item[k] for k in ('source_pid', 'source_image_id', 'split',
                                             'caption_source', 'sample_id', 'generated_image_id',
                                             'augmentation_provenance') if k in item}
        else:
            pid, image_id, img_path, caption = item[:4]
            metadata = {'source_pid': pid, 'source_image_id': str(image_id),
                        'split': 'train', 'caption_source': 'human'}
        metadata['img_path'] = img_path
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens, attention_mask = tokenize(
            caption, tokenizer=self.tokenizer, text_length=self.text_length,
            truncate=self.truncate, return_attention_mask=True)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
            'attention_mask': attention_mask,
            'metadata': metadata,
        }
        if self.nac_corruptor is not None:
            ret.update(self.nac_corruptor(caption, tokens, attention_mask,
                                         metadata.get('caption_source', 'unknown')))

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True,
                 tokenizer=None,
                 bpe_path=None):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer(bpe_path)

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption

def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        # Reference-only MLM path uses the old experiment's explicit mask layout.
        self.tokenizer = SimpleTokenizer(legacy_mask_token=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption, sim = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # 调用已实现的噪声生成方法
        noisy_tokens, original_tokens = self._create_noisy_text(caption_tokens)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy().copy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'noisy_tokens': noisy_tokens,  # 添加带噪声的token
            'original_tokens': original_tokens,  # 添加原始token用于计算损失
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    # 人为制造噪声
    def _create_noisy_text(self, caption_tokens):
        """创建带噪声的文本，用于训练生成器"""
        tokens = caption_tokens.clone().cpu().numpy()
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 有效token范围

        # 随机选择15%的token进行处理
        num_tokens = sum(1 for t in tokens if t not in [0, self.tokenizer.encoder["<|startoftext|>"],
                                                        self.tokenizer.encoder["<|endoftext|>"]])
        num_mask = max(1, int(num_tokens * 0.15))  # 至少mask一个token

        # 选择要处理的token位置
        valid_positions = [i for i, t in enumerate(tokens) if t not in [0, self.tokenizer.encoder["<|startoftext|>"],
                                                                        self.tokenizer.encoder["<|endoftext|>"]]]
        positions = random.sample(valid_positions, num_mask)

        original_tokens = tokens.copy()  # 保存原始token用于计算损失

        for pos in positions:
            prob = random.random()
            if prob < 0.8:
                # 80%概率替换为[MASK]
                tokens[pos] = mask
            elif prob < 0.9:
                # 10%概率替换为随机token
                tokens[pos] = random.choice(token_range) if token_range else tokens[pos]
            # 10%概率保持不变

        return torch.tensor(tokens), torch.tensor(original_tokens)

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        tokens = np.array(tokens, copy=True)
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range) if token_range else token

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)


class FilterDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer(legacy_mask_token=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption, sim = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
                                  truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy(), sim)
        ori_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'caption_ids_ori': ori_tokens
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens, sim):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        tokens = np.array(tokens, copy=True)
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405

        labels = []

        if tokens[-1] == 0:
            valid_token_num = np.where(tokens == 0)[0][0]
        else:
            valid_token_num = len(tokens)
        ori_sim = np.array(sim)
        ori_pro = 1 - ori_sim
        if ori_pro[-1] != 0.15:
            valid_prob = ori_pro[1:valid_token_num - 1]
            # normalize the probisibility to match E = 0.15
            mean_prob = np.mean(valid_prob)
            normed_prob = valid_prob - mean_prob
            normalized_prob = normed_prob + 0.15
            normalized_prob = np.clip(normalized_prob, 0, 1)
            ori_pro[1:valid_token_num - 1] = normalized_prob

        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < ori_pro[i]:
                    prob /= ori_pro[i]

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range) if token_range else token

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

