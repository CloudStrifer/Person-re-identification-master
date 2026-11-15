# Multi-subject training script for LoRA with DreamBooth + Text Inversion + Prior Preservation
# 添加类先验保留损失 - 防止过拟合和灾难性遗忘

import argparse
import hashlib
import itertools
import math
import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import inspect
from pathlib import Path
from typing import Optional, List, Dict
import random
import json
import string

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# ============ 使用 diffusers 官方 LoRA 实现 ============
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Sampler
from torchvision import transforms

logger = get_logger(__name__)

# ============ 双占位符模板 (更强的表达能力) ============
PEDESTRIAN_TEMPLATES = [
    "a {} photo of a {}",
    "a {} photo of a {} walking",
    "a {} photo of a {} standing",
    "a {} photo of a {} from the side",
    "a {} photo of a {} from behind",
    "a {} photo of a {} in different pose",
    "a {} cropped photo of the {}",
    "a {} full body photo of the {}",
    "a {} photo of the {} from different angle",
    "a {} bright photo of the {}",
    "a {} good quality photo of the {}",
    "a {} photo of the nice {}",
    "a {} high resolution photo of the {}",
]

# ============ 类别通用描述模板 (用于先验样本) ============
CLASS_TEMPLATES = [
    "a photo of a person",
    "a rendering of a person",
    "a cropped photo of the person",
    "the photo of a person",
    "a good photo of a person",
    "a photo of one person",
    "a photo of the nice person",
    "a bright photo of the person",
]


def generate_random_identifier(length=8):
    """生成随机标识符"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


# ============ 推荐训练参数 ============
RECOMMENDED_ARGS = {
    "resolution": (512, 256),
    "train_batch_size": 4,
    "num_train_epochs": 80,
    "lora_rank": 16,
    "learning_rate": 2e-4,
    "learning_rate_ti": 5e-4,
    "max_train_steps": 3000,
    "save_steps": 500,
    "gradient_accumulation_steps": 1,
    "prior_loss_weight": 1.0,  # 先验损失权重
    "num_class_images": 200,  # 类别先验图片数量
}


# class MultiSubjectDataset(Dataset):
#     """DreamBooth风格数据集 - 每个subject用独立的placeholder token"""
#
#     def __init__(
#             self,
#             subject_dirs: Dict[str, str],
#             placeholder_tokens: Dict[str, str],
#             tokenizer,
#             size: tuple = (512, 256),
#             center_crop: bool = True,
#             color_jitter: bool = False,
#             resize: bool = True,
#     ):
#         self.size = size
#         self.center_crop = center_crop
#         self.tokenizer = tokenizer
#         self.resize = resize
#         self.placeholder_tokens = placeholder_tokens
#         self.templates = PEDESTRIAN_TEMPLATES
#
#         self.image_paths = []
#         self.subject_ids = []
#         self.valid_images = 0
#
#         # ============ 为每个subject生成固定的随机ID ============
#         self.subject_random_ids = {}
#         for subject_id in subject_dirs.keys():
#             self.subject_random_ids[subject_id] = generate_random_identifier(length=8)
#             logger.info(f"Subject {subject_id} assigned random ID: {self.subject_random_ids[subject_id]}")
#
#         for subject_id, directory in subject_dirs.items():
#             subject_path = Path(directory)
#             if not subject_path.exists():
#                 logger.warning(f"Subject directory {subject_path} does not exist, skipping...")
#                 continue
#
#             image_files = list(subject_path.glob("*.jpg")) + \
#                     list(subject_path.glob("*.png")) + \
#                     list(subject_path.glob("*.jpeg"))
#
#             if len(image_files) == 0:
#                 logger.warning(f"No images found in {subject_path}, skipping...")
#                 continue
#
#             valid_files = []
#             for img_file in sorted(image_files):
#                 try:
#                     img = Image.open(img_file)
#                     valid_files.append(img_file)
#                 except Exception as e:
#                     logger.warning(f"Cannot open {img_file}: {e}")
#
#             if len(valid_files) < 3:
#                 logger.warning(f"Subject {subject_id} has less than 3 valid images")
#                 continue
#
#             for img_file in valid_files:
#                 self.image_paths.append(img_file)
#                 self.subject_ids.append(subject_id)
#
#             self.valid_images += len(valid_files)
#
#         if len(self.image_paths) == 0:
#             raise ValueError("No valid images found in any subject directory!")
#
#         self._length = len(self.image_paths)
#
#         # ============ 数据增强管道 ============
#         transform_list = [
#             transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ]
#
#         self.image_transforms = transforms.Compose(transform_list)
#         logger.info(f"Loaded {self.valid_images} valid images for {len(subject_dirs)} subjects")
#
#     def __len__(self):
#         return self._length
#
#     def __getitem__(self, index):
#         example = {}
#         image = Image.open(self.image_paths[index])
#         if not image.mode == "RGB":
#             image = image.convert("RGB")
#
#         example["instance_images"] = self.image_transforms(image)
#
#         subject_id = self.subject_ids[index]
#         placeholder_token = self.placeholder_tokens[subject_id]
#         random_id = self.subject_random_ids[subject_id]
#
#         content = f"{placeholder_token} person"
#         template = random.choice(self.templates)
#         text = template.format(random_id, content)
#
#         example["instance_prompt_ids"] = self.tokenizer(
#             text,
#             padding="do_not_pad",
#             truncation=True,
#             max_length=self.tokenizer.model_max_length,
#         ).input_ids
#
#         example["subject_id"] = subject_id
#         return example

class MultiSubjectDataset(Dataset):
    """改进版：所有样本在一个大列表中，完全随机混合，类似 HuggingFace Dataset"""

    def __init__(
            self,
            subject_dirs: Dict[str, str],
            placeholder_tokens: Dict[str, str],
            tokenizer,
            size: tuple = (512, 256),
            center_crop: bool = True,
            color_jitter: bool = False,
            resize: bool = True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize
        self.placeholder_tokens = placeholder_tokens
        self.templates = PEDESTRIAN_TEMPLATES

        # ============ 关键改变：所有样本先收集到一个平面列表 ============
        self.all_image_paths = []
        self.all_subject_ids = []
        self.subject_to_indices = {}  # 建立 subject -> indices 映射，但只用于调试

        # ============ 为每个subject生成固定的随机ID ============
        self.subject_random_ids = {}
        for subject_id in subject_dirs.keys():
            self.subject_random_ids[subject_id] = generate_random_identifier(length=8)
            logger.info(f"Subject {subject_id} assigned random ID: {self.subject_random_ids[subject_id]}")

        # ============ 收集所有图片（不分块存储） ============
        valid_images_count = 0

        for subject_id, directory in subject_dirs.items():
            subject_path = Path(directory)
            if not subject_path.exists():
                logger.warning(f"Subject directory {subject_path} does not exist, skipping...")
                continue

            image_files = list(subject_path.glob("*.jpg")) + \
                          list(subject_path.glob("*.png")) + \
                          list(subject_path.glob("*.jpeg"))

            if len(image_files) == 0:
                logger.warning(f"No images found in {subject_path}, skipping...")
                continue

            valid_files = []
            for img_file in sorted(image_files):
                try:
                    img = Image.open(img_file)
                    valid_files.append(img_file)
                except Exception as e:
                    logger.warning(f"Cannot open {img_file}: {e}")

            if len(valid_files) < 3:
                logger.warning(f"Subject {subject_id} has less than 3 valid images")
                continue

            # ============ 记录映射关系（用于诊断） ============
            self.subject_to_indices[subject_id] = []

            for img_file in valid_files:
                self.all_image_paths.append(img_file)
                self.all_subject_ids.append(subject_id)
                self.subject_to_indices[subject_id].append(len(self.all_image_paths) - 1)
                valid_images_count += 1

        if len(self.all_image_paths) == 0:
            raise ValueError("No valid images found in any subject directory!")

        # ============ 新增：过采样每个类到 15 张（解决数据不足问题）============
        logger.info("Starting oversampling...")
        oversampled_count = 0

        for subject_id in self.subject_to_indices.keys():
            indices = self.subject_to_indices[subject_id]
            current_count = len(indices)

            if current_count < 15:
                shortage = 15 - current_count
                # 用 random.choices 随机重复选择这个 subject 的样本
                additional_indices = random.choices(indices, k=shortage)

                # 将重复的样本添加到列表末尾
                for idx in additional_indices:
                    self.all_image_paths.append(self.all_image_paths[idx])
                    self.all_subject_ids.append(subject_id)
                    oversampled_count += 1

                logger.info(f"Subject {subject_id}: {current_count} → 15 images (added {shortage})")

        logger.info(f"Oversampling completed. Added {oversampled_count} duplicated images")

        # ============ 关键：随机打乱所有样本（彻底破坏块状结构） ============
        shuffled_indices = list(range(len(self.all_image_paths)))
        random.shuffle(shuffled_indices)

        self.all_image_paths = [self.all_image_paths[i] for i in shuffled_indices]
        self.all_subject_ids = [self.all_subject_ids[i] for i in shuffled_indices]

        # 更新 subject_to_indices（重新映射）
        for subject_id in self.subject_to_indices.keys():
            self.subject_to_indices[subject_id] = [
                i for i, sid in enumerate(self.all_subject_ids) if sid == subject_id
            ]

        logger.info(f"Loaded {valid_images_count} valid images for {len(subject_dirs)} subjects")
        logger.info(f"After oversampling and shuffle, total images: {len(self.all_image_paths)}")
        logger.info(f"After shuffle, samples are fully mixed (not blocked by subject)")

        # ============ 打印采样分布检查 ============
        logger.info("Sample distribution (first 50 samples):")
        first_50_subjects = self.all_subject_ids[:50]
        logger.info(f"  Subjects in first 50: {first_50_subjects}")

        self._length = len(self.all_image_paths)

        # ============ 数据增强管道 ============
        transform_list = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]

        self.image_transforms = transforms.Compose(transform_list)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image = Image.open(self.all_image_paths[index])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["instance_images"] = self.image_transforms(image)

        subject_id = self.all_subject_ids[index]
        placeholder_token = self.placeholder_tokens[subject_id]
        random_id = self.subject_random_ids[subject_id]

        content = f"{placeholder_token} person"
        template = random.choice(self.templates)
        text = template.format(random_id, content)

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        example["subject_id"] = subject_id
        return example


# ============ 新增: 类别先验数据集 ============
class ClassPriorDataset(Dataset):
    """
    类别先验保留数据集

    用途：提供通用的"person"图片，防止模型忘记什么是普通人
    这些图片不包含特定subject，只用于保持类别的通用性
    """

    def __init__(
            self,
            class_data_root: str,
            tokenizer,
            size: tuple = (512, 256),
            num_class_images: int = 200,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.class_data_root = Path(class_data_root)
        self.templates = CLASS_TEMPLATES

        # ============ 收集所有类别图片 ============
        self.class_images_path = []

        if not self.class_data_root.exists():
            logger.warning(f"Class data directory {self.class_data_root} does not exist!")
            logger.warning("Prior preservation will be disabled.")
        else:
            image_files = list(self.class_data_root.glob("*.jpg")) + \
                    list(self.class_data_root.glob("*.png")) + \
                    list(self.class_data_root.glob("*.jpeg"))

            for img_file in sorted(image_files)[:num_class_images]:
                try:
                    img = Image.open(img_file)
                    self.class_images_path.append(img_file)
                except Exception as e:
                    logger.warning(f"Cannot open class image {img_file}: {e}")

        if len(self.class_images_path) == 0:
            logger.warning("No class images found! Prior preservation will be disabled.")
        else:
            logger.info(f"Loaded {len(self.class_images_path)} class prior images")

        # ============ 数据增强 ============
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.class_images_path)

    def __getitem__(self, index):
        example = {}
        image = Image.open(self.class_images_path[index])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["class_images"] = self.image_transforms(image)

        # 随机选择类别描述模板
        class_prompt = random.choice(self.templates)

        example["class_prompt_ids"] = self.tokenizer(
            class_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example


def get_subject_directories(organize_dir: str) -> Dict[str, str]:
    """获取所有subject目录"""
    organize_path = Path(organize_dir)
    subject_dirs = {}

    for item in sorted(organize_path.iterdir()):
        if item.is_dir():
            subject_dirs[item.name] = str(item)

    if len(subject_dirs) == 0:
        raise ValueError(f"No subject directories found in {organize_dir}")

    logger.info(f"Found {len(subject_dirs)} subject directories: {list(subject_dirs.keys())}")
    return subject_dirs


def setup_subject_tokens(
        subject_dirs: Dict[str, str],
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        initializer_token: str,
) -> tuple:
    """为每个subject设置placeholder token"""
    placeholder_token_ids = {}
    placeholder_tokens = {}

    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")
    initializer_token_id = token_ids[0]

    logger.info(f"Initializer token '{initializer_token}' has ID: {initializer_token_id}")

    for subject_id in subject_dirs.keys():
        placeholder_token = f"<{subject_id}>"

        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            logger.warning(f"Token {placeholder_token} already exists, reusing...")

        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        placeholder_token_ids[subject_id] = placeholder_token_id
        placeholder_tokens[subject_id] = placeholder_token

        logger.info(f"Subject '{subject_id}' -> token '{placeholder_token}' (ID: {placeholder_token_id})")

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data

    with torch.no_grad():
        for subject_id, token_id in placeholder_token_ids.items():
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            logger.info(f"Initialized <{subject_id}> (ID: {token_id}) with initializer token embedding")

    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
    for token_id in placeholder_token_ids.values():
        index_no_updates[token_id] = False

    logger.info(f"Created embedding mask: {index_no_updates.sum()} frozen, "
                f"{(~index_no_updates).sum()} trainable")

    return placeholder_token_ids, placeholder_tokens, index_no_updates


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Multi-subject LoRA training with DreamBooth + Prior Preservation")

    parser.add_argument(
        "--organize_dir",
        type=str,
        default='./Organize_Pedestrians-300',
        help="Path to directory containing subject directories",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default='./Prior_Images',
        help="Path to directory containing class prior images (通用行人图片)",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/workspace/GuangjinOuyang/DiffusionStable/stable-diffusion",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multi_subject_lora_output",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default="person",
        help="Initializer token for text embedding",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 256],
        #default=[384, 192],
        help="Training image resolution (height width)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training (instance images)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=650,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        #default=15000,
        default=150000,
        help="Total training steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for LoRA",
    )
    parser.add_argument(
        "--learning_rate_ti",
        type=float,
        default=5e-4,
        help="Learning rate for text inversion",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="LoRA rank",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--use_xformers",
        action="store_true",
        help="Use xformers for memory efficiency",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Apply color jitter to images",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        default=False,
        help="Center crop images",
    )
    # ============ 新增: 先验保留相关参数 ============
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        default=True,
        help="Enable prior preservation loss",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=0.1,
        help="Weight of prior preservation loss (先验损失权重)",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=400,
        help="Number of class images for prior preservation (建议100-500张)",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.resolution = tuple(args.resolution)
    return args


def main(args):
    logging_dir = Path(args.output_dir) / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ============ 加载模型 ============
    subject_dirs = get_subject_directories(args.organize_dir)
    logger.info(f"Training {len(subject_dirs)} subjects: {list(subject_dirs.keys())}")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    placeholder_token_ids, placeholder_tokens, index_no_updates = setup_subject_tokens(
        subject_dirs,
        tokenizer,
        text_encoder,
        args.initializer_token,
    )

    if args.use_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"xformers not available: {e}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    # ============ 注入 LoRA ============
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.lora_rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    # ============ 创建实例数据集 (你的目标行人) ============
    train_dataset = MultiSubjectDataset(
        subject_dirs=subject_dirs,
        placeholder_tokens=placeholder_tokens,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
    )

    # ============ 创建类别先验数据集 (通用行人) ============
    class_dataset = None
    if args.with_prior_preservation:
        class_dataset = ClassPriorDataset(
            class_data_root=args.class_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            num_class_images=args.num_class_images,
        )

        if len(class_dataset) == 0:
            logger.warning("No class images found, disabling prior preservation")
            args.with_prior_preservation = False
            class_dataset = None

    # ============ 定义 collate 函数 ============
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        # ============ 新增: 添加类别先验数据 ============
        if args.with_prior_preservation and class_dataset is not None:
            # 从类别数据集中随机采样
            class_indices = torch.randint(0, len(class_dataset), (len(examples),))
            class_examples = [class_dataset[i] for i in class_indices]

            class_input_ids = [example["class_prompt_ids"] for example in class_examples]
            class_pixel_values = [example["class_images"] for example in class_examples]

            class_pixel_values = torch.stack(class_pixel_values)
            class_pixel_values = class_pixel_values.to(memory_format=torch.contiguous_format).float()

            class_input_ids = tokenizer.pad(
                {"input_ids": class_input_ids},
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            batch["class_input_ids"] = class_input_ids
            batch["class_pixel_values"] = class_pixel_values

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0,
    )

    # ============ Setup optimizer ============
    params_to_optimize = [
        {"params": lora_layers.parameters(), "lr": args.learning_rate},
        {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate_ti},
    ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    # ============ Setup scheduler ============
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    # ✅ 计算实际训练步数
    steps_based_on_epochs = args.num_train_epochs * num_update_steps_per_epoch

    if args.max_train_steps is None:
        actual_max_steps = steps_based_on_epochs
        logger.info(f"max_train_steps not set, using epochs: {actual_max_steps} steps")
    else:
        actual_max_steps = min(args.max_train_steps, steps_based_on_epochs)
        if args.max_train_steps > steps_based_on_epochs:
            logger.warning(
                f"max_train_steps ({args.max_train_steps}) > steps from epochs ({steps_based_on_epochs}). "
                f"Training will stop at {steps_based_on_epochs} steps."
            )

    # ✅ 更新为实际值
    args.max_train_steps = actual_max_steps


    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.max_train_steps,
    )

    # ============ Prepare with accelerator ============
    unet, text_encoder, lora_layers, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        unet, text_encoder, lora_layers, train_dataloader, optimizer, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    #text_encoder.text_model.embeddings.token_embedding.weight.data = text_encoder.get_input_embeddings().weight.data.float()

    text_encoder_unwrapped = accelerator.unwrap_model(text_encoder)
    text_encoder_unwrapped.text_model.embeddings.token_embedding.weight.data = (
        text_encoder_unwrapped.get_input_embeddings().weight.data.float()
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0

    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    unet.train()
    text_encoder.train()

    # ============ 训练循环 (带先验保留) ============
    for epoch in range(args.num_train_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            try:
                # ============ 显存监控 ============
                if global_step % 50 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved = torch.cuda.memory_reserved() / 1024 ** 3
                    logger.info(
                        f"Step {global_step} | "
                        f"Allocated: {allocated:.2f}GB | "
                        f"Reserved: {reserved:.2f}GB"
                    )

                # ============ 处理实例图片 (你的目标行人) ============
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # ============ 计算实例损失 ============
                with accelerator.accumulate(unet):
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type")

                    instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # ============ 计算类别先验损失 ============
                    if args.with_prior_preservation and "class_pixel_values" in batch:
                        # 处理类别图片
                        class_latents = vae.encode(
                            batch["class_pixel_values"].to(dtype=weight_dtype)
                        ).latent_dist.sample()
                        class_latents = class_latents * 0.18215

                        class_noise = torch.randn_like(class_latents)
                        class_timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (class_latents.shape[0],),
                            device=class_latents.device,
                        ).long()

                        class_noisy_latents = noise_scheduler.add_noise(
                            class_latents, class_noise, class_timesteps
                        )

                        class_encoder_hidden_states = text_encoder(batch["class_input_ids"])[0]

                        class_model_pred = unet(
                            class_noisy_latents, class_timesteps, class_encoder_hidden_states
                        ).sample

                        if noise_scheduler.config.prediction_type == "epsilon":
                            class_target = class_noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            class_target = noise_scheduler.get_velocity(
                                class_latents, class_noise, class_timesteps
                            )

                        prior_loss = F.mse_loss(
                            class_model_pred.float(), class_target.float(), reduction="mean"
                        )

                        # ============ 组合损失 ============
                        loss = instance_loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = instance_loss
                        prior_loss = torch.tensor(0.0)

                    # ============ NaN 检测 ============
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"NaN/Inf loss at step {global_step}!")
                        raise ValueError("NaN/Inf loss detected!")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = itertools.chain(
                            lora_layers.parameters(),
                            text_encoder.get_input_embeddings().parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # ============ 冻结非placeholder token ============
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                # ============ 显存清理 ============
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                progress_bar.update(1)
                global_step += 1

                # ============ 更新日志：显示实例损失和先验损失 ============
                logs = {
                    "loss": loss.detach().item(),
                    "instance_loss": instance_loss.detach().item(),
                    "prior_loss": prior_loss.detach().item() if args.with_prior_preservation else 0.0,
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)

                # ============ 详细日志 ============
                if global_step % 100 == 0:
                    logger.info(
                        f"Step {global_step}/{args.max_train_steps} | "
                        f"Epoch {epoch + 1} | "
                        f"Total Loss: {loss.item():.6f} | "
                        f"Instance Loss: {instance_loss.item():.6f} | "
                        f"Prior Loss: {prior_loss.item():.6f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.6e}"
                    )

                # ============ 保存 checkpoint ============
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_dir = Path(args.output_dir) / f"checkpoint_{global_step}"
                    save_dir.mkdir(exist_ok=True)

                    # 保存 LoRA 权重
                    unet_lora_state_dict = {}
                    for name, processor in accelerator.unwrap_model(unet).attn_processors.items():
                        if isinstance(processor, (LoRAAttnProcessor, LoRAAttnProcessor2_0)):
                            unet_lora_state_dict[name] = {
                                "to_k_lora.down.weight": processor.to_k_lora.down.weight,
                                "to_k_lora.up.weight": processor.to_k_lora.up.weight,
                                "to_q_lora.down.weight": processor.to_q_lora.down.weight,
                                "to_q_lora.up.weight": processor.to_q_lora.up.weight,
                                "to_v_lora.down.weight": processor.to_v_lora.down.weight,
                                "to_v_lora.up.weight": processor.to_v_lora.up.weight,
                                "to_out_lora.down.weight": processor.to_out_lora.down.weight,
                                "to_out_lora.up.weight": processor.to_out_lora.up.weight,
                            }
                    torch.save(unet_lora_state_dict, save_dir / "unet_lora.pt")

                    # 保存 token embeddings
                    embeds_dict = {}
                    for subject_id, token_id in placeholder_token_ids.items():
                        embeds_dict[placeholder_tokens[subject_id]] = (
                            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[token_id]
                            .detach()
                            .cpu()
                        )
                    torch.save(embeds_dict, save_dir / "embeddings.pt")

                    # 保存 metadata
                    metadata = {
                        "subject_ids": list(placeholder_tokens.keys()),
                        "placeholder_tokens": placeholder_tokens,
                        "subject_random_ids": train_dataset.subject_random_ids,
                        "global_step": global_step,
                        "prior_loss_weight": args.prior_loss_weight,
                        "with_prior_preservation": args.with_prior_preservation,
                    }
                    with open(save_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"✓ Checkpoint saved to {save_dir}")

                if global_step >= args.max_train_steps:
                    logger.info(f"Reached max_train_steps ({args.max_train_steps}), stopping...")
                    break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA OOM at step {global_step}!")
                    logger.error(f"Batch size: {batch['pixel_values'].shape[0]}")
                    logger.error(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
                    logger.error(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB")

                    torch.cuda.empty_cache()
                    gc.collect()

                    logger.error("Training stopped due to OOM. Try reducing:")
                    logger.error("  - train_batch_size")
                    logger.error("  - lora_rank")
                    logger.error("  - resolution")
                    raise e
                else:
                    logger.error(f"Runtime error at step {global_step}: {str(e)}")
                    raise e

            except Exception as e:
                logger.error(f"Unexpected error at step {global_step}: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                raise e

        if global_step >= args.max_train_steps:
            break

    # ============ Save final model ============
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "final_model"
        final_dir.mkdir(exist_ok=True)

        # 保存 UNet LoRA
        unet_lora_state_dict = {}
        for name, processor in accelerator.unwrap_model(unet).attn_processors.items():
            if isinstance(processor, (LoRAAttnProcessor, LoRAAttnProcessor2_0)):
                unet_lora_state_dict[name] = {
                    "to_k_lora.down.weight": processor.to_k_lora.down.weight,
                    "to_k_lora.up.weight": processor.to_k_lora.up.weight,
                    "to_q_lora.down.weight": processor.to_q_lora.down.weight,
                    "to_q_lora.up.weight": processor.to_q_lora.up.weight,
                    "to_v_lora.down.weight": processor.to_v_lora.down.weight,
                    "to_v_lora.up.weight": processor.to_v_lora.up.weight,
                    "to_out_lora.down.weight": processor.to_out_lora.down.weight,
                    "to_out_lora.up.weight": processor.to_out_lora.up.weight,
                }
        torch.save(unet_lora_state_dict, final_dir / "unet_lora.pt")

        # 保存 embeddings
        embeds_dict = {}
        for subject_id, token_id in placeholder_token_ids.items():
            embeds_dict[placeholder_tokens[subject_id]] = (
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[token_id]
                .detach()
                .cpu()
            )
        torch.save(embeds_dict, final_dir / "embeddings.pt")

        # 保存 metadata
        metadata = {
            "subject_ids": list(placeholder_tokens.keys()),
            "placeholder_tokens": placeholder_tokens,
            "subject_random_ids": train_dataset.subject_random_ids,
            "prior_loss_weight": args.prior_loss_weight,
            "with_prior_preservation": args.with_prior_preservation,
        }
        with open(final_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Training completed! Final model saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
