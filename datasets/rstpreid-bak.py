import os.path as op
from typing import List

from utils.iotools import read_json
from utils.iotools import read_json_from_LLM_captions
from .bases import BaseDataset

from .llm_augment import LLMTextAugmenter
import numpy as np

# 修改 rstpreid.py - 正确整合生成图像

import os
import os.path as op
from typing import List
import numpy as np
from utils.iotools import read_json, read_json_from_LLM_captions
from .bases import BaseDataset


class RSTPReid(BaseDataset):
    """
    RSTPReid with support for generated images
    """
    dataset_dir = 'RSTPReid'

    def __init__(self, root='', verbose=True, use_llm_augment=False,
                 llm_augmented_path='./captions/pedestrian_captions_results.json',
                 text_length=77,
                 use_generated=False,
                 generated_img_dir='E:/output_generate',
                 generated_captions_path='./captions/pedestrian_captions_results.json'):

        super(RSTPReid, self).__init__()
        self.text_length = text_length
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'data_captions.json')
        self.llm_augmented_path = llm_augmented_path
        self.use_llm_augment = use_llm_augment

        self.use_generated = use_generated
        self.generated_img_dir = generated_img_dir
        self.generated_captions_path = generated_captions_path
        self.generated_captions = {}

        self._check_before_run()

        self.llm_generated_captions = read_json_from_LLM_captions(
            self.llm_augmented_path) if use_llm_augment else None

        if self.use_generated:
            self.generated_captions = self._load_generated_captions()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(
            self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RSTPReid Images and Captions are loaded")
            if self.use_generated:
                self.logger.info(f"=> Generated images: {len(self.generated_captions)} captions loaded")
            self.show_dataset_info()

    def _load_generated_captions(self):
        """加载生成图像的标注"""
        try:
            generated_captions = read_json_from_LLM_captions(
                self.generated_captions_path)
            self.logger.info(f"Loaded {len(generated_captions)} generated image captions")
            return generated_captions
        except Exception as e:
            self.logger.warning(f"Failed to load generated captions: {e}")
            return {}

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()

        if training:
            dataset = []
            image_id = 0

            # ========== 第一部分：原始训练数据 ==========
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                img_filename = op.basename(anno['img_path'])

                original_captions = anno['captions']
                all_captions = original_captions.copy()

                if self.use_llm_augment and self.llm_generated_captions:
                    llm_captions = self.llm_generated_captions.get(img_filename, [])
                    if llm_captions:
                        all_captions.extend(llm_captions)

                for caption in all_captions:
                    sim = np.array([1 - 0.15] * self.text_length)
                    dataset.append((pid, image_id, img_path, caption, sim))
                image_id += 1

            # ========== 第二部分：生成图像 - 每张图一个image_id ==========
            if self.use_generated and self.generated_captions:
                for pid in sorted(pid_container):
                    generated_pid_dir = op.join(self.generated_img_dir, f"{pid:04d}_selected")

                    if not op.exists(generated_pid_dir):
                        continue

                    generated_images = sorted([f for f in os.listdir(generated_pid_dir)
                                               if f.endswith(('.png', '.jpg', '.jpeg'))])

                    for img_file in generated_images:
                        img_path = op.join(generated_pid_dir, img_file)

                        # 查找对应的标注
                        json_key = op.join(f"{pid:04d}_selected", img_file).replace('\\', '/')
                        json_key_backslash = f"{pid:04d}_selected\\{img_file}"

                        captions = self.generated_captions.get(json_key, None)
                        if captions is None:
                            captions = self.generated_captions.get(json_key_backslash, None)

                        if captions is None:
                            continue

                        # 关键改动：每张生成图只分配一个 image_id，但有多条caption
                        for caption in captions:
                            sim = np.array([1 - 0.15] * self.text_length)
                            dataset.append((pid, image_id, img_path, caption, sim))

                        # 每张生成图递增一次 image_id
                        image_id += 1

                self.logger.info(f"Added {image_id - len(annos)} generated images to training set")

            # 验证pid连续性
            for idx, pid in enumerate(pid_container):
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"

            return dataset, pid_container

        else:
            # 测试/验证集不使用生成图像
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []

            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions']
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        """检查必要文件是否存在"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))

        if self.use_llm_augment and not op.exists(self.llm_augmented_path):
            raise RuntimeError("预生成LLM文本文件不存在: '{}'".format(self.llm_augmented_path))

        if self.use_generated:
            if not op.exists(self.generated_img_dir):
                raise RuntimeError("生成图像目录不存在: '{}'".format(self.generated_img_dir))
            if not op.exists(self.generated_captions_path):
                raise RuntimeError("生成图像标注文件不存在: '{}'".format(self.generated_captions_path))


# # 这是使用了高质量数据集captions的代码。
# class RSTPReid(BaseDataset):
#     """
#     RSTPReid
#
#     Reference:
#     DSSL: Deep Surroundings-person Separation Learning for Text-based Person Retrieval MM 21
#
#     URL: http://arxiv.org/abs/2109.05534
#
#     Dataset statistics:
#     # identities: 4101
#     """
#     dataset_dir = 'RSTPReid'
#
#     def __init__(self, root='', verbose=True, use_llm_augment=False, llm_augmented_path='./captions/pedestrian_captions_results.json'
#                  , text_length=77): #新增：预生成JSON路径参数
#         super(RSTPReid, self).__init__()
#         self.text_length = text_length
#         self.dataset_dir = op.join(root, self.dataset_dir)
#         self.img_dir = op.join(self.dataset_dir, 'imgs/')
#
#         self.anno_path = op.join(self.dataset_dir, 'data_captions.json')
#         # 预生成LLM captions的默认路径（如果未指定）
#         self.llm_augmented_path = llm_augmented_path
#         self.use_llm_augment = use_llm_augment
#         self._check_before_run()  # 检查预生成文件是否存在
#
#         # 加载预生成的LLM captions（键：图像文件名，值：生成的文本列表）
#         self.llm_generated_captions = read_json_from_LLM_captions(self.llm_augmented_path) if use_llm_augment else None
#
#         self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
#
#         self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
#         self.test, self.test_id_container = self._process_anno(self.test_annos)
#         self.val, self.val_id_container = self._process_anno(self.val_annos)
#
#         if verbose:
#             self.logger.info("=> RSTPReid Images and Captions are loaded")
#             self.show_dataset_info()
#
#     def _split_anno(self, anno_path: str):
#         train_annos, test_annos, val_annos = [], [], []
#         annos = read_json(anno_path)
#         for anno in annos:
#             if anno['split'] == 'train':
#                 train_annos.append(anno)
#             elif anno['split'] == 'test':
#                 test_annos.append(anno)
#             else:
#                 val_annos.append(anno)
#         return train_annos, test_annos, val_annos
#
#     def _process_anno(self, annos: List[dict], training=False):
#         pid_container = set()
#
#         if training:
#             dataset = []
#             image_id = 0
#             for anno in annos:
#                 pid = int(anno['id'])
#                 pid_container.add(pid)
#                 img_path = op.join(self.img_dir, anno['img_path'])
#                 # 提取图像文件名（用于匹配预生成JSON的键，如"0000_c14_0031.jpg"）
#                 img_filename = op.basename(anno['img_path'])
#
#                 # 原始文本
#                 original_captions = anno['captions']  # 原始caption列表
#                 all_captions = original_captions.copy()  # 先保留原始文本
#
#                 # 合并预生成的LLM文本（如果启用）
#                 if self.use_llm_augment and self.llm_generated_captions:
#                     # 从预生成JSON中获取当前图像的生成文本
#                     llm_captions = self.llm_generated_captions.get(img_filename, [])
#                     if llm_captions:
#                         all_captions.extend(llm_captions)  # 合并原始+生成文本
#                     else:
#                         self.logger.warning(f"未找到图像 {img_filename} 的预生成LLM文本，已跳过")
#
#                 # 将所有文本（原始+生成）加入数据集
#                 for caption in all_captions:
#                     # 初始化sim（与text_length长度一致，默认值可参考Synth_PEDES）
#                     sim = np.array([1 - 0.15] * self.text_length)
#                     #dataset.append((pid, image_id, img_path, caption))
#                     dataset.append((pid, image_id, img_path, caption, sim))
#                 image_id += 1
#
#             # 验证pid连续性（保持原始逻辑）
#             for idx, pid in enumerate(pid_container):
#                 assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
#             return dataset, pid_container
#         else:
#             # 测试/验证集不使用生成文本（保持原始逻辑）
#             dataset = {}
#             img_paths = []
#             captions = []
#             image_pids = []
#             caption_pids = []
#             for anno in annos:
#                 pid = int(anno['id'])
#                 pid_container.add(pid)
#                 img_path = op.join(self.img_dir, anno['img_path'])
#                 img_paths.append(img_path)
#                 image_pids.append(pid)
#                 caption_list = anno['captions']
#                 for caption in caption_list:
#                     captions.append(caption)
#                     caption_pids.append(pid)
#             dataset = {
#                 "image_pids": image_pids,
#                 "img_paths": img_paths,
#                 "caption_pids": caption_pids,
#                 "captions": captions
#             }
#             return dataset, pid_container
#
#
#
#         # if training and self.use_llm_augment:
#         #     # 初始化LLM扩充器
#         #     augmenter = LLMTextAugmenter(
#         #         save_path=op.join(self.dataset_dir, "augmented_captions.json")
#         #     )
#         #
#         # if training:
#         #     dataset = []
#         #     image_id = 0
#         #     for anno in annos:
#         #         pid = int(anno['id'])
#         #         pid_container.add(pid)
#         #         img_path = op.join(self.img_dir, anno['img_path'])
#         #
#         #         original_captions = anno['captions']  # 原始描述列表
#         #         all_captions = []
#         #
#         #         if self.use_llm_augment:
#         #             # 使用带风格的生成方法，基于特征生成
#         #             # 合并两条原始描述 (用分隔符连接)
#         #             # combined_caption = " ; ".join((original_captions))
#         #             combined_caption = [] #合并两条原始描述 (用list连接)
#         #             combined_caption.extend(original_captions)
#         #             augmented = augmenter.generate(
#         #                 combined_caption,
#         #                 num_aug=self.llm_num_aug,
#         #                 styles=self.llm_styles
#         #             )
#         #             all_captions.extend(original_captions) #保留原始两条描述
#         #             all_captions.extend(augmented) #加入基于合并信息生成的新描述
#         #         else:
#         #             all_captions.extend(original_captions)
#         #
#         #         for caption in all_captions:
#         #             dataset.append((pid, image_id, img_path, caption))
#         #         image_id += 1
#         #
#         #     for idx, pid in enumerate(pid_container):
#         #         assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
#         #     return dataset, pid_container
#         # else:
#         #     dataset = {}
#         #     img_paths = []
#         #     captions = []
#         #     image_pids = []
#         #     caption_pids = []
#         #     for anno in annos:
#         #         pid = int(anno['id'])
#         #         pid_container.add(pid)
#         #         img_path = op.join(self.img_dir, anno['img_path'])
#         #         img_paths.append(img_path)
#         #         image_pids.append(pid)
#         #         caption_list = anno['captions']
#         #         for caption in caption_list:
#         #             captions.append(caption)
#         #             caption_pids.append(pid)
#         #     dataset = {
#         #         "image_pids": image_pids,
#         #         "img_paths": img_paths,
#         #         "caption_pids": caption_pids,
#         #         "captions": captions
#         #     }
#         #     return dataset, pid_container
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not op.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not op.exists(self.img_dir):
#             raise RuntimeError("'{}' is not available".format(self.img_dir))
#         if not op.exists(self.anno_path):
#             raise RuntimeError("'{}' is not available".format(self.anno_path))
#         # 如果启用LLM增强，检查预生成文件是否存在
#         if self.use_llm_augment and not op.exists(self.llm_augmented_path):
#             raise RuntimeError("预生成LLM文本文件不存在: '{}'".format(self.llm_augmented_path))
#



# 这是使用了垃圾数据集captions的代码。

# 这是使用低质量数据集（但是第一轮性能高)
# class RSTPReid(BaseDataset):
#     """
#     RSTPReid
#
#     Reference:
#     DSSL: Deep Surroundings-person Separation Learning for Text-based Person Retrieval MM 21
#
#     URL: http://arxiv.org/abs/2109.05534
#
#     Dataset statistics:
#     # identities: 4101
#     """
#     dataset_dir = 'RSTPReid'
#
#     #def __init__(self, root='', verbose=True, use_llm_augment=True, llm_augmented_path='./captions/pedestrian_captions_results.json'): #新增：预生成JSON路径参数
#     def __init__(self, root='', verbose=True, use_llm_augment=True, llm_num_aug=1, llm_styles=["concise", "formal"]):  # 新增风格参数
#         super(RSTPReid, self).__init__()
#         self.dataset_dir = op.join(root, self.dataset_dir)
#         self.img_dir = op.join(self.dataset_dir, 'imgs/')
#
#         self.anno_path = op.join(self.dataset_dir, 'data_captions.json')
#         self._check_before_run()
#         # 预生成LLM captions的默认路径（如果未指定）
#         # self.llm_augmented_path = llm_augmented_path
#         # self.use_llm_augment = use_llm_augment
#         # self._check_before_run()  # 检查预生成文件是否存在
#
#         # 加载预生成的LLM captions（键：图像文件名，值：生成的文本列表）
#         #self.llm_generated_captions = read_json_from_LLM_captions(self.llm_augmented_path) if use_llm_augment else None
#
#         self.use_llm_augment = use_llm_augment
#         self.llm_num_aug = llm_num_aug
#         self.llm_styles = llm_styles  # 存储风格列表
#
#         self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
#
#         self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
#         self.test, self.test_id_container = self._process_anno(self.test_annos)
#         self.val, self.val_id_container = self._process_anno(self.val_annos)
#
#         if verbose:
#             self.logger.info("=> RSTPReid Images and Captions are loaded")
#             self.show_dataset_info()
#
#     def _split_anno(self, anno_path: str):
#         train_annos, test_annos, val_annos = [], [], []
#         annos = read_json(anno_path)
#         for anno in annos:
#             if anno['split'] == 'train':
#                 train_annos.append(anno)
#             elif anno['split'] == 'test':
#                 test_annos.append(anno)
#             else:
#                 val_annos.append(anno)
#         return train_annos, test_annos, val_annos
#
#     def _process_anno(self, annos: List[dict], training=False):
#         pid_container = set()
#
#         # if training:
#         #     dataset = []
#         #     image_id = 0
#         #     for anno in annos:
#         #         pid = int(anno['id'])
#         #         pid_container.add(pid)
#         #         img_path = op.join(self.img_dir, anno['img_path'])
#         #         # 提取图像文件名（用于匹配预生成JSON的键，如"0000_c14_0031.jpg"）
#         #         img_filename = op.basename(anno['img_path'])
#         #
#         #         # 原始文本
#         #         original_captions = anno['captions']  # 原始caption列表
#         #         all_captions = original_captions.copy()  # 先保留原始文本
#         #
#         #         # 合并预生成的LLM文本（如果启用）
#         #         if self.use_llm_augment and self.llm_generated_captions:
#         #             # 从预生成JSON中获取当前图像的生成文本
#         #             llm_captions = self.llm_generated_captions.get(img_filename, [])
#         #             if llm_captions:
#         #                 all_captions.extend(llm_captions)  # 合并原始+生成文本
#         #             else:
#         #                 self.logger.warning(f"未找到图像 {img_filename} 的预生成LLM文本，已跳过")
#         #
#         #         # 将所有文本（原始+生成）加入数据集
#         #         for caption in all_captions:
#         #             dataset.append((pid, image_id, img_path, caption))
#         #         image_id += 1
#         #
#         #     # 验证pid连续性（保持原始逻辑）
#         #     for idx, pid in enumerate(pid_container):
#         #         assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
#         #     return dataset, pid_container
#         # else:
#         #     # 测试/验证集不使用生成文本（保持原始逻辑）
#         #     dataset = {}
#         #     img_paths = []
#         #     captions = []
#         #     image_pids = []
#         #     caption_pids = []
#         #     for anno in annos:
#         #         pid = int(anno['id'])
#         #         pid_container.add(pid)
#         #         img_path = op.join(self.img_dir, anno['img_path'])
#         #         img_paths.append(img_path)
#         #         image_pids.append(pid)
#         #         caption_list = anno['captions']
#         #         for caption in caption_list:
#         #             captions.append(caption)
#         #             caption_pids.append(pid)
#         #     dataset = {
#         #         "image_pids": image_pids,
#         #         "img_paths": img_paths,
#         #         "caption_pids": caption_pids,
#         #         "captions": captions
#         #     }
#         #     return dataset, pid_container
#         #
#
#
#         if training and self.use_llm_augment:
#             # 初始化LLM扩充器
#             augmenter = LLMTextAugmenter(
#                 save_path=op.join(self.dataset_dir, "augmented_captions.json")
#             )
#
#         if training:
#             dataset = []
#             image_id = 0
#             for anno in annos:
#                 pid = int(anno['id'])
#                 pid_container.add(pid)
#                 img_path = op.join(self.img_dir, anno['img_path'])
#
#                 original_captions = anno['captions']  # 原始描述列表
#                 all_captions = []
#
#                 if self.use_llm_augment:
#                     # 使用带风格的生成方法，基于特征生成
#                     # 合并两条原始描述 (用分隔符连接)
#                     # combined_caption = " ; ".join((original_captions))
#                     combined_caption = [] #合并两条原始描述 (用list连接)
#                     combined_caption.extend(original_captions)
#                     augmented = augmenter.generate(
#                         combined_caption,
#                         num_aug=self.llm_num_aug,
#                         styles=self.llm_styles
#                     )
#                     all_captions.extend(original_captions) #保留原始两条描述
#                     all_captions.extend(augmented) #加入基于合并信息生成的新描述
#                 else:
#                     all_captions.extend(original_captions)
#
#                 for caption in all_captions:
#                     dataset.append((pid, image_id, img_path, caption))
#                 image_id += 1
#
#             for idx, pid in enumerate(pid_container):
#                 assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
#             return dataset, pid_container
#         else:
#             dataset = {}
#             img_paths = []
#             captions = []
#             image_pids = []
#             caption_pids = []
#             for anno in annos:
#                 pid = int(anno['id'])
#                 pid_container.add(pid)
#                 img_path = op.join(self.img_dir, anno['img_path'])
#                 img_paths.append(img_path)
#                 image_pids.append(pid)
#                 caption_list = anno['captions']
#                 for caption in caption_list:
#                     captions.append(caption)
#                     caption_pids.append(pid)
#             dataset = {
#                 "image_pids": image_pids,
#                 "img_paths": img_paths,
#                 "caption_pids": caption_pids,
#                 "captions": captions
#             }
#             return dataset, pid_container
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not op.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not op.exists(self.img_dir):
#             raise RuntimeError("'{}' is not available".format(self.img_dir))
#         if not op.exists(self.anno_path):
#             raise RuntimeError("'{}' is not available".format(self.anno_path))
#         # 如果启用LLM增强，检查预生成文件是否存在

