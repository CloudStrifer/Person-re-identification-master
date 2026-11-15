# datasets/llm_augment.py
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# class LLMTextAugmenter:
#     def __init__(self,
#                  #model_path="../Qwen2.5-7B",  # 本地Qwen2.5-7B模型路径
#                  #model_path="E:\PyCharm_Code\Qwen2.5-7B",
#                  model_path="E:\PyCharm_Code\Qwen2.5-0.5B-Instruct",
#                  save_path="./augmented_captions/rstpreid.json",
#                  device="cuda" if torch.cuda.is_available() else "cpu"):
#         self.save_path = save_path
#         self.device = device
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         self.augmented_data = self._load_existing()
#
#         # 加载Qwen2.5-0.5B模型和tokenizer（关键适配）
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             trust_remote_code=True,
#             padding_side="left"  # Qwen默认左padding，需保持一致
#         )
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             trust_remote_code=True,
#             torch_dtype=torch.float16,  # 16位精度节省显存
#             device_map="auto"  # 自动分配设备（支持多卡）
#         ).eval()
#
#         # Qwen需要手动设置pad_token（默认无）
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#
#     def _load_existing(self):
#         if os.path.exists(self.save_path):
#             with open(self.save_path, "r") as f:
#                 return json.load(f)
#         return {}
#
#     def _save_augmented(self):
#         with open(self.save_path, "w") as f:
#             json.dump(self.augmented_data, f, indent=2)
#
#     def generate(self, original_caption, num_aug=1):
#         """调用Qwen2.5-7B生成扩充描述"""
#         if original_caption in self.augmented_data:
#             return self.augmented_data[original_caption]
#
#         # 构造提示词（适配Qwen的对话格式）
#         # prompt = f"""请基于以下行人描述，生成{num_aug}个语义相似但表述不同的描述，保持细节一致（如服装颜色、性别、动作等）。每个描述单独成行，不要添加序号或额外说明
#         prompt = f"""Please generate {num_aug} semantically similar but differently worded descriptions based on the following pedestrian descriptions, while keeping the details consistent (such as clothing color, gender, actions, etc.). Each description should be written in a separate line without adding a serial number or additional explanation
#         Rules:
#         - Each description must be on a separate line.
#         - No serial numbers, no explanations, no self-introduction.
#         - Only output the generated descriptions.
#
#         原始描述：{original_caption}
#         生成结果："""
#
#         # Qwen生成逻辑（使用chat模板格式）
#         messages = [
#             #{"role": "system", "content": "You are a text augmentation assistant and need to strictly follow the requirements to generate pedestrian descriptions"},
#             #{"role": "system", "content": "你是一个文本扩充助手，需严格按照要求生成行人描述"},
#             {"role": "system", "content": "你是文本扩充助手，仅负责生成行人描述。严格遵循以下规则：1. 只输出生成的描述，不添加任何自我介绍、解释或额外说明；2. 不回复与任务无关的内容；3. 每个描述单独成行，无序号。"},
#             {"role": "user", "content": prompt}
#         ]
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True  # 自动添加生成前缀
#         )
#
#         with torch.no_grad():
#             inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=200,  # 控制生成长度（每个描述约30词）
#                 temperature=0.5,  # 随机性（0.5-1.0为宜）
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 repetition_penalty=1.2,  # 抑制重复生成
#                 top_p=0.9   # 限制采样范围，减少低概率错误tokens的生成
#             )
#
#         # 解析生成结果（去除输入部分）
#         response = self.tokenizer.decode(
#             outputs[0][len(inputs["input_ids"][0]):],
#             skip_special_tokens=True
#         ).strip()
#
#         # 按行分割，过滤空行和无效内容
#         augmented = [line.strip() for line in response.split("\n") if line.strip()]
#
#         # 确保生成数量达标（不足则用原始描述补充）
#         if len(augmented) < num_aug:
#             augmented += [original_caption] * (num_aug - len(augmented))
#
#         self.augmented_data[original_caption] = augmented[:num_aug]
#         self._save_augmented()
#         return augmented[:num_aug]
#
#     def augment_dataset(self, dataset_captions, num_aug=3):
#         """批量扩充数据集描述"""
#         augmented_captions = []
#         for caption in tqdm(dataset_captions, desc="Augmenting with Qwen2.5-7B"):
#             augmented = self.generate(caption, num_aug)
#             augmented_captions.append(augmented)
#         return augmented_captions



class LLMTextAugmenter:
    def __init__(self,
                 model_path="./Qwen2.5-0.5B-Instruct/",
                 save_path="./augmented_captions/rstpreid.json",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.save_path = save_path
        self.device = device
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.data = self._load_existing()  # 现在存储特征和不同风格的扩充结果

        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        # 支持的生成风格
        self.styles = {
            # "concise": "简洁风格，用最少的词语描述关键特征",
            # "detailed": "详细风格，丰富细节描述",
            # "colloquial": "口语化风格，像日常对话一样自然",
            # "formal": "正式风格，使用规范书面语"

            "concise": "Concise style, describing key features with minimal words",
            # "detailed": "Detailed style, rich description of detailed",
            # "colloquial": "Colloquial style, as natural as daily colloquial",
            "formal": "Formal style, using standardized written documents"
        }

    def _load_existing(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                return json.load(f)
        return {}

    def _save_data(self):
        with open(self.save_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def extract_features(self, caption):
        # """从描述中抽取关键特征"""
        # if caption in self.data and "features" in self.data[caption]:
        #     return self.data[caption]["features"]

        """从一个或多个描述中抽取关键特征（支持单字符串或列表）"""
        # 统一将输入转换为字符串，方便处理和缓存
        if isinstance(caption, list):
            # 将列表转换为字符串作为唯一标识
            caption_key = " || ".join(caption)
        else:
            caption_key = caption

        # 检查缓存
        if caption_key in self.data and "features" in self.data[caption_key]:
            return self.data[caption_key]["features"]

        # 构造特征抽取提示词
        # prompt = f"""请从以下行人描述中抽取关键特征，包括但不限于：性别、发型、服装（类型、颜色、款式）、配饰、动作等。
        # 只需返回提取的特征列表，每个特征用逗号分隔，不要添加额外说明。

        if isinstance(caption, list): #如果是多个描述的话，就这样做
            prompt = f"""Please extract key features from the following pedestrian descriptions, including but not limited to: gender, hairstyle, clothing (type, color, style), accessories, actions, etc.
            Just return the extracted feature list, separated by commas for each feature, without adding any additional explanations. The feature should be not exceed 8.
            
            describe 1：{caption[0]}
            describe 2：{caption[1]}
            feature："""
        else:
            prompt = f"""Please extract key features from the following pedestrian descriptions, including but not limited to: gender, hairstyle, clothing (type, color, style), accessories, actions, etc.
            Just return the extracted feature list, separated by commas for each feature, without adding any additional explanations. The feature should be not exceed 8.

            describe ：{caption}
            feature: """

        # messages = [
        #     {"role": "system", "content": "你是特征提取助手，仅提取行人描述中的关键特征，不添加额外内容"},
        #     {"role": "user", "content": prompt}
        # ]

        messages = [
            {"role": "system", "content": "You are a feature extraction assistant than only extracts key features from pedestrian descriptions without adding additional content"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.no_grad():
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,  # 低温度保证提取准确性
                do_sample=True,  # 确定性生成
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        ).strip()

        # 解析特征列表
        features = [f.strip() for f in response.split(",") if f.strip()]

        # 初始化该描述的数据结构
        # if caption not in self.data:
        #     self.data[caption] = {}
        # self.data[caption]["features"] = features
        # self._save_data()
        if caption_key not in self.data:
            self.data[caption_key] = {}
        self.data[caption_key]["features"] = features
        self._save_data()
        return features

    def generate_with_style(self, original_caption, caption_key, features, num_aug=1, style="concise"):
        """基于特征生成特定风格的描述"""
        # 检查缓存
        if (caption_key in self.data and
                "augmented" in self.data[caption_key] and
                style in self.data[caption_key]["augmented"] and
                len(self.data[caption_key]["augmented"][style]) >= num_aug):
            return self.data[caption_key]["augmented"][style][:num_aug]

        # 确保风格有效
        # style_desc = self.styles.get(style, self.styles["concise"])

        style_desc = self.styles.get(style)
        # 构造带风格的生成提示词
        # prompt = f"""基于以下关键特征，生成{num_aug}个{style_desc}的行人描述。
        # 关键特征：{', '.join(features)}
        # 要求：
        # - 保持特征准确性
        # - 每个描述单独成行
        # - 不添加序号和额外说明
        # - 体现指定的风格
        #
        # 生成结果："""

        if isinstance(original_caption, list):
            prompt = f"""Based on the following key features and origin caption, generate {num_aug} {style_desc} pedestrian descriptions.
            key features：{', '.join(features)}
            requirement：
            - here is origin caption 1. {original_caption[0]}
            - here is origin caption 2. {original_caption[1]}
            - Maintain feature accuracy
            - Each description should be written separately
            - Do not add serial numbers and additional explanations
            - Reflect the designated style
    
            Generate Results："""

        else:
            prompt = f"""Based on the following key features and origin caption, generate {num_aug} {style_desc} pedestrian descriptions.
            key features：{', '.join(features)}
            requirement：
             - here is origin caption {original_caption}
             - Maintain feature accuracy
             - Each description should be written separately
             - Do not add serial numbers and additional explanations
             - Reflect the designated style

            Generate Results："""

        # messages = [
        #     {"role": "system", "content": "你是文本生成助手，能根据特征生成不同风格的行人描述"},
        #     {"role": "user", "content": prompt}
        # ]

        messages = [
            {"role": "system", "content": "You are a text generation assistant that can generate pedestrian descriptions of different styles based on features"},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.no_grad():
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_p=0.9
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        ).strip()

        augmented = [line.strip() for line in response.split("\n") if line.strip()]

        # 确保生成数量
        if len(augmented) < num_aug:
            augmented += [caption_key] * (num_aug - len(augmented))

        # 保存结果
        if "augmented" not in self.data[caption_key]:
            self.data[caption_key]["augmented"] = {}
        self.data[caption_key]["augmented"][style] = augmented[:num_aug]
        self._save_data()
        return augmented[:num_aug]

    def generate(self, original_caption, num_aug=1, styles=None):
        """综合方法：先提取特征，再生成多种风格的描述"""
        if styles is None:
            styles = ["concise", "formal"]  # 默认风格

        # 提取特征
        features = self.extract_features(original_caption)

        # 统一处理缓存key（和extract_features保持一致）
        if isinstance(original_caption, list):
            caption_key = " || ".join(original_caption)
        else:
            caption_key = original_caption

        # 收集所有风格的生成结果
        all_augmented = []
        for style in styles:
            all_augmented.extend(self.generate_with_style(
                original_caption,
                caption_key,
                features,
                num_aug=num_aug,
                style=style
            ))

        return all_augmented

    def augment_dataset(self, dataset_captions, num_aug=3, styles=None):
        """批量扩充数据集描述"""
        augmented_captions = []
        for caption in tqdm(dataset_captions, desc="Augmenting with Qwen2.5-7B"):
            augmented = self.generate(caption, num_aug=num_aug, styles=styles)
            augmented_captions.append(augmented)
        return augmented_captions
