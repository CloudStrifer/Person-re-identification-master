import os
import base64
import json
from openai import OpenAI
from typing import List, Dict, Optional


class QwenVLImageCaptioner:
    def __init__(self, api_key=None):
        """初始化Qwen-VL图片描述生成器（兼容OpenAI格式）"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 行人特征列表
        self.target_features = [
            'gender', 'clothing type', 'age-related', 'physical characteristic',
            'color of the clothing or mountings', 'material characteristics of clothing or mountings',
            'mountings', 'Pedestrian actions', 'pedestrians of posture',
            'Background details', 'scene'
        ]

        # 参考模板
        self.reference_templates = [
            "A [gender] of average height strode confidently, wearing a [clothing type] that included [physical characteristic] boots and a [color of the clothing or mountings] skirt.",
            "An [age-related] [gender] approached the street corner, sporting a fashionable [clothing type] paired with black [mountings], which were made from leather.",
            "A [gender] in a [clothing type] with [physical characteristic] legs was walking through the park, wearing a [color of the clothing or mountings] dress.",
            "A tall [gender] crossed the road, adorned with a [clothing type] having a [color of the clothing or mountings] coat that matched their [physical characteristic] jeans.",
            "[gender] carried a backpack while walking, dressed in a [clothing type] that had [physical characteristic] jeans and a [color of the clothing or mountings] t-shirt.",
            "An energetic [gender] ran along the path, decked out in a [clothing type] that was both practical and stylish, with [physical characteristic] shorts exposed.",
            "A [gender] in a [clothing type] with [physical characteristic] hips was seen walking confidently, wearing a [color of the clothing or mountings] skirt.",
            "A [age-related] [gender] sat on the bench, wearing a [clothing type] that appeared to be made from a [material characteristics of clothing or mountings], complete with a [color of the clothing or mountings] hat.",
            "A [gender] with [physical characteristic] build was walking casually, wearing a [clothing type] with [color of the clothing or mountings] vest.",
            "A [gender] in a [clothing type] with [physical characteristic] hair was walking quickly toward the library, wearing a [color of the clothing or mountings] jacket.",
            "A [gender] wearing a [clothing type] with [physical characteristic] arms and a [color of the clothing or mountings] cardigan walked into the store.",
            "A middle-aged [gender] moved slowly, his [clothing type] showing signs of wear, particularly around the knees, where it was [color of the clothing or mountings].",
            "A slender [gender] walked with grace, carrying a [mountings] with [physical characteristic] sunglasses completing their look.",
            "[gender] was seen jogging through the park, wearing a light [clothing type] made from breathable [material characteristics of clothing or mountings].",
            "A [gender] in a [clothing type] with [physical characteristic] hands held a bag, sporting [color of the clothing or mountings] jeans and a striped [physical characteristic] shirt.",
            "A [gender] with [physical characteristic] skin tone was moving swiftly, wearing a [clothing type] with [color of the clothing or mountings] socks.",
            "A [gender] who looked like they were in their early [age-related] years was walking briskly, wearing a [clothing type] with [color of the clothing or mountings] boots.",
            "A robust [gender] was jogging, wearing a [clothing type] that included [physical characteristic] sweatpants and a [color of the clothing or mountings] jacket.",
            "A [gender] in a [clothing type] with [physical characteristic] arms was walking through the city streets, wearing a [color of the clothing or mountings] sweater.",
            "A [gender] wearing a [clothing type] with [physical characteristic] waist showed off a [color of the clothing or mountings] blouse.",
            "[gender] walked with a confident stride, wearing a [clothing type] that is made from [material characteristics of clothing or mountings]. The [physical characteristic] was visible under the dark [color of the clothing or mountings].",
            "A [gender] of [age-related] age was walking slowly, dressed in a [clothing type] featuring [color of the clothing or mountings] shorts and a [physical characteristic] hoodie.",
            "A [gender] in a [clothing type] with a [physical characteristic] headband was walking steadily, wearing a [color of the clothing or mountings] jacket.",
            "An elderly [gender] stood upright, holding a pair of [mountings], dressed in a [clothing type] with a pattern of [material characteristics of clothing or mountings].",
            "A [gender] wearing a [clothing type] with [physical characteristic] sleeves passed by, sporting a [color of the clothing or mountings] scarf tied around their neck.",
            "[gender] rushed through the crowded area, wearing a [clothing type] that showcased [physical characteristic] legs and a [color of the clothing or mountings] top.",
            "A [gender] with [physical characteristic] eyes was seen walking briskly, wearing a [clothing type] with [color of the clothing or mountings] pants.",
            "A [gender] with [physical characteristic] hair was seen walking briskly, wearing a [clothing type] with [color of the clothing or mountings] pants.",
            "A [gender] with a [physical characteristic] frame was standing in front of the museum, wearing a [clothing type] that included a [color of the clothing or mountings] vest over a blouse.",
            "A [gender] with a [physical characteristic] figure was moving steadily, wearing a [clothing type] with [color of the clothing or mountings] heels.",
            "[gender] passed by with an elegant demeanor, donning a sleek [clothing type] with [physical characteristic], its fabric a blend of [material characteristics of clothing or mountings].",
            "[gender] jogged briskly down the sidewalk, clad in a [clothing type] featuring a bold [color of the clothing or mountings] shirt over fitted trousers.",
            "[gender] walked with a relaxed pace, dressed in a [clothing type] with [physical characteristic] pants, and a [color of the clothing or mountings] sweater.",
            "An athletic [gender] was running with determination, wearing a [clothing type] that featured [physical characteristic] leggings and a [color of the clothing or mountings] hoodie.",
            "A [gender] with [physical characteristic] face was walking slowly, wearing a [clothing type] with [color of the clothing or mountings] gloves.",
            "A fit [gender] jogged through the park, clad in a [clothing type] that was primarily [color of the clothing or mountings] with a white [physical characteristic] sleeve.",
            "A [gender] in a [clothing type] with [physical characteristic] legs held a purse, wearing a [color of the clothing or mountings] coat.",
            "A [gender] who seemed quite [age-related] was seated on the bus stop bench, clad in a [clothing type] featuring [physical characteristic] sneakers and a [color of the clothing or mountings] cap.",
            "A [gender] with [physical characteristic] legs was jogging with agility, wearing a [clothing type] that included [color of the clothing or mountings] tights.",
            "A [gender] of advanced [age-related] age was strolling at a slower pace, dressed in a [clothing type] featuring [color of the clothing or mountings] pants and a [physical characteristic] jacket.",
            "A [gender] who seemed to be in [age-related] years was sitting on the steps, wearing a [clothing type] that included [color of the clothing or mountings] sandals.",
            "A young [gender] strolled past, her [clothing type] reflecting a casual and vibrant style, featuring a [color of the clothing or mountings] jacket over a [physical characteristic] dress.",
            "A [gender] with a [physical characteristic] face was walking purposefully, wearing a [clothing type] with [color of the clothing or mountings] shoes.",
            "A [age-related] [gender] leaned against the wall, wearing a simple yet well-fitted [clothing type] with [color of the clothing or mountings] hoodie.",
            "[gender] wore a bright red [clothing type] with [physical characteristic], walking towards the coffee shop on [scene].",
            "[gender] strolled leisurely, wearing a loose [clothing type] with [physical characteristic] pants, and a [color of the clothing or mountings] shirt.",
            "A [gender] in a [clothing type] with [physical characteristic] shoulders sported a [color of the clothing or mountings] long-sleeved top and matching [mountings].",
            "A [gender] in a [clothing type] with [physical characteristic] glasses walked calmly through the busy street, wearing a [color of the clothing or mountings] blazer.",
            "[gender] jogged with ease, wearing tight [clothing type] with a [color of the clothing or mountings] track jacket.",
            "A [gender] in a [clothing type] with [physical characteristic] boots walked past the fountain, wearing a [color of the clothing or mountings] trench coat."
        ]

    def image_to_base64(self, image_path):
        """将本地图片转换为Base64编码"""
        try:
            image_size = os.path.getsize(image_path)
            if image_size > 5 * 1024 * 1024:
                print(
                    f"⚠️ 图片 {os.path.basename(image_path)} 超过5MB（{image_size / 1024 / 1024:.2f}MB），可能导致API报错")

            with open(image_path, "rb") as image_file:
                base64_str = base64.b64encode(image_file.read()).decode("utf-8")
                image_ext = os.path.splitext(image_path)[1].lower()
                mime_type = f"image/{image_ext[1:]}" if image_ext in ['.jpg', '.jpeg', '.png', '.bmp'] else "image/jpeg"
                return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            print(f"❌ 图片编码失败（{os.path.basename(image_path)}）：{str(e)}")
            return None

    def generate_captions(self, image_base64, num_captions=2):
        """生成指定数量的图片描述"""
        if not image_base64:
            return [f"❌ 无效的图片数据"] * num_captions

        system_prompt = {
            "type": "text",
            "text": f"""
            你的任务是：基于提供的图片，生成{num_captions}条不同的行人自然语言描述，每条单独成句，必须满足：
            1. 参考模板风格{self.reference_templates}，保持句式结构相似但内容不同。尽可能的用不同句式或风格
            2. 包含图片中存在的以下特征（不存在的可忽略）：
               {chr(10).join([f"- {f}" for f in self.target_features])}
            3. 用完整英文句子，不使用任何占位符，自然流畅
            4. 每条长度1-2句话，不超过50词
            5. 确保{num_captions}条描述从不同角度描述，避免重复
            6. 若图片中无行人，所有描述都返回"图片中未检测到行人"
            """
        }

        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": image_base64, "detail": "high"}
            },
            {
                "type": "text",
                "text": f"请生成{num_captions}条不同的行人描述，严格按要求执行，每条描述单独一行。"
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[
                    {"role": "system", "content": [system_prompt]},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.8,
                top_p=0.85,
                max_tokens=400,
            )

            caption = completion.choices[0].message.content.strip()
            captions = [line.strip() for line in caption.split('\n') if line.strip()]

            if len(captions) < num_captions:
                captions += [captions[-1]] * (num_captions - len(captions))
            return captions[:num_captions]

        except Exception as e:
            error_msg = str(e)
            error = "❌ API调用失败"
            if "InvalidApiKey" in error_msg:
                error = "❌ API密钥无效"
            elif "ModelNotAuthorized" in error_msg:
                error = "❌ 无该模型权限"
            elif "413 Request Entity Too Large" in error_msg:
                error = "❌ 图片过大"
            return [f"{error}：{error_msg[:50]}..."] * num_captions

    def load_existing_results(self, output_file):
        """加载已生成的结果，用于断点续传"""
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ 检测到{output_file}格式错误，将重新生成该文件")
                return {}
        return {}

    def process_folder(self, folder_path, num_captions=2, output_file="pedestrian_captions_results.json"):
        """处理文件夹中的图片，支持边生成边写入和断点续传"""
        if not os.path.exists(folder_path):
            print(f"❌ 文件夹不存在：{folder_path}")
            return {}

        # 1. 加载已处理的结果，避免重复处理
        existing_results = self.load_existing_results(output_file)
        processed_files = set(existing_results.keys())
        print(f"✅ 已处理 {len(processed_files)} 张图片，将跳过这些文件")

        # 2. 获取所有待处理图片（排除已处理的）
        supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        all_image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]
        pending_files = [f for f in all_image_files if f not in processed_files]

        if not pending_files:
            print(f"✅ 所有图片已处理完成，无需继续")
            return existing_results

        print(f"✅ 发现 {len(pending_files)} 张未处理图片，开始生成描述...\n")

        # 3. 处理未完成的图片，边生成边写入
        results = existing_results  # 基于已有结果继续累加
        total = len(pending_files)

        for idx, filename in enumerate(pending_files, 1):
            image_path = os.path.join(folder_path, filename)
            print(f"🔄 处理第 {idx}/{total} 张（总进度：{idx + len(processed_files)}/{len(all_image_files)}）：{filename}")

            # 生成描述
            image_base64 = self.image_to_base64(image_path)
            captions = self.generate_captions(image_base64, num_captions)

            # 保存到结果字典
            results[filename] = captions
            print(f"✅ 生成描述：")
            for i, cap in enumerate(captions, 1):
                print(f"   {i}. {cap}")
            print()

            # 实时写入文件（每处理完一张就更新）
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"💾 已保存到 {output_file}\n")
            except Exception as e:
                print(f"⚠️ 写入文件失败：{str(e)}，将在下次处理时重试")

        print(f"📊 所有图片处理完成！共处理 {len(all_image_files)} 张，结果已保存到：{output_file}")
        return results


if __name__ == "__main__":
    # 配置参数
    DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxx"  # 替换为你的API密钥
    IMAGE_FOLDER_PATH = "../data/RSTPReid/imgs"  # 图片文件夹路径

    # 初始化生成器并处理图片（支持断点续传）
    captioner = QwenVLImageCaptioner(api_key=DASHSCOPE_API_KEY)
    captioner.process_folder(
        folder_path=IMAGE_FOLDER_PATH,
        num_captions=2,  # 每张图片生成2条描述
        output_file="pedestrian_captions_results.json"  # 结果文件路径
    )
