import requests
import json
import time
from typing import List, Dict, Optional


class Qwen25TemplateGenerator:
    def __init__(self, api_key: str, api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化Qwen2.5-7B模板生成器
        :param api_key: 阿里云API密钥
        :param api_base: API基础地址
        """
        self.api_key = api_key
        self.api_base = api_base
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_templates(self, attributes: List[str], num_templates: int = 50,
                           model: str = "qwen2.5-7b-instruct") -> List[str]:
        """
        生成行人特征描述模板
        :param attributes: 行人特征列表
        :param num_templates: 生成模板数量
        :param model: 使用的Qwen2.5模型，默认qwen2.5-7b-instruct
        :return: 生成的模板列表
        """
        # 构建提示词，针对Qwen2.5模型特点优化
        system_prompt = """
        你是一位行人特征描述专家，需要生成多样化的行人特征描述模板。
        请基于提供的行人特征，创建自然、流畅且多样化的描述模板。
        要求：
        1. 充分利用所有提供的特征，确保不同层级的特征都能被涵盖
        2. 模板应具有多样性，句式结构和特征组合方式应有所变化
        3. 必须使用[特征名称]作为占位符，如[gender]、[clothing type]等
        4. 描述应符合自然语言习惯，避免重复和生硬
        5. 考虑特征之间的逻辑关系，合理组织描述顺序
        6. 生成的模板数量为{num_templates}个，每个模板单独一行
        7. 不要加入额外的内容，例如无关的背景或者是不必要的语言
        8. 生成内容必须全部是英文的
        """.format(num_templates=num_templates)

        user_prompt = f"""
        以下是需要包含的行人特征列表，分为不同层级：
        {', '.join(attributes)}

        请生成{num_templates}个多样化的行人特征描述模板，每个模板应尽可能包含多个不同层级的特征，
        使用上述特征名称作为占位符，确保模板的丰富性和实用性。
        每个模板单独一行。
        """

        # 构建请求体，针对7B模型调整参数
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.9,  # 保持适当多样性
            "top_p": 0.9,
            "max_tokens": 3000,  # 7B模型建议适当减少最大token数
            "repetition_penalty": 1.1  # 增加重复惩罚，减少模板重复
        }

        try:
            # 发送请求
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload)
            )

            response.raise_for_status()
            result = response.json()

            # 解析响应
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                # 提取模板列表（每个模板占一行，带序号）
                templates = [line.strip() for line in content.split('\n') if
                             line.strip() and line.strip()[0] in '0123456789.']
                # 去除序号
                templates = [template.split('.', 1)[1].strip() if '.' in template else template for template in
                             templates]
                return templates[:num_templates]  # 确保不超过请求数量
            else:
                raise ValueError("未能从API响应中提取模板")

        except Exception as e:
            print(f"生成模板时出错: {str(e)}")
            return []



def main():
    # 配置API密钥（请替换为你的实际API密钥）
    API_KEY = "sk-820205eae5ec49abbd3e373a4f9384e8"

    # 定义行人特征
    attributes = [
        'gender', 'clothing type', 'age-related', 'physical characteristic',
        'color of the clothing or mountings', 'material characteristics of clothing or mountings',
        'mountings', 'Pedestrian actions', 'pedestrians of posture',
        'Background details', 'scene'
    ]

    # 初始模板
    initial_templates = [
        "Wearing [clothing description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
        "Sporting [hair description], the [person/woman/man] is dressed in [clothing description] and is carrying [belongings description].",
        # 可以根据需要添加更多初始模板
    ]

    # 创建生成器实例
    generator = Qwen25TemplateGenerator(API_KEY)

    print("正在生成初始精细化模板...")
    new_templates = generator.generate_templates(attributes, num_templates=50)  # 7B模型适当减少数量

    if new_templates:
        print(f"成功生成{len(new_templates)}个初始模板")

        if new_templates:
            print(f"成功生成{len(new_templates)}个优化模板")

            # 合并所有模板并去重
            all_templates = list(set(new_templates))

            # 保存结果
            with open("pedestrian_templates_qwen2.5.json", "w", encoding="utf-8") as f:
                json.dump(all_templates, f, ensure_ascii=False, indent=2)

            print(f"\n总共生成{len(all_templates)}个独特的精细化模板，已保存到pedestrian_templates_qwen2.5.json")

            # 打印部分示例
            print("\n模板示例：")
            for i, template in enumerate(all_templates[:5]):
                print(f"{i + 1}. {template}")
    else:
        print("生成模板失败")


if __name__ == "__main__":
    main()
