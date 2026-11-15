import json
import os
import re
from tqdm import tqdm
from dashscope import Generation  # 阿里通义千问官方SDK
import dashscope

# 配置远程API
API_KEY = "sk-820205eae5ec49abbd3e373a4f9384e8"
Generation.api_key = API_KEY


def load_json_file(file_path):
    """加载JSON数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('“', '"').replace('”', '"').replace('，', ',').replace('。', '.')
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []
    except Exception as e:
        print(f"加载文件错误: {e}")
        return []


def is_exact_phrase_in_text(phrase, text):
    """严格验证短语是否为原文连续子串"""
    text_with_spaces = re.sub(r'([.,;])', r' \1 ', text)
    phrase_with_spaces = re.sub(r'([.,;])', r' \1 ', phrase)
    return re.search(rf'\b{re.escape(phrase_with_spaces)}\b', text_with_spaces) is not None


def clean_phrase(phrase, original_text):
    """清洗短语"""
    phrase = re.sub(r'^[-*•:]\s*', '', phrase.strip())
    if '.' in phrase or len(phrase.split()) > 6:
        return None
    if not is_exact_phrase_in_text(phrase, original_text):
        return None
    invalid_keywords = ['output', 'answer', 'explain', '说明']
    if any(keyword in phrase.lower() for keyword in invalid_keywords):
        return None
    return phrase


def extract_phrases_remote(text):
    """调用远程Qwen2.5-7B API提取短语"""
    prompt = f"""任务：从以下文本中提取描述人物穿着、配饰和动作等的属性短语。
严格要求：
1. 短语必须是原文中连续出现的完整片段，不允许省略任何单词（包括a/an/the/his等）。
2. 每个短语一行，仅包含短语本身，不添加解释、标题或额外内容。
3. 只保留与穿着、配饰、动作等相关的短语，其他内容忽略。
4. 每句话提取3-5个短语即可
文本：{text}

正确示例:
a grey jacket
black shoes
A clear plastic bag
a gray and black shirt
long hair
a black tight jacket
long brown hair

"""

    response = dashscope.Generation.call(
        api_key='sk-820205eae5ec49abbd3e373a4f9384e8',
        model="qwen2.5-7b-instruct",
        prompt=prompt,
        parameters={
            "temperature": 0.2,
            "max_new_tokens": 100,
            "top_p": 0.8
        }
    )

    if response.status_code == 200:
        result = response.output.text.strip()
        raw_phrases = [line.strip() for line in result.split('\n') if line.strip()]
        valid_phrases = [p for p in [clean_phrase(phrase, text) for phrase in raw_phrases] if p is not None]
        return valid_phrases
    else:
        print(f"API调用失败: {response.message}")
        return []


def append_phrases(phrases, output_file, existing_phrases):
    """追加新短语到文件，避免重复"""
    new_phrases = []
    for phrase in phrases:
        if phrase not in existing_phrases:
            new_phrases.append(phrase)
            existing_phrases.add(phrase)  # 添加到集合中，用于后续去重

    # 追加新短语到文件，每行一个短语
    if new_phrases:
        with open(output_file, 'a', encoding='utf-8') as f:  # 使用'a'模式追加
            for phrase in new_phrases:
                f.write(f"{phrase}\n")  # 仅写入短语本身，不加其他内容
    return new_phrases


def main(json_file_path, output_file_path):
    # 初始化输出文件（清空已有内容）
    with open(output_file_path, 'w', encoding='utf-8') as f:
        pass  # 清空文件

    # 用于跟踪已保存的短语，避免重复
    existing_phrases = set()

    print("加载JSON数据...")
    data = load_json_file(json_file_path)
    if not data:
        print("没有有效的数据可处理")
        return

    total_new_phrases = 0
    print("开始调用远程Qwen2.5-7B API提取短语...")

    for item in tqdm(data, desc="处理进度"):
        for title in item['captions']:
            phrases = extract_phrases_remote(title)
            # 追加新短语到文件
            new_phrases = append_phrases(phrases, output_file_path, existing_phrases)
            total_new_phrases += len(new_phrases)

    print(f"共提取到 {total_new_phrases} 个独特短语")
    print(f"短语已保存到 {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='调用远程Qwen2.5-7B API提取短语')
    parser.add_argument('--input', type=str, default='../data/RSTPReid/data_captions.json', help='输入JSON文件路径')
    parser.add_argument('--output', type=str, default='../data/RSTPReid/phrases.txt', help='输出短语文件路径')
    args = parser.parse_args()
    main(args.input, args.output)
