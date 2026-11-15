import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import re

def load_json_file(file_path):
    """加载JSON文件数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 修复可能的JSON格式错误
            content = f.read().replace('“', '"').replace('”', '"').replace('，', ',').replace('。', '.')
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        # 尝试修复常见错误
        try:
            # 处理可能的结尾不完整问题
            if content.endswith(','):
                content = content[:-1] + ']'
            return json.loads(content)
        except:
            print("无法修复JSON格式错误")
            return []
    except Exception as e:
        print(f"加载文件错误: {e}")
        return []


def load_model():
    """加载Qwen2.5-0.5B模型和tokenizer"""
    model_name = "../Qwen2.5-7B-Instruct"

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用float16节省内存
        device_map="auto",  # 自动分配设备
        trust_remote_code=True
    )

    # 模型设置为评估模式
    model.eval()

    # 创建文本生成管道
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return generator


def clean_phrase(phrase):
    """清洗短语：去除前缀符号、空内容和无关关键词"""
    # 去除开头的符号（如-、*、•等）
    phrase = re.sub(r'^[-*•\s]+', '', phrase.strip())

    # 过滤包含解释性关键词的行（如Explanation、answer、说明等）
    invalid_keywords = ['explanation', 'answer', '说明', '解析', '：']
    if any(keyword in phrase.lower() for keyword in invalid_keywords):
        return None

    # 过滤过短或无意义的短语
    if len(phrase) < 2:
        return None

    return phrase


def extract_phrases(generator, text):
    """使用模型从文本中提取描述性短语"""
    # 构建提示词
    # prompt = f"""Task: Extract phrases describing a character's attire, accessories, and actions from the following text.
    # requirement:
    # 1. Only return the phrase itself, one line per phrase
    # 2. Do not provide any explanations, markings (such as -, *), summaries, or additional explanations
    # 3. Phrases should be concise (such as "grey jacket" instead of "the man's grey jacket")
    # {text}
    # """
    # """严格约束的短语提取：强制匹配原文+限制格式"""
    # 提示词优化：明确限定“只提取原文中存在的短语”，并给出示例
    prompt = f"""Task: Extract phrases describing characters' clothing, accessories, and actions from the following text.
    Strict requirements:
    1. The phrase must be a complete fragment that appears continuously in the original text, and it is not allowed to omit, simplify, or rewrite any words (including a/an/he/his, etc.).
    2. Each phrase is on one line, containing only the phrase itself without adding any explanation, title, or additional contentc.
    3. Only extract phrases related to clothing, accessories, and actions, and ignore other content.


    text：{text}

    # Correct example output (phrase only):
    # grey jacket
    # blue shirt
    # hands in pockets
    # glasses
    # black bag
    """


    # 生成响应
    response = generator(
        prompt,
        max_new_tokens=100,  # 最大生成 tokens
        temperature=0.1,  # 控制随机性，值越低越确定
        top_p=0.9,
        repetition_penalty=1.3,  # 减少重复
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    # 提取生成的文本
    result = response[0]['generated_text'].replace(prompt, '').strip()

    # 清洗每一行短语
    raw_phrases = [line.strip() for line in result.split('\n') if line.strip()]
    clean_phrases = [p for p in [clean_phrase(phrase) for phrase in raw_phrases] if p is not None]

    return clean_phrases


def save_phrases(phrases, output_file):
    """保存短语到文件，每个短语一行"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for phrase in phrases:
            f.write(f"{phrase}\n")


def main(json_file_path, output_file_path):
    """主函数"""
    # 加载数据
    print("加载JSON数据...")
    data = load_json_file(json_file_path)
    if not data:
        print("没有有效的数据可处理")
        return

    # 加载模型
    print("加载Qwen2.5-0.5B模型...")
    generator = load_model()

    # 提取所有短语
    all_phrases = []
    print("开始提取短语...")

    for item in tqdm(data, desc="处理进度"):
        for title in item['captions']:
            # 提取短语
            phrases = extract_phrases(generator, title)
            all_phrases.extend(phrases)

    # 去重
    unique_phrases = list(set(all_phrases))
    print(f"共提取到 {len(unique_phrases)} 个独特短语")

    # 保存结果
    save_phrases(unique_phrases, output_file_path)
    print(f"短语已保存到 {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='从JSON文件的标题中提取短语')
    parser.add_argument('--input', type=str, default='../data/RSTPReid/data_captions.json', help='输入JSON文件路径')
    parser.add_argument('--output', type=str, default='../data/RSTPReid/phrases.txt', help='输出短语文件路径')

    args = parser.parse_args()

    main(args.input, args.output)
