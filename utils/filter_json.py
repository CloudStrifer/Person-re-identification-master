
# 读取原始JSON文件
#input_file = "F:/BaiduNetdiskDownload/new/synthpedes-dataset.json"  # 替换为你的JSON文件路径
#output_file = "F:/BaiduNetdiskDownload/new/filter-dataset.json"  # 筛选后的输出文件路径

import json

# # 读取原始JSON文件
# with open('F:/BaiduNetdiskDownload/new/synthpedes-dataset.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 过滤只保留Part1的记录（精确匹配，不包括Part11、Part12等）
# filtered_data = [item for item in data if item['file_path'].startswith('Part1/')]
#
# # 将过滤后的数据写入新文件
# with open('F:/BaiduNetdiskDownload/new/filter-dataset.json', 'w', encoding='utf-8') as f:
#     json.dump(filtered_data, f, indent=4, ensure_ascii=False)
#
# print(f"原始记录数: {len(data)}")
# print(f"过滤后记录数: {len(filtered_data)}")
# print("已保存到 filtered_file.json")

import json

# 读取原始JSON文件
with open('F:/BaiduNetdiskDownload/new/synthpedes-dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤逻辑：
# 1. 路径以 'Part1/' 开头（精确匹配Part1）
# 2. Part1后第一个文件夹是≤15000的数字（如Part1/12345/...保留，Part1/16000/...排除）
filtered_data = []
for item in data:
    file_path = item['file_path']
    # 先判断是否以Part1/开头
    if file_path.startswith('Part1/'):
        # 分割路径，取Part1后的第一个部分（如"Part1/12345/x.jpg" → 取"12345"）
        parts = file_path.split('/')
        # 确保路径格式正确（至少有3部分：Part1/数字/文件名）
        if len(parts) >= 3 and parts[1].isdigit():
            num = int(parts[1])
            if num <= 15000:
                filtered_data.append(item)

# 保存过滤后的数据
with open('F:/BaiduNetdiskDownload/new/filter-dataset2.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print(f"原始记录数: {len(data)}")
print(f"过滤后记录数（Part1/1至Part1/15000）: {len(filtered_data)}")
print("已保存到 filter-dataset.json")