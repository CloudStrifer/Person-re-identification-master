import os


def find_missing_folders(directory):
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return []

    # 生成应该存在的文件夹名称列表（0000到1000，共1001个）
    expected_folders = [f"{i:04d}" for i in range(0, 1001)]

    # 获取目录中实际存在的文件夹名称
    existing_folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):  # 只考虑文件夹
            existing_folders.append(item)

    # 找出缺失的文件夹
    missing_folders = [folder for folder in expected_folders if folder not in existing_folders]

    return missing_folders


if __name__ == "__main__":
    # 让用户输入目标目录
    target_dir = input("请输入要检查的目录路径：").strip()

    # 查找缺失的文件夹
    missing = find_missing_folders(target_dir)

    # 输出结果
    if missing:
        print(f"\n共发现 {len(missing)} 个缺失的文件夹：")
        # 每10个打印一行，方便查看
        for i, folder in enumerate(missing, 1):
            print(folder, end="  ")
            if i % 10 == 0:
                print()
        print()  # 最后换行
    else:
        print("\n所有0000到1000的文件夹都存在！")