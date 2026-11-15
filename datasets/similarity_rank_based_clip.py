import os
import torch
import numpy as np
from PIL import Image
from clip import clip
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_image_feature(image_path):
    """提取单张图片的CLIP特征"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

        return image_feature.cpu().numpy().flatten()
    except Exception as e:
        print(f"特征提取失败 {image_path}: {e}")
        return None


def select_generated_images_by_reference(original_folder, generated_folder, num_select=10):
    """
    根据原始图片，从生成的图片中选择相似度最高的

    Args:
        original_folder: 原始5张照片所在的文件夹 (例如 0000)
        generated_folder: 生成的20张照片所在的文件夹
        num_select: 要选择的生成图片数量

    Returns:
        selected_files: 选中的文件名列表
    """
    # 获取原始图片
    original_files = sorted([f for f in os.listdir(original_folder)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 获取生成图片
    generated_files = sorted([f for f in os.listdir(generated_folder)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(original_files) == 0:
        print(f"警告: 原始文件夹中没有图片: {original_folder}")
        return None

    if len(generated_files) < num_select:
        print(f"警告: 生成图片数量({len(generated_files)}) < 要选择的数量({num_select})")
        return None

    print(f"原始图片: {len(original_files)} 张, 生成图片: {len(generated_files)} 张")

    # 提取原始图片特征
    print("提取原始图片特征...")
    original_features = []
    for img_file in original_files:
        img_path = os.path.join(original_folder, img_file)
        feature = get_image_feature(img_path)
        if feature is not None:
            original_features.append(feature)

    if len(original_features) == 0:
        print("错误: 无法提取原始图片特征")
        return None

    original_features = np.array(original_features)
    # 计算原始图片特征的平均值（作为该类别的代表特征）
    avg_original_feature = original_features.mean(axis=0)
    avg_original_feature = avg_original_feature / np.linalg.norm(avg_original_feature)

    # 提取生成图片特征
    print("提取生成图片特征...")
    generated_features = []
    valid_generated_files = []

    for img_file in generated_files:
        img_path = os.path.join(generated_folder, img_file)
        feature = get_image_feature(img_path)
        if feature is not None:
            generated_features.append(feature)
            valid_generated_files.append(img_file)

    generated_features = np.array(generated_features)

    # 计算生成图片与原始图片平均特征的相似度
    print("计算相似度...")
    similarity_scores = cosine_similarity(
        generated_features,
        avg_original_feature.reshape(1, -1)
    ).flatten()

    # 按相似度排序，选择最高的num_select张
    top_indices = np.argsort(-similarity_scores)[:num_select]
    selected_files = [valid_generated_files[i] for i in top_indices]

    print(f"\n选中的 {num_select} 张生成图片 (按相似度从高到低):")
    for idx, file_idx in enumerate(top_indices):
        print(f"  {idx + 1}. {valid_generated_files[file_idx]} (相似度: {similarity_scores[file_idx]:.4f})")

    return selected_files, similarity_scores[top_indices]


def batch_process_all_pedestrians(original_base_path, generated_base_path,
                                  num_select=10, output_suffix="_selected"):
    """
    批量处理所有行人类别

    Args:
        original_base_path: 原始数据的基础路径 (包含 0000, 0001, 0002 等文件夹)
        generated_base_path: 生成数据的基础路径 (同样的文件夹结构)
        num_select: 每个类别选择的图片数量
        output_suffix: 输出文件夹后缀
    """
    # 获取所有行人类别
    pedestrian_ids = sorted([d for d in os.listdir(original_base_path)
                             if os.path.isdir(os.path.join(original_base_path, d))])

    print(f"找到 {len(pedestrian_ids)} 个行人类别\n")
    print("=" * 60)

    success_count = 0

    for ped_id in pedestrian_ids:
        print(f"\n处理类别: {ped_id}")


        original_path = os.path.join(original_base_path, ped_id)
        generated_path = os.path.join(generated_base_path, ped_id)
        output_path = os.path.join("E:/output_generate", ped_id + output_suffix)

        # 检查路径是否存在
        if not os.path.exists(generated_path):
            print(f"  跳过: 生成图片文件夹不存在")
            continue

        os.makedirs(output_path, exist_ok=True)

        # 选择图片
        result = select_generated_images_by_reference(
            original_path, generated_path, num_select
        )

        if result is None:
            continue

        selected_files, scores = result

        # 复制选中的图片到输出文件夹
        for img_file in selected_files:
            src = os.path.join(generated_path, img_file)
            dst = os.path.join(output_path, img_file)
            shutil.copy2(src, dst)

        print(f"  已保存到: {output_path}")
        success_count += 1
        print("=" * 60)

    print(f"\n处理完成！成功处理 {success_count}/{len(pedestrian_ids)} 个类别")


# 使用示例
if __name__ == "__main__":
    # 设置你的数据路径
    original_data_path = "G:/Code/DiVE-main/Data/Organize_Pedestrians-all"  # 原始数据的基础路径
    generated_data_path = "G:/Code/generate/生成的合格照片4(3001-4040)"  # 生成数据的基础路径

    # 批量处理所有行人
    batch_process_all_pedestrians(
        original_base_path=original_data_path,
        generated_base_path=generated_data_path,
        num_select=10,
        output_suffix="_selected"
    )