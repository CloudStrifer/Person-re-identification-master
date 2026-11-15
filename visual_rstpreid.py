import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 设置显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ==================== 导入模块 ====================
try:
    from utils.checkpoint import Checkpointer
    from model import build_model
    from utils.metrics import Evaluator
    from utils.iotools import load_train_configs
    from datasets import build_dataloader
except ImportError as e:
    print(f"导入项目模块出错: {e}")

# 导入 RSTPReid
try:
    from datasets.rstpreid import RSTPReid
except ImportError:
    try:
        from datasets.RSTPReid import RSTPReid
    except ImportError:
        print("Error: 找不到 RSTPReid 类")


# ==================== 工具函数定义 ====================

def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, gt_img_paths):
    """
    根据索引获取查询文本、检索结果和 Ground Truth
    """
    query_caption = captions[idx]
    query_id = qids[idx]

    # 获取预测出的前k个图像路径
    # indices[idx] 是前K个相似图片的索引
    retrieved_indices = indices[idx].tolist()
    image_paths = [img_paths[j] for j in retrieved_indices]

    # 获取预测出的前k个图像ID
    image_ids = gids[indices[idx]]

    # 获取 Ground Truth 图片路径
    gt_image_path = gt_img_paths[idx]

    return query_id, image_ids, query_caption, image_paths, gt_image_path


def plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path, fname=None):
    """
    绘制检索结果可视化图，包含 Rank 标题
    """
    print(f"Query ID: {query_id}")
    # print(f"Retrieved IDs: {image_ids}")
    print(f"Caption: {query_caption}")

    col = len(image_paths)
    fig = plt.figure(figsize=(15, 5))  # 调整画布宽一点，适合显示 Rank 标题

    # 1. Plot Ground Truth image (最左侧)
    plt.subplot(1, col + 1, 1)
    try:
        if gt_img_path:
            img = Image.open(gt_img_path)
            img = img.resize((128, 256))
            plt.imshow(img)
            plt.title("Ground Truth", color='blue', fontweight='bold')  #
        else:
            plt.text(0.5, 0.5, 'No GT', ha='center')
    except Exception as e:
        print(f"Error loading GT image: {e}")
    plt.xticks([])
    plt.yticks([])

    # 2. Plot Retrieval images (右侧列表)
    for i in range(col):
        plt.subplot(1, col + 1, i + 2)
        try:
            img = Image.open(image_paths[i])
            img = img.resize((128, 256))
            plt.imshow(img)
        except Exception as e:
            print(f"Error loading retrieval image: {e}")
            continue

        ax = plt.gca()  # 获取当前子图的坐标轴

        # --- ID 比较逻辑 (处理 Tensor 和 Int) ---
        # 确保 ID 类型一致进行比较
        current_img_id = image_ids[i].item() if hasattr(image_ids[i], 'item') else image_ids[i]
        target_id = query_id.item() if hasattr(query_id, 'item') else query_id

        is_match = (current_img_id == target_id)

        # 绿色代表正确，红色代表错误
        color = 'lawngreen' if is_match else 'red'

        # --- [关键修改] 添加 Rank 标题 ---
        title = f"Rank-{i + 1}"
        plt.title(title, color=color, fontweight='bold')

        # 设置边框颜色和粗细
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)  # 加粗边框使其更明显

        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    if fname:
        plt.savefig(fname, dpi=300)


# ==================== 主执行逻辑 ====================

if __name__ == "__main__":
    # 1. 配置文件路径
    config_file = 'F:/MultiMediaOpt/configs.yaml'  # RSTPReid 的配置文件
    checkpoint_path = 'F:/MultiMediaOpt/best2.pth'  # RSTPReid 的权重文件

    # 加载配置
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"找不到配置文件: {config_file}")

    args = load_train_configs(config_file)

    # 2. 设置环境与参数
    args.dataset_name = 'RSTPReid'  # 强制指定
    args.batch_size = 1024  # 根据显存调整
    args.training = False
    device = "cuda"

    # 3. 构建 DataLoader
    # build_dataloader 会自动读取 RSTPReid 数据
    print("Building dataloader...")
    test_img_loader, test_txt_loader, num_class = build_dataloader(args)

    # 4. 构建模型 (修正 num_classes)
    print(f"Building model with num_classes=37010...")
    model = build_model(args, num_classes=37010)

    # 5. 加载权重
    checkpointer = Checkpointer(model)
    if os.path.exists(checkpoint_path):
        checkpointer.load(checkpoint_path)
    else:
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

    model.to(device)

    # 6. 计算特征
    print("Computing embeddings...")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    # 这一步触发多进程 DataLoader，必须在 if main 保护下运行
    qfeats, gfeats, qids, gids = evaluator._compute_embedding(model.eval())

    # 归一化特征
    qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
    gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

    # 计算相似度矩阵
    print("Calculating similarity...")
    similarity = qfeats @ gfeats.t()

    # 获取 Top-10 结果
    _, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)  # q * topk
    indices = indices.cpu()

    # 7. 实例化 RSTPReid 数据集用于可视化读取图片路径
    # 确保当前目录下有 data 文件夹，或者修改 root 为绝对路径
    dataset = RSTPReid(root='./data')
    test_dataset = dataset.test

    img_paths = test_dataset['img_paths']
    captions = test_dataset['captions']
    gt_img_paths = test_dataset['gt_img_paths']

    # --- 可视化部分 ---
    print("Plotting results...")

    # query_idx 是查询样本的索引，修改它可以查看不同的 case
    query_idx = 1100

    query_id, image_ids, query_caption, image_paths, gt_img_path = get_one_query_caption_and_result_by_id(
        query_idx, indices, qids, gids, captions, img_paths, gt_img_paths
    )

    plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path)
    print("Done.")