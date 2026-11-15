import os
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image
from datasets.cuhkpedes import CUHKPEDES


# ==========================================
# 1. 先定义函数（放在最外层，不要放在 if main 里）
# ==========================================

def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths):
    """
            idx: 要可视化的第几个 query (0 ~ len(captions))
            indices: 检索结果的索引矩阵
            qids: 所有 query 的 ID 列表
            gids: 所有 gallery 图片的 ID 列表
            """
    query_caption = captions[idx]
    query_id = qids[idx]  # 当前文本的真实 ID

    # 1. 获取预测（Retrieved）的图片路径和ID
    # indices[idx] 是前K个相似图片的索引
    retrieved_indices = indices[idx].tolist()
    image_paths = [img_paths[j] for j in retrieved_indices]
    image_ids = gids[indices[idx].cpu()]  # 这些是预测出来的图片ID

    # 2. 手动查找 Ground Truth (GT) 图片路径
    # 逻辑：在所有图片 ID (gids) 中，找到一个跟 query_id 一样的
    # 注意：gids 是 tensor，需要处理一下
    gt_image_path = None

    # 查找所有匹配该 ID 的图片索引
    # (gids == query_id) 会返回一个布尔 tensor，nonzero 取出索引
    match_indices = (gids == query_id).nonzero(as_tuple=True)[0]

    if len(match_indices) > 0:
        # 我们只取第一张匹配的图片作为 GT 展示
        gt_idx = match_indices[0].item()
        gt_image_path = img_paths[gt_idx]
    else:
        print(f"Warning: No GT image found for ID {query_id}")
        # 如果找不到，随便拿一张防止报错，或者设为 None
        gt_image_path = img_paths[0]

    return query_id, image_ids, query_caption, image_paths, gt_image_path

def plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path, fname=None):
    print(f"Query ID: {query_id}")
    print(f"Retrieved IDs: {image_ids}")
    print(f"Caption: {query_caption}")

    fig = plt.figure(figsize=(15, 5))
    col = len(image_paths)

    # 1. 画 Ground Truth 图片
    plt.subplot(1, col + 1, 1)
    try:
        if gt_img_path:
            img = Image.open(gt_img_path)
            img = img.resize((128, 256))
            plt.imshow(img)
            plt.title("Ground Truth", color='blue', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No GT', ha='center')
    except Exception as e:
        print(f"Error loading GT: {e}")
    plt.xticks([])
    plt.yticks([])

    # 2. 画检索结果
    for i in range(col):
        plt.subplot(1, col + 1, i + 2)
        try:
            img = Image.open(image_paths[i])
            img = img.resize((128, 256))
            plt.imshow(img)
        except:
            continue

        # 边框颜色逻辑
        ax = plt.gca()
        # 确保 ID 类型一致进行比较 (tensor 转 int)
        current_img_id = image_ids[i].item() if isinstance(image_ids[i], torch.Tensor) else image_ids[i]
        target_id = query_id.item() if isinstance(query_id, torch.Tensor) else query_id

        is_match = (current_img_id == target_id)

        # 绿色代表正确，红色代表错误
        color = 'lawngreen' if is_match else 'red'
        title = f"Rank-{i + 1}"

        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)  # 加粗一点

        plt.title(title, color=color)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    fig.show()
    if fname:
        plt.savefig(fname, dpi=300)


# ==========================================
# 2. 主执行逻辑（必须放在 if __name__ == '__main__': 下）
# ==========================================
if __name__ == '__main__':
    # 显卡设置放在这里
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # config_file  = '/xxx/configs.yaml'
    config_file = 'F:/BaiduNetdiskDownload/new/ham_finetuneIRRAcuhk5e-6_bs128/configs.yaml'

    args = load_train_configs(config_file)
    args.batch_size = 1024  # 如果显存不够，可以适当调小，比如 128 或 256
    args.training = False
    args.dataset_name = 'CUHK-PEDES'
    args.root_dir = 'E:/PyCharm_Code/data/'
    device = "cuda"

    print("Building dataloader...")
    test_img_loader, test_txt_loader, num_class = build_dataloader(args)

    print("Building model...")
    model = build_model(args)
    checkpointer = Checkpointer(model)

    # checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    # 确保路径存在
    checkpoint_path = 'F:/BaiduNetdiskDownload/new/ham_finetuneIRRAcuhk5e-6_bs128/best0.pth'
    if os.path.exists(checkpoint_path):
        checkpointer.load(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model.to(device)

    print("Evaluating...")
    evaluator = Evaluator(test_img_loader, test_txt_loader)

    # 这一步触发多进程 DataLoader，必须在 if main 保护下运行
    qfeats, gfeats, qids, gids = evaluator._compute_embedding(model.eval())

    qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
    gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

    print("Calculating similarity...")
    similarity = qfeats @ gfeats.t()

    # acclerate sort with topk
    _, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)  # q * topk

    dataset = CUHKPEDES(root='./data')
    test_dataset = dataset.test

    img_paths = test_dataset['img_paths']
    captions = test_dataset['captions']
    #gt_img_paths = test_dataset['gt_img_paths']

    # idx is the index of qids(A list of query ids, range from 0 - len(qids))
    print("Plotting results...")
    query_id, image_ids, query_caption, image_paths, gt_img_path = get_one_query_caption_and_result_by_id(
        260, indices, qids, gids, captions, img_paths
    )

    plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path)
    print("Done.")