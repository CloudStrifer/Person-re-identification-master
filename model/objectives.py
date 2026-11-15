import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加文本 - 文本对比损失函数
def compute_ttc(text_features, logit_scale, pids, batch_size_per_image=4, text_quality=None):
    """
    Text-Text Contrastive Loss: 同一实例的不同文本互为正例，不同实例为负例
    修复设备不匹配问题
    """
    # 原始的
    # batch_size = text_features.shape[0]
    # # 生成标签：同一pid的文本为正例（构建正例掩码）
    # pid_matrix = pids.unsqueeze(0) == pids.unsqueeze(1)  # [batch_size, batch_size]
    # # 排除自身对比（对角线置为False）
    # pid_matrix = pid_matrix & ~torch.eye(batch_size, dtype=torch.bool, device=text_features.device)
    #
    # # 若某样本无其他同pid文本，则将自身作为正例（避免无正例情况）
    # has_positive = pid_matrix.sum(dim=1) > 0
    # # 关键修复：将索引张量移动到pid_matrix所在设备
    # indices = torch.arange(batch_size, device=pid_matrix.device)  # 直接在目标设备生成索引
    # pid_matrix[~has_positive, indices[~has_positive]] = True  # 现在索引和被索引张量在同一设备
    #
    # # 归一化特征
    # text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    # # 计算文本-文本相似度
    # logits = logit_scale * text_norm @ text_norm.t()  # [batch_size, batch_size]
    #
    # # 计算交叉熵损失（每个样本的正例是所有同pid文本）
    # loss = F.cross_entropy(logits, pid_matrix.float())
    # return loss

    """
    改进点：
    1. 按`batch_size_per_image`分组，确保同一图片的4个文本互为强正例
    2. 弱化不同图片但同`pid`的文本关联性（若存在多图同`pid`场景）
    """


    # batch_size = text_features.shape[0] #因为有可能训练到最后一轮长度不匹配导致
    batch_size = pids.shape[0]
    device = text_features.device

    # ✅ 确保 text_features 大小与 pids 匹配
    if text_features.shape[0] != batch_size:
        print(f"[WARNING] text_features size {text_features.shape[0]} != pids size {batch_size}")
        text_features = text_features[:batch_size]  # 或截断或补充，根据实际情况

    # 1. 构建分组掩码：同一图片的4个文本为正例（组内正例）
    num_groups = batch_size // batch_size_per_image  # 总图片数
    group_labels = torch.repeat_interleave(
        torch.arange(num_groups, device=device),
        repeats=batch_size_per_image
    )  # 形状：[4N]，值为0,0,0,0,1,1,1,1,...（每组对应一个图片）
    group_matrix = (group_labels.unsqueeze(0) == group_labels.unsqueeze(1))  # [4N,4N]，组内为True

    # 2. 构建PID掩码：同PID的文本为弱正例（若需要保留跨图同PID关联）
    pid_matrix = (pids.unsqueeze(0) == pids.unsqueeze(1))  # [4N,4N]，同PID为True

    # 3. 最终正例掩码：组内必为正例，跨组同PID可选为正例（根据业务场景调整）
    # 若仅需同一图片的文本互为主正例，直接用group_matrix即可
    positive_mask = group_matrix  # 优先保证同一图片的文本一致性

    # 排除自身对比（对角线置为False）
    positive_mask = positive_mask & ~torch.eye(batch_size, dtype=torch.bool, device=device)

    # 确保每个样本至少有一个正例（避免Loss计算错误）
    has_positive = positive_mask.sum(dim=1) > 0
    indices = torch.arange(batch_size, device=device)
    positive_mask[~has_positive, indices[~has_positive]] = True

    # 计算组内平均相似度（奖励项）
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    sim_matrix = text_norm @ text_norm.t()  # [4N,4N] 余弦相似度
    group_sim = sim_matrix * group_matrix.float()  # 仅保留组内相似度
    avg_group_sim = group_sim.sum() / (group_matrix.sum() + 1e-8)  # 组内平均相似度

    # 交叉熵损失（惩罚项）
    logits = logit_scale * sim_matrix
    ce_loss = F.cross_entropy(logits, positive_mask.float())

    # 若提供文本质量权重，对损失进行加权
    if text_quality is not None:
        ce_loss = ce_loss * text_quality  # 质量高的文本，其损失权重更大

    # 总损失 = 惩罚项 - 奖励项（鼓励组内相似度升高）
    loss = ce_loss.mean() - 0.1 * avg_group_sim  # 0.1为奖励权重，可调整
    return loss


# 新增文本一致性损失函数
def compute_text_consistency_loss(text_feats, batch_size_per_image=4): #依然先改成2
    N = len(text_feats) // batch_size_per_image
    loss = 0.0
    text_feats = F.normalize(text_feats, dim=1)

    for i in range(N):
        # 正确取同一图片的4条文本
        same_texts = text_feats[i*batch_size_per_image : (i+1)*batch_size_per_image]  # [4, D]
        # 内部相似度：4条文本两两之间的平均相似度（原逻辑正确，分组后生效）
        intra_sim = torch.mean(F.cosine_similarity(same_texts.unsqueeze(1), same_texts.unsqueeze(0)))
        # 外部相似度：与其他所有图片的文本平均相似度
        other_texts = torch.cat([
            text_feats[:i*batch_size_per_image],
            text_feats[(i+1)*batch_size_per_image:]
        ], dim=0)
        inter_sim = torch.mean(F.cosine_similarity(same_texts.unsqueeze(1), other_texts.unsqueeze(0)))
        # 增强约束权重（原5倍可能不够，可暂时保持，后续调优）
        loss += -torch.log(intra_sim / (intra_sim + inter_sim + 1e-8))

    return loss / N

    # 原始的"""
    # 强化同一图像的4条文本之间的语义一致性
    # """
    # N = len(text_feats) // batch_size_per_image
    # loss = 0.0
    # text_feats = F.normalize(text_feats, dim=1)  # 归一化特征（余弦相似度）
    #
    # for i in range(N):
    #     # 当前图像的4条文本特征
    #     same_texts = text_feats[i * batch_size_per_image: (i + 1) * batch_size_per_image]  # [4, D]
    #
    #     # 内部相似度：4条文本彼此的平均余弦相似度
    #     intra_sim = torch.mean(F.cosine_similarity(
    #         same_texts.unsqueeze(1),  # [4,1,D]
    #         same_texts.unsqueeze(0)  # [1,4,D]
    #     ))  # [4,4] -> 均值
    #
    #     # 外部相似度：与其他所有行人文本的平均余弦相似度
    #     other_texts = torch.cat([
    #         text_feats[:i * batch_size_per_image],
    #         text_feats[(i + 1) * batch_size_per_image:]
    #     ], dim=0)  # [4(N-1), D]
    #
    #     inter_sim = torch.mean(F.cosine_similarity(
    #         same_texts.unsqueeze(1),  # [4,1,D]
    #         other_texts.unsqueeze(0)  # [1,4(N-1),D]
    #     ))  # [4,4(N-1)] -> 均值
    #
    #     # 约束：内部相似度 > 外部相似度
    #     loss += -torch.log(intra_sim / (intra_sim + inter_sim + 1e-8))  # 加epsilon避免除0
    #
    # return loss / N  # 平均到每个图像

# 替换原有的compute_cmpm函数
def compute_multi_cmpm_loss(image_feats, text_feats, batch_size_per_image=2): #同样先改成2
    """
     适配多文本场景的跨模态匹配损失
     """
    N = len(image_feats) // batch_size_per_image  # 实际图像数量
    loss = 0.0
    device = image_feats.device

    # 图像->文本的分布约束（原逻辑正确，无需修改）
    for i in range(N):
        img_feat = image_feats[i * batch_size_per_image]  # [D]
        pos_texts = text_feats[i * batch_size_per_image: (i + 1) * batch_size_per_image]  # [2, D]
        neg_texts = torch.cat([
            text_feats[:i * batch_size_per_image],
            text_feats[(i + 1) * batch_size_per_image:]
        ], dim=0)  # [2*(N-1), D]

        pos_sim = torch.mean(F.cosine_similarity(img_feat.unsqueeze(0), pos_texts))  # 图像与正文本的平均相似度
        neg_sim = torch.mean(F.cosine_similarity(img_feat.unsqueeze(0), neg_texts))  # 图像与负文本的平均相似度
        loss += F.relu(neg_sim - pos_sim + 0.1)  # margin=0.1

    # 文本->图像的分布约束（修复维度不匹配问题）
    for i in range(N):
        text_feats_i = text_feats[i * batch_size_per_image: (i + 1) * batch_size_per_image]  # [2, D]
        pos_img = image_feats[i * batch_size_per_image].unsqueeze(0)  # [1, D]

        # 负图像特征（去重后）
        neg_imgs = torch.cat([
            image_feats[:i * batch_size_per_image: batch_size_per_image],  # 每batch_size_per_image取一个（去重）
            image_feats[(i + 1) * batch_size_per_image:: batch_size_per_image]
        ], dim=0)  # [N-1, D]

        # 正样本相似度：当前文本与正图像的平均相似度（原逻辑正确）
        pos_sim = torch.mean(F.cosine_similarity(text_feats_i, pos_img))

        # 负样本相似度：当前文本与所有负图像的平均相似度（修复维度不匹配）
        # 扩展维度使广播生效：[2, D] → [2, 1, D]，[N-1, D] → [1, N-1, D]
        text_feats_expanded = text_feats_i.unsqueeze(1)  # [2, 1, D]
        neg_imgs_expanded = neg_imgs.unsqueeze(0)  # [1, N-1, D]
        # 计算所有文本与所有负图像的相似度（形状：[2, N-1]），再取平均
        neg_sim = torch.mean(F.cosine_similarity(text_feats_expanded, neg_imgs_expanded, dim=2))

        loss += F.relu(neg_sim - pos_sim + 0.1)

    return loss / (2 * N)  # 平均到每个样本

def compute_multi_itc_loss(image_feats, text_feats, logit_scale, batch_size_per_image=2): #先改成2，后面再改回4
    """
    适配多文本场景的跨模态对比损失
    image_feats: [4N, D]，同一图像特征重复4次（对应4条文本）
    text_feats: [4N, D]，4条文本的特征
    logit_scale: 温度系数（可学习参数）
    """
    # 计算图像-文本相似度矩阵（4N × 4N）
    logits_per_image = logit_scale * image_feats @ text_feats.t()
    logits_per_text = logits_per_image.t()

    # 构建标签：同一图像的4条文本共享相同标签（0,0,0,0,1,1,1,1,...）
    N = len(image_feats) // batch_size_per_image
    labels = torch.repeat_interleave(
        torch.arange(N, device=image_feats.device),
        repeats=batch_size_per_image
    )

    # 计算交叉熵损失（图像侧和文本侧对称）
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2


def compute_infonce(image_features, text_features, pid, logit_scale, image_id=None, temperature=0.1, epsilon=1e-8):
    """
    InfoNCE loss for cross-modal matching
    """
    batch_size = image_features.shape[0]

    # 特征归一化
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    # 计算相似度矩阵
    logits_per_image = logit_scale * image_norm @ text_norm.t()  # 图像到文本的相似度
    logits_per_text = logits_per_image.t()  # 文本到图像的相似度

    # 构建正样本标签（基于pid）
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    positive_mask = (pid_dist == 0).float()

    # 如果提供了image_id，进一步细化正样本
    if image_id is not None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        # 结合pid和image_id确定正样本
        positive_mask = positive_mask * image_id_mask

    # 确保每个样本至少有一个正样本
    positive_mask = positive_mask + torch.eye(batch_size, device=positive_mask.device) * (
                positive_mask.sum(dim=1, keepdim=True) == 0).float()

    # 计算每个样本的正样本数量
    num_positives = positive_mask.sum(dim=1, keepdim=True)

    # 对相似度进行温度缩放
    logits_per_image /= temperature
    logits_per_text /= temperature

    # 计算InfoNCE损失
    # 图像到文本方向
    exp_logits_image = torch.exp(logits_per_image)
    sum_exp = exp_logits_image.sum(dim=1, keepdim=True)
    denominator = sum_exp - exp_logits_image + epsilon
    i2t_loss = -torch.log((exp_logits_image * positive_mask) / denominator + epsilon)
    i2t_loss = (i2t_loss * positive_mask).sum(dim=1) / num_positives.squeeze()

    # 文本到图像方向
    exp_logits_text = torch.exp(logits_per_text)
    sum_exp_text = exp_logits_text.sum(dim=1, keepdim=True)
    denominator_text = sum_exp_text - exp_logits_text + epsilon
    t2i_loss = -torch.log((exp_logits_text * positive_mask) / denominator_text + epsilon)
    t2i_loss = (t2i_loss * positive_mask).sum(dim=1) / num_positives.squeeze()

    # 总损失
    loss = (torch.mean(i2t_loss) + torch.mean(t2i_loss)) / 2

    return loss

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = F.normalize(image_fetures.float(), dim=1)
    text_norm = F.normalize(text_fetures.float(), dim=1)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1, keepdim=True)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels, label_smoothing=0.1):
    """Paper equations (24)-(26): sum the two label-smoothed CE losses."""
    return (F.cross_entropy(image_logits.float(), labels, label_smoothing=label_smoothing)
            + F.cross_entropy(text_logits.float(), labels, label_smoothing=label_smoothing))


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

