# 阶段 1：恢复基础训练与评估

日期：2026-11-10。范围：单设备、单数据集 CLIP + SDM + ID 基线。后续创新模块仍关闭。

## 完成内容

| 环节 | 已做修改 |
| --- | --- |
| 入口 | `train.py`、`test.py`、`processor/baseline.py` 共用模型和配置；旧两个训练入口改为兼容包装；不再嵌套 DP/DDP |
| 配置 | baseline 默认参数对齐计划；布尔值、图像尺寸、warmup/epochs 校验；额外目标和未恢复模块显式拒绝，不静默降级 |
| 数据 | `datasets/annotations.py` 统一原始标注解析；只载入所选数据集；训练 PID 连续映射、类别数按身份统计、同图多描述共享 image ID；保留 CPU 来源元数据 |
| 划分 | 原始目标集要求身份隔离；验证集缺失时不以测试集替代；SYNTH 无 split 时全量训练，有显式 split 则遵守标注 |
| 文本 | 干净输入、明确 attention mask，旧 MLM 复制数组后再修改；官方 CLIP 词表完整保留，避免旧 mask 占位删除 `jekyll` 导致编码错误；旧布局仅显式 opt-in |
| 图像 | 保留论文规定的 resize/flip/crop/erasing；损坏输入报错，不无限重试 |
| 模型 | 统一双编码器基线，去除按 batch 大小触发的旧 NAM 特例；CPU/FP32 默认，不再全模型无条件转 half；支持本地 CLIP 和按保存架构重建 |
| 损失 | SDM + 共享无偏置 ID 分类器；ID label smoothing 0.1，两模态 CE 相加；总损失明确组装，指标不再被重复求和 |
| 优化 | 骨干和新增模块显式学习率分组；稳定 warmup/cosine 调度；CUDA AMP 为可选项，本次仅验证 CPU/FP32 |
| checkpoint | 严格参数匹配、前缀规范化、安全格式；保存模型/训练配置、标签映射、optimizer/scheduler、RNG、loader/scaler 和数据签名；新身份微调重置 classifier，即使类别数相同 |
| 评估 | CPU 可用，独立编码、小图库和分块查询指标；训练仅使用 val，最后一轮保证执行验证；无 val 使用最终模型 |

## 资源复核结果

阶段 0 的“未见”是当时目录库存；本阶段重新扫描时，RSTPReid 原始标注、图片和 BPE 已可读取，没有覆盖这些本地资源。

- RSTPReid：20,505 条图像标注，对应图片均存在；train/val/test 分别为 18,505/1,000/1,000 张，身份数为 3,701/200/200；训练图文对 37,010。
- 标注 SHA-256：`97ea46fde66a22f41f821b563d0f89bb239baaf7d98b141aae367e8f470454fb`。
- BPE 与官方文件逐字节一致，SHA-256 为 `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`；未覆盖原文件。
- 下载官方 CLIP ViT-B/16 至 `.recovery/model_cache/ViT-B-16.pt`，350,837,078 字节，SHA-256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；核对官方模型 URL 中的摘要。许可证和资源来源见 `data/CLIP_ASSETS.json`。
- 本次仓库局部 `.venv` 使用 Python 3.12.14、torch 2.7.1+cpu、torchvision 0.22.1+cpu；未修改系统 Python。包版本已记录，CUDA 执行未验证。
- `nvidia-smi` 检测到 RTX 3050 Laptop、4GB 显存、驱动 581.83；未进行论文 batch size 的 GPU 显存实验。

## 验证层级

1. **静态与资源检查：** CLI 帮助可用；存在性检查返回所选数据集/词表/权重位置及依赖；原始标注与全部图片路径完成核对，41,010 条原始描述全部通过实际分词和 EOT/范围检查。原始快照摘要及 11 份既有备份完整性通过。
2. **小样例算法检查：** 真实 torch/torchvision 运算，覆盖数据副作用、标签、身份采样、词表、梯度、batch 独立性、前缀权重加载、手算 AP/INP、小图库、学习率和随机状态。
3. **完整微型流程：** 临时原始格式图像/标注 + 官方 BPE + 微型 CLIP；两轮训练模拟 epoch 边界中断并恢复，最终全部模型参数与连续训练逐位一致；移走原 CLIP 文件后恢复和独立测试仍通过；末轮在 eval_period 大于总轮数时仍能选模。
4. **真实数据单步：** 官方预训练 CLIP、原始 RSTPReid、输入 `[2,3,384,128]` / `[2,77]`，执行 SDM/ID 前向、反向和一次 Adam 更新。图像 patch、文本投影和分类器梯度均有限且非零、参数确实改变；独立样本编码通过。另用官方 val 的 2 个身份、4 张图像、4 条查询检查评估流程。

真实单步记录：SDM 约 2.642619，ID 约 16.431856，总损失约 19.074474。仅为一次流程检查，不代表收敛结果。小型 val 指标不作为论文或完整数据集结果报告。

报告位置：`.recovery/stage1_validation/` 下 `environment.json`、`resource_check.json`、`real_data_step.json` 和最终自动检查记录。

自动检查最终命令为 `.venv/Scripts/python.exe -m pytest tests -q`，结果 **42 passed**。分组：数据/词表 17、模型 11、训练工具 10、完整流程与输入边界 4。记录见 `pytest.txt`；完整描述检查见 `all_captions.json`；快照/源文件审计见 `change_audit.json`。

**尚未完成：** 完整真实数据 epoch/30 轮训练、完整 test 集指标、GPU/AMP 运行、多 worker 的完整训练、跨数据集实验、论文结果复现。代码实现和以上分级验证完成，不把这些未做事项计为完成实验。

## 下一步

进入阶段 2：在已验证基线上分别接入 A2DFP 和 CDFP，先做独立特征/损失检查，再联合验证。保留当前基线配置作为消融参照，不在此次改动中启用 NAC 或离线扩充。

技术依据：CLIP 标准编码及词表参考 [OpenAI CLIP](https://github.com/openai/CLIP)；版本组合参考 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/#v271)；权重恢复采用 [torch.load](https://docs.pytorch.org/docs/2.7/generated/torch.load.html) 的显式 `weights_only=True`；CUDA AMP 设计参考 [PyTorch AMP](https://docs.pytorch.org/docs/2.7/amp.html)，后者运行留待有对应环境时验证。
