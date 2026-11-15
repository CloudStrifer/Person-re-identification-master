# 阶段 2：A2DFP / CDFP 接入与验证

日期：2026-11-10。正式训练以用户的 Ubuntu 服务器为准；本机只做小规模验证。本阶段未连接或更改服务器，未执行完整训练。

## 已完成的实现

| 环节 | 实现 |
| --- | --- |
| A2DFP | 按论文式 (16)–(20)：重要性 MLP 与上下文门控的乘积做 softmax，调制局部序列，再学习 soft pooling；投影为 512→2048→ReLU→512 |
| 双模态与掩码 | 两模态同结构、非共享参数；视觉排除 CLS，文本排除控制 token、EOT 及其后的 padding，保留有效普通 token 0；全无有效局部信息时返回原全局表示 |
| 融合与推理 | 默认残差系数 0.5；训练和 `encode_image/encode_text` 共用特征路径，编码不接受 PID，不依赖评估 batch 的组成 |
| CDFP | 最终检索特征的原始余弦；排除自身，最难正样本、Top-2 负样本均值、margin=0.1 hinge；仅对有效 anchor 求均值 |
| 总目标 | `L_sdm + L_id + 0.1 * (L_cdfp_image + L_cdfp_text) / 2`，CDFP 只加权一次，分支诊断放在 metrics 中 |
| 独立开关 | A2DFP-only、CDFP-only、联合模式可独立选择；CDFP-only 不创建 A2DFP 参数，都关闭时保持阶段 1 基线 |
| 采样和日志 | `--sampler auto` 在 CDFP 开启时选身份采样，否则随机；CDFP 拒绝随机采样、单身份或 K<2 的配置；日志含有效及激活 anchor 数和双模态损失 |
| 权重 | 保存开关、架构和超参；独立评估按配置重建；精确续训不改变网络。新模块初始化需要显式允许，部分缺参或尺寸不符仍报错 |

2048 作为隐藏维度、CDFP 位于残差融合后、双模态取均值及不使用 0.07 温度缩放，均沿用阶段 0 的补充实现约定，不表示找回了历史实验设置。当前公式下显式拒绝 `--cdfp_temperature`，不添加未定义的 A2DFP 辅助正则。

## 验证结果

- 算法检查：手算困难样本与损失、梯度、缺正/负样本、Top-k 不足、零特征、空 batch、padding/NaN 屏蔽、全无有效局部信息、CPU bfloat16 autocast。
- 模型检查：独立和联合开关、手算总目标、CDFP 梯度到两编码器、双 A2DFP 的参数和梯度、冻结预设、普通 token 0、掩码边界、无 PID 推理、架构重建。
- 微型完整流程：三个变体真实运算的训练/保存/独立评估；移走原 CLIP 文件后仍可评估；联合模式 epoch 边界中断恢复后，最终全部模型参数与连续训练逐位一致。
- 基线对照：读取阶段 1 快照中的实际旧模型代码；相同 seed 和微型 CLIP 下，与当前关闭两个开关的模型比较，参数键、初始化和前向损失逐位一致。
- 真实数据：官方 CLIP ViT-B/16 + 用户提供的 RSTPReid。三个变体分别执行一个 CPU/FP32 Adam 更新，batch=4（2 身份×2 样本），4 张不同源图，输入 `[4,3,384,128]` / `[4,77]`。含 A2DFP 的两个变体均有 32/32 个增强参数张量获得有限非零梯度；CDFP-only 与联合模式均有 4 个有效 anchor，且存在激活的 hinge 项。
- 评估抽查：更新后单样本编码与 batch 编码一致；用官方 val 的 2 个身份、4 张图像、4 条查询检查指标流程，没有用 test 选模。

真实单步损失仅用于确认流程，不用于收敛或性能比较：

| 模式 | SDM | ID | 原始 CDFP |
| --- | --- | --- | --- |
| A2DFP-only | 3.927630 | 16.427994 | 未启用 |
| CDFP-only | 3.919917 | 16.423367 | 0.008481 |
| 联合 | 3.927630 | 16.427994 | 0.008535 |

本机使用 torch 2.7.1+cpu，不占用 4GB GPU 显存。CPU autocast 算法检查不等同 CUDA AMP 验证；Ubuntu、CUDA/AMP、完整 epoch、正式消融和论文指标均未实测。

报告在 `.recovery/stage2_validation/`：`pytest.txt`、`baseline_compatibility.json`、三个 `*_real_step.json` 及完整性审计。阶段 1 的程序/文档已另存 `.recovery/baselines/2026-11-10_stage1_baseline/`（99 文件，归档 SHA-256 `125c92e2ba2760a5a053c74f8c863938fe7edef07a20e45ea54f3c4465ee363d`）。

最终自动检查：`.venv/Scripts/python.exe -m pytest tests -q`，**82 passed**。包含阶段 1 的 42 项、算法 15 项、阶段 2 模型 11 项、完整流程 3 项和配置/权重边界 11 项。

## Ubuntu 命令与实验口径

在服务器使用自己的 Python/CUDA 环境；建议采用本次验证的 Python 3.12，不复制 Windows `.venv`。按驱动选择匹配的 torch/torchvision CUDA 构建，再安装项目依赖，参考 [PyTorch 官方版本清单](https://pytorch.org/get-started/previous-versions/#v271)。

新增 `scripts/run_stage2.sh`，在项目根目录运行。四个变体只改变模块开关，统一身份采样；下面服务器路径需替换为实际位置。这些为待执行命令，本机未运行 Ubuntu 脚本：

```bash
bash scripts/run_stage2.sh baseline --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --clip_weights /models/ViT-B-16.pt --device cuda
bash scripts/run_stage2.sh a2dfp   --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --clip_weights /models/ViT-B-16.pt --device cuda
bash scripts/run_stage2.sh cdfp    --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --clip_weights /models/ViT-B-16.pt --device cuda
bash scripts/run_stage2.sh joint   --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --clip_weights /models/ViT-B-16.pt --device cuda
```

脚本统一 `--sampler identity --num_instance 4`，默认 batch=64。**正式比较模块收益时，基线和所有变体均需相同采样设置**，避免把随机→身份采样的差异算作模块收益。

也可以直接使用 `python train.py --use_a2dfp true --use_cdfp true ...`。`--loss_names` 保持 `sdm+id`，CDFP 由独立开关控制，不额外添加 `cdfp` 字符串。

本机重复真实单步检查：

```powershell
.\.venv\Scripts\python.exe tools/validate_stage2.py --variant a2dfp --clip_weights .recovery/model_cache/ViT-B-16.pt
.\.venv\Scripts\python.exe tools/validate_stage2.py --variant cdfp --clip_weights .recovery/model_cache/ViT-B-16.pt
.\.venv\Scripts\python.exe tools/validate_stage2.py --variant joint --clip_weights .recovery/model_cache/ViT-B-16.pt
```

从阶段 1 基线权重初始化含 A2DFP 的新实验，需要显式允许初始化完整缺失的新分支：

```bash
python train.py --mode finetune --finetune /runs/baseline/last.pth --initialize_new_modules --use_a2dfp --use_cdfp --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --device cuda --output_dir /runs/joint_new
```

该命令重置身份分类器和 optimizer；输入尺寸/长度须匹配源 checkpoint。已有分支的部分缺参不会被跳过，`--resume` 始终完整严格加载。

迁移 checkpoint 到 Ubuntu 时，同时覆盖数据和 BPE 的 Windows 绝对路径；保存的架构已足够重建模型，不需要原 CLIP 文件：

```bash
python train.py --resume /runs/joint/last.pth --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --device cuda
python test.py --checkpoint /runs/joint/best.pth --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --device cuda
```

迁移时一并保留 run 目录的最佳 checkpoint。跨 CPU/CUDA 可恢复状态，但不承诺逐位数值相同。下一阶段为 NAC 的监督、纠正与下游使用流程。
