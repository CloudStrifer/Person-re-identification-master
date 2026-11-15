# NCHNet 分阶段恢复

当前已接入基线、A2DFP/CDFP、训练期 NAC 和离线图文扩充入口，默认仍为 **CLIP + SDM + ID**。各模块与图文扩充可独立开关。按用户要求不再创建快照。

- 进度：[恢复计划](NCHNET_RECOVERY_PLAN.md)
- 方法解释：[阶段 0 实现约定](docs/NCHNET_STAGE0_DECISIONS.md)
- 修改与验证：[阶段 1 记录](docs/NCHNET_STAGE1_BASELINE.md)
- 模块及服务器命令：[阶段 2 记录](docs/NCHNET_STAGE2_FEATURES.md)
- 离线图文扩充与续接命令：[阶段 4 说明](docs/NCHNET_STAGE4_AUGMENTATION.md)
- 预训练、迁移、微调与消融：[阶段 5 运行](docs/NCHNET_STAGE5_EXPERIMENTS.md)
- 当前进度及中断恢复：[恢复进度](docs/RECOVERY_PROGRESS.md)
- Ubuntu 服务器常用命令：[命令速查](docs/UBUNTU_COMMANDS.md)
- 原始版本：[快照说明](.recovery/README.md)

## 环境

本次实际使用 Python 3.12.14、PyTorch 2.7.1+cpu、torchvision 0.22.1+cpu，安装在仓库的 `.venv`。验证版本写入 `requirements.txt` 和 `requirements-dev.txt`；完整记录在 `.recovery/stage1_validation/environment.json`。

已有环境可直接使用 `./.venv/Scripts/python.exe`。如在另一台机器重新建环境，先用 Python 3.12 创建 venv，再安装成对的 torch/torchvision。本次采用的 CPU 组合来自 [PyTorch 官方版本清单](https://pytorch.org/get-started/previous-versions/#v271)：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

CUDA 训练需另装匹配的 CUDA 构建。当前本机为 4GB RTX 3050 Laptop，本次未验证 GPU 训练或论文规模 batch size 的显存需求。

## 数据和权重

`--root_dir` 可以是数据父目录或所选数据集目录。RSTPReid 默认结构：

```text
data/
  bpe_simple_vocab_16e6.txt.gz
  RSTPReid/
    data_captions.json
    imgs/...
```

本阶段扫描确认 RSTPReid 的 20,505 张图片均存在：18,505 张训练、1,000 张验证、1,000 张测试。人工标注共有 41,010 条描述。现有生成描述 JSON 不是原始标注的替代品。

BPE 已与 [OpenAI CLIP 官方资源](https://github.com/openai/CLIP/tree/main/clip)核对一致。官方 ViT-B/16 权重已下载到 `.recovery/model_cache/ViT-B-16.pt` 并核验其官方 SHA-256；来源、校验值和许可证见 `data/CLIP_ASSETS.json`、`data/CLIP_LICENSE.txt`。正常运行不自动下载；显式 `--allow_download` 才允许模型下载。

先检查资源（仅检查存在性，不启动训练）：

```powershell
.\.venv\Scripts\python.exe train.py --check_resources --download_root .recovery/model_cache
```

## 已验证的检查

运行自动检查：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

检查包括临时原始格式标注和图片、真实 BPE、微型真实 CLIP 架构、前后向、完整训练/保存/恢复/独立评估，以及手算检索指标。临时数据明确用于测试，不作为论文结果。

真实 RSTPReid + 官方 CLIP 的单步验证命令如下，包含一次 optimizer 更新及小型官方验证子集，不执行完整 epoch：

```powershell
.\.venv\Scripts\python.exe tools/validate_baseline.py --clip_weights .recovery/model_cache/ViT-B-16.pt --device cpu
```

报告默认写入 `.recovery/stage1_validation/real_data_step.json`。其中子集指标仅作计算流程检查，不可作为完整数据集检索结果。

## 正式基线入口

以下是保留论文训练轮数、学习率和 batch size 的命令示例，**本次没有执行完整训练**；需要准备合适的 CUDA 环境并实测显存：

```powershell
python train.py --mode baseline --dataset_name RSTPReid --root_dir data --clip_weights .recovery/model_cache/ViT-B-16.pt --device cuda --precision fp32 --batch_size 64 --output_dir logs/rstp_baseline
```

默认 seed 42、30 epochs、5 epochs warmup、cosine、骨干学习率 `1e-5`、新增参数 `1e-4`、weight decay `4e-5`；图像增强开启，文本保持原始干净输入。调小 batch 需要记录为不同实验配置。

输出包括 `configs.json`、`history.jsonl`、`train_log.txt`、`last.pth`；存在官方验证集时保存 `best.pth`，按验证 Rank-1 选择，同分保留较早模型。没有验证集时使用固定训练结束后的 `last.pth`。训练阶段不评估测试集。

按 epoch 边界恢复（继续原 run 目录和训练计划）：

```powershell
python train.py --resume logs/rstp_baseline/last.pth
```

恢复包含参数、优化器、调度器、AMP scaler（启用时）、随机状态和训练 loader 状态，并检查标注校验值及身份映射。更换训练轮数、precision、worker 数或输出目录不能作为精确恢复；应建立新的实验。迁移设备可恢复状态，但不保证逐位数值相同。尚不支持从 epoch 中间的某个 batch 继续。

独立测试只使用 checkpoint 保存的模型架构，不需要再次读取原始 CLIP 权重：

```powershell
python test.py --checkpoint logs/rstp_baseline/best.pth --root_dir data --device cuda --output logs/rstp_baseline/test_metrics.json
```

`--mode finetune --finetune PATH` 使用已保存的骨干初始化新实验，重置身份分类器，不加载优化器状态；`finetune.py` 和 `train_one_dataset.py` 均调用相同实现。完整 SYNTH-PEDES 预训练、跨数据集评估、微调和消融配置已提供，见阶段 5 运行说明；正式实验尚未执行。

## NAC 使用

`--use_nac` 开启训练期纠正，生成器默认 6 层、CMT 4 层、噪声阈值 0.7、生成损失权重 0.5。`--loss_names` 保持 `sdm+id`，NAC 由开关控制。

人工描述按 `--nac_corruption_rate 0.15` 对可替换属性跨度独立抽样，使用相同 BPE 长度的普通属性词替换，保留干净目标。生成 CE 的位置由视觉检测掩码决定；人工噪声真值只用于诊断。没有对齐目标的 LLM/合成描述不伪造逐位置监督。独立评估保持原始查询，默认不调用 NAC。

Ubuntu 服务器示例（替换实际路径，完整训练本机未执行）：

```bash
python train.py --name nac_joint --use_nac --use_a2dfp --use_cdfp --root_dir /data/reid --bpe_path /data/reid/bpe_simple_vocab_16e6.txt.gz --clip_weights /models/ViT-B-16.pt --device cuda
```

从先前不含 NAC 的 checkpoint 建立新实验时，添加 `--mode finetune --finetune PATH --initialize_new_modules`；现有模型继续训练则用 `--resume PATH`，不初始化新参数。中断恢复命令和最新验证结果见恢复进度文件。
