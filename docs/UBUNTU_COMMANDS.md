# Ubuntu 服务器常用运行命令

适用环境：Ubuntu、单张 RTX 3090 24GB。以下使用 Bash，**所有命令均在项目根目录执行**。首次使用按“环境 → 数据检查 → 单步检查 → 基线训练”进行，其余按实验需要选择。

本文是命令速查；方法约定见[阶段 5 实验协议](NCHNET_STAGE5_PAPER_PROTOCOL.md)，详细扩充流程见[阶段 4 说明](NCHNET_STAGE4_AUGMENTATION.md)。命令已与当前入口参数核对，服务器 CUDA 和完整训练仍需实际运行验证。

## 1. 进入项目、准备环境

将第一行替换成服务器上的实际项目目录。已有合适环境时直接激活；Windows 的 `.venv` 需要在 Ubuntu 上重新创建。

```bash
cd /path/to/Person-re-identification-master

# 已安装 Python 3.12 时，新建环境（首次执行）
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA 12.6 构建示例，需与服务器 NVIDIA 驱动兼容
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-dev.txt
```

上述 torch/torchvision 对应仓库的固定版本，CUDA 安装组合依据 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/#v271)。使用其他 CUDA 构建时按该页选择匹配组合。

每次重新登录服务器，先进入项目并激活环境：

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

`available` 应为 `True`，再运行下文的 `--device cuda` 命令。

## 2. 默认数据、权重路径及资源检查

| 数据集参数 | 默认原始标注 | 默认图片位置 |
| --- | --- | --- |
| `RSTPReid` | `data/RSTPReid/data_captions.json` | `data/RSTPReid/imgs/` |
| `CUHK-PEDES` | `data/CUHK-PEDES/reid_raw.json` | `data/CUHK-PEDES/imgs/` |
| `ICFG-PEDES` | `data/ICFG-PEDES/ICFG-PEDES.json`，也支持 `ICFG_PEDES.json` | `data/ICFG-PEDES/imgs/` |
| `Synth_PEDES` | `data/Synth-Pedes/synthpedes-dataset.json` | `data/Synth-Pedes/Part1/…` 等，按标注路径放置 |

Linux 区分大小写；命令参数 `Synth_PEDES` 对应目录 `Synth-Pedes`。BPE 放在 `data/bpe_simple_vocab_16e6.txt.gz`，官方 CLIP 权重放在 `.recovery/model_cache/ViT-B-16.pt`。权重需随代码资源一起传到服务器。

```bash
# 只检查资源，不训练、不下载
python train.py --check_resources \
  --dataset_name RSTPReid --root_dir data \
  --clip_weights .recovery/model_cache/ViT-B-16.pt
```

检查其他数据集时，将 `RSTPReid` 换成上表的参数名。`--root_dir` 也可指定数据父目录或某个数据集目录。该命令检查资源存在性，实际图片解码和 CUDA 运算由下面的单步检查确认。

## 3. 首次试跑与自动检查

```bash
# 微型 CPU 全流程：不需要完整数据集，不使用 GPU
python tools/validate_experiments.py

# RSTPReid + 官方 CLIP，GPU 基线单步
python tools/validate_baseline.py \
  --root_dir data --clip_weights .recovery/model_cache/ViT-B-16.pt \
  --device cuda --output experiments/checks/baseline_cuda.json

# GPU 联合模块单步：NAC + A2DFP + CDFP，batch=4
python tools/validate_stage2.py --variant joint --use_nac \
  --root_dir data --clip_weights .recovery/model_cache/ViT-B-16.pt \
  --device cuda --output experiments/checks/joint_cuda.json

# 运行自动检查
python -m pytest tests -q
```

单步检查只做一次参数更新和少量验证数据的流程检查，不执行完整 epoch，也不输出正式论文指标。

## 4. 单数据集训练和模块开关

RSTPReid 基线（CLIP + SDM + ID）：

```bash
python train.py --mode baseline --name rstp_baseline \
  --dataset_name RSTPReid --root_dir data \
  --clip_weights .recovery/model_cache/ViT-B-16.pt \
  --device cuda --precision fp32 \
  --batch_size 64 --test_batch_size 128 --num_workers 4 \
  --sampler identity --num_instance 4 \
  --output_dir logs/rstp_baseline
```

三个训练期模块联合开启，使用原训练数据：

```bash
python train.py --mode baseline --name rstp_joint \
  --dataset_name RSTPReid --root_dir data \
  --clip_weights .recovery/model_cache/ViT-B-16.pt \
  --use_nac --use_a2dfp --use_cdfp \
  --device cuda --precision fp32 \
  --batch_size 64 --test_batch_size 128 --num_workers 4 \
  --sampler identity --num_instance 4 \
  --output_dir logs/rstp_joint
```

单独开启模块时，按下表保留所需开关，并使用不同输出目录：

| 模式 | 模块参数 |
| --- | --- |
| NAC | `--use_nac` |
| A2DFP | `--use_a2dfp` |
| CDFP | `--use_cdfp` |
| A2DFP + CDFP | `--use_a2dfp --use_cdfp` |
| 三模块联合 | `--use_nac --use_a2dfp --use_cdfp` |

默认为 30 epochs、5 epochs warmup、seed 42。消融比较时统一身份采样；CDFP 要求 batch 可被 `num_instance` 整除，且每批至少两个身份。显存试跑可改 `--batch_size 16`，并使用新的输出目录记录差异；开启 AMP 使用 `--precision amp`，其 CUDA 数值表现需另行验证。

训练输出位于指定目录：`last.pth`、有官方验证集时的 `best.pth`、`configs.json`、`history.jsonl`、`train_log.txt`。已经含有 `last.pth` 的目录使用续训命令；新实验使用新目录。

## 5. 中断续训与独立评估

```bash
# 按最后保存的 epoch 恢复；数据和 BPE 路径可覆盖旧机器路径
python train.py --resume logs/rstp_joint/last.pth \
  --root_dir data --bpe_path data/bpe_simple_vocab_16e6.txt.gz \
  --device cuda

# 独立测试 RSTPReid：使用验证集选出的 best
python test.py --checkpoint logs/rstp_joint/best.pth \
  --dataset_name RSTPReid --root_dir data \
  --bpe_path data/bpe_simple_vocab_16e6.txt.gz \
  --device cuda --test_batch_size 128 --num_workers 4 \
  --output logs/rstp_joint/test_metrics.json
```

恢复包含 optimizer、scheduler 和随机状态；当前不支持从 epoch 中间某个 batch 恢复。精确续训保持原 batch、训练轮数、精度和 worker 数。增强实验迁移路径时额外传 `--augmentation_manifest 新路径`，清单内容应相同。

有官方 val 的目标使用 `best.pth`；没有官方 val 的 ICFG-PEDES 默认使用 `last.pth`。独立评估不需要原始 CLIP 文件或训练用增强文件，读取 checkpoint 中保存的架构。

## 6. 手动预训练、直接迁移与微调

完整 SYNTH 源训练的默认恢复约定：A2DFP + CDFP，NAC 关闭，源数据不再额外扩充。目标人工描述微调时再新增 NAC。具体依据见实验协议说明。

```bash
# SYNTH 预训练
python train.py --mode pretrain --name synth_pretrain \
  --dataset_name Synth_PEDES --root_dir data \
  --clip_weights .recovery/model_cache/ViT-B-16.pt \
  --use_a2dfp --use_cdfp --sampler identity --num_instance 4 \
  --device cuda --batch_size 64 --test_batch_size 128 --num_workers 4 \
  --output_dir logs/synth_pretrain

# 直接迁移：不在 RSTPReid 上训练，使用源最终 checkpoint
python test.py --checkpoint logs/synth_pretrain/last.pth \
  --dataset_name RSTPReid --root_dir data \
  --bpe_path data/bpe_simple_vocab_16e6.txt.gz \
  --device cuda --test_batch_size 128 --num_workers 4 \
  --output logs/synth_pretrain/direct_rstp.json

# 从源模型创建目标微调实验，重置身份分类器及 optimizer
python train.py --mode finetune --finetune logs/synth_pretrain/last.pth \
  --initialize_new_modules --name rstp_finetune \
  --dataset_name RSTPReid --root_dir data \
  --use_nac --use_a2dfp --use_cdfp --sampler identity --num_instance 4 \
  --device cuda --batch_size 64 --test_batch_size 128 --num_workers 4 \
  --output_dir logs/rstp_finetune
```

该微调示例使用目标原训练数据；启用离线图文扩充时添加第 7 节的清单参数。切换目标数据集时，各目标从同一源 checkpoint 独立初始化。新增完整缺失的模块使用 `--initialize_new_modules`；已有模型继续训练使用 `--resume`。

## 7. 离线文本和图片扩充

先导出服务器上的原训练源图目录，再导入已有文本缓存：

```bash
python tools/augment.py catalog --dataset RSTPReid --root_dir data \
  --output data/augmentation/RSTPReid/catalog.json

python tools/augment.py import-captions \
  --catalog data/augmentation/RSTPReid/catalog.json \
  --captions captions/pedestrian_captions_results.json \
  --output data/augmentation/RSTPReid/text.json
```

导入范围仅为 train；不合格/重复描述记录在输出 report 中。新提取的属性缓存可替代已有描述缓存，生成两条模板描述：

```bash
python tools/augment.py text \
  --catalog data/augmentation/RSTPReid/catalog.json \
  --attributes attributes.json --limit 2 --seed 42 \
  --output data/augmentation/RSTPReid/text_new.json
```

仅启用已有文本扩充的训练示例：

```bash
python train.py --name rstp_text_aug --dataset_name RSTPReid --root_dir data \
  --clip_weights .recovery/model_cache/ViT-B-16.pt \
  --augmentation_manifest data/augmentation/RSTPReid/text.json --use_llm_augment \
  --use_nac --use_a2dfp --use_cdfp --sampler identity --num_instance 4 \
  --device cuda --batch_size 64 --test_batch_size 128 --num_workers 4 \
  --output_dir logs/rstp_text_aug
```

图片扩充需要另备本地 SD1.5、已训练的 LoRA/词嵌入和生成环境依赖。将下面两个 `/path/to/…` 替换成实际模型目录；这组命令先处理两张源图。

```bash
python tools/augment.py images \
  --catalog data/augmentation/RSTPReid/catalog.json \
  --local_model_dir /path/to/sd15 --checkpoint_dir /path/to/reid_lora/final_model \
  --num_candidates 20 --limit 2 --seed 42 --device cuda --dtype float16 \
  --output data/augmentation/RSTPReid/candidates.json

python tools/augment.py filter \
  --catalog data/augmentation/RSTPReid/catalog.json \
  --candidates data/augmentation/RSTPReid/candidates.json \
  --clip_weights .recovery/model_cache/ViT-B-16.pt --keep 10 --device cuda \
  --output data/augmentation/RSTPReid/selected.json

python tools/augment.py caption-images \
  --catalog data/augmentation/RSTPReid/catalog.json \
  --selected data/augmentation/RSTPReid/selected.json \
  --attributes generated_attributes.json --seed 42 \
  --output data/augmentation/RSTPReid/image_captions.json

python tools/augment.py merge \
  --catalog data/augmentation/RSTPReid/catalog.json \
  --inputs data/augmentation/RSTPReid/text.json data/augmentation/RSTPReid/image_captions.json \
  --output data/augmentation/RSTPReid/train_manifest.json
```

`generated_attributes.json` 的键为 selected 清单中的生成图 `image_path`，内容取自生成图自身。生成图须有自身描述再进入训练；完成后，将文本训练示例改为 `--augmentation_manifest data/augmentation/RSTPReid/train_manifest.json --use_llm_augment --use_generated`，并使用新输出目录。

属性提取也支持显式 Qwen provider；服务凭据、端点和模型配置见阶段 4 说明。上述图片命令中的 `--limit 2` 是试跑范围，省略 limit 默认处理当前全部源图。扩充命令用相同输入/参数再次执行可续接，改变输入或参数使用新输出文件。catalog 包含机器路径，搬到 Ubuntu 后重新导出。

## 8. 按实验计划运行论文主流程与消融

该入口统一管理配置、依赖和选模；任务使用本节产生的训练记录，手动 `train.py` 产物不会自动登记为计划已完成。默认数据路径已在 `configs/experiments/paths.example.json` 中设为 `data/`。

```bash
# 生成主实验、核心消融和扩充消融计划，不启动训练
python tools/experiments.py plan \
  --paths configs/experiments/paths.example.json --output_dir experiments/paper \
  --group main --group module_ablation --group augmentation_ablation

# 列出任务；查看单项详细配置时追加 --job 任务ID
python tools/experiments.py show --plan experiments/paper/experiments.json

# 主源预训练
python tools/experiments.py check --plan experiments/paper/experiments.json --job seed42-synth-pretrain
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-synth-pretrain

# 源完成后，直接迁移 → 独立目标微调 → 测试
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-direct-rstpreid
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-finetune-rstpreid
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-test-finetune-rstpreid

# 消融共用的 baseline 源预训练，然后运行某个消融及其评估
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-synth-baseline
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-rstp-baseline
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-test-rstp-baseline

# 计划任务中断后恢复
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-finetune-rstpreid --resume

# 汇总：未测项目保持 pending，输出 JSON/CSV
python tools/collect_results.py --root experiments/paper \
  --plan experiments/paper/experiments.json \
  --output_json experiments/paper/summary.json --output_csv experiments/paper/summary.csv
```

任务 ID 中 `rstpreid` 可按实际任务替换为 `cuhk-pedes`、`icfg-pedes`。核心消融 ID、敏感性配置和初始化约定见[阶段 5 运行说明](NCHNET_STAGE5_EXPERIMENTS.md)。每次 run 仅执行指定任务；未完成的源训练、缺失的双扩充清单会明确报错。

## 9. 后台运行和日志查看

服务器已有 tmux 时，可在会话里运行上面的任意训练命令：

```bash
tmux new -s nchnet
# 在 tmux 内进入项目，source .venv/bin/activate，再执行所需训练命令
# Ctrl+B 后按 D：离开会话，训练继续

# 重新连接
tmux attach -t nchnet

# 查看训练日志；将路径改为对应 run
tail -n 50 -f logs/rstp_joint/train_log.txt

# 观察 GPU 使用情况
watch -n 2 nvidia-smi
```

日志显示 loss/validation；正式测试指标保存在单独的评估 JSON 中。断线后重新接入 tmux 即可；训练进程终止时用第 5 节或第 8 节的 resume 命令继续。
