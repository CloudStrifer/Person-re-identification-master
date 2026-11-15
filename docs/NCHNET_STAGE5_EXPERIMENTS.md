# 阶段 5：完整实验运行

本轮完成实验配置、单项执行、续训、跨集评估与结果汇总工具；完整论文实验尚未执行。服务器按用户提供的 **Ubuntu + 单张 RTX 3090 24GB** 准备，所有命令在项目根目录执行，不需要把服务器绝对路径写进代码。

## 数据按原格式放入 data

```text
data/
  bpe_simple_vocab_16e6.txt.gz
  Synth-Pedes/
    synthpedes-dataset.json
    Part1/7/1.jpg ...
  CUHK-PEDES/
    reid_raw.json
    imgs/cam_a/... imgs/cam_b/... ...
  ICFG-PEDES/
    ICFG-PEDES.json          # 或 ICFG_PEDES.json
    imgs/train/... imgs/test/... ...
  RSTPReid/
    data_captions.json
    imgs/...
  augmentation/
    CUHK-PEDES/train_manifest.json
    ICFG-PEDES/train_manifest.json
    RSTPReid/train_manifest.json
    RSTPReid/train_unfiltered_manifest.json
```

CUHK/ICFG 原始标注为 JSON 列表，每条有 `id`、`file_path`、`captions`、`split`。路径相对于 `imgs/`；划分读取 `split`，不能用图片目录的 train/test 名字代替。CUHK 格式依据[原作者仓库](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description#data-preparation)；ICFG 格式及目录依据[原作者 SSAN](https://github.com/zifyloo/SSAN#dataset-preparation)。ICFG 两种标注名均支持；若同时存在但内容不同，会明确报错，避免选错标注。

SYNTH 每条同样读取 `id`、`file_path`、`captions`，`file_path` 如 `Part1/7/1.jpg` 相对 `Synth-Pedes/`；没有 split 时全部作为 train，保留全部身份和图片，不自行切 95%。这与 [PLIP 原作者发布的 SYNTH 格式](https://github.com/Zplusdragon/PLIP#synth-pedes)一致。发布包若标注文件名不同，按上表命名即可，不需转换已有字段。属性、prompt 等额外字段可保留，训练读取 captions。

增强清单来自阶段 4，生成图片相对于各自清单目录；原始数据可先放好再准备增强。Text-only 缓存不能充当图文双扩充。默认源 SYNTH 使用完整发布数据，不再额外扩充；目标 full 实验启用图文双扩充。

## 生成计划，不启动训练

默认路径在 `configs/experiments/paths.example.json`，均相对项目根目录。按上表放置时可直接使用；有不同存储位置时复制该 JSON 并修改路径。CLIP 默认 `.recovery/model_cache/ViT-B-16.pt`，需一并带到服务器。

```bash
python tools/experiments.py plan --paths configs/experiments/paths.example.json --output_dir experiments/paper --group main --group module_ablation --group augmentation_ablation
python tools/experiments.py show --plan experiments/paper/experiments.json
python tools/experiments.py show --plan experiments/paper/experiments.json --job seed42-synth-pretrain
```

计划只写 JSON；每项的配置和依赖可先查看。相同计划可重复生成，参数变化使用新目录。省略 group 时也包含 Fig.6/7 敏感性配置；全部支持配置共 67 个任务（33 训练、34 评估），不表示必须一次运行。四个参数的默认点复用同配置实验，通过 `reused_evaluations` 追踪，不重复训练。

已配置 Table 5 六行核心消融、Table 8 其余扩充消融及 Table 12 当前支持的四行；不是凭三模块生成八组合。未实现的 Table 9 特殊 NAC 分支、Table 12 static average，以及额外质量分析单列在计划 `unsupported_experiments` 和[论文协议说明](NCHNET_STAGE5_PAPER_PROTOCOL.md)，不能标为已复现。

## 主流程：逐项运行

先做资源检查（存在性与 CUDA 可用性，不下载）：

```bash
python tools/experiments.py check --plan experiments/paper/experiments.json --job seed42-synth-pretrain
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-synth-pretrain
```

源预训练：A2DFP+CDFP，NAC 关闭；完整 SYNTH synthetic 文本没有对齐干净目标，不能伪造生成器 CE。目标 fine-tune 才显式新增 NAC。这是恢复约定，论文未明确源阶段完整开关，详见协议说明。预设保留 train batch=64、30 epochs、seed=42，所有消融统一 identity sampler/K=4。

源训练成功后才能直接迁移和目标微调。以下以 RSTPReid 为例；CUHK/ICFG 的 ID 分别替换成 `cuhk-pedes`、`icfg-pedes`，每个目标都从同一源模型独立开始。

```bash
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-direct-rstpreid
python tools/experiments.py check --plan experiments/paper/experiments.json --job seed42-finetune-rstpreid
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-finetune-rstpreid
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-test-finetune-rstpreid
```

直接迁移不更新目标参数，不使用目标 test/val 选择源模型，使用源训练最终 `last.pth`。目标微调重置身份分类器及 optimizer，复用源骨干和特征分支，显式初始化 NAC。目标有官方 val 时按 val Rank-1 选最早最佳模型；无 val 时取最终 last。测试独立执行，不按测试分数择优。

训练任务成功后写 `runs/<ID>/experiment_result.json`，评估写 `results/<ID>.json`。缺数据/增强或依赖未完成会明确失败，不会静默关闭模块或拿中断 last 继续下游。每次 run 只执行指定任务，不自动启动整个实验矩阵；已完成的结果也不会覆盖。

## 消融、敏感性和续训

```bash
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-synth-baseline
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-rstp-baseline
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-test-rstp-baseline
```

六行核心目标 ID 为 `rstp-baseline`、`rstp-dual`、`rstp-dual_nac`、`rstp-dual_nac_a2dfp`、`rstp-dual_nac_cdfp`、`rstp-full`；均加 `seed42-`，评估前再加 `test-`。额外扩充行是 `rstp-text_only`、`rstp-image_only`、`rstp-dual_unfiltered`。这些实验统一从 baseline-only SYNTH checkpoint 初始化新增分支，明确记录共同初始化约定；不会从 full 权重静默删除模块。

Unfiltered 必须提供独立的无相似度筛选清单，不能直接复用 filtered 路径；生成图仍须有自身两条描述，并记录实际保留数量。未准备该清单时可先运行其余实验。

敏感性生成时增加 `--group sensitivity`；四组值已对齐论文横轴。改变 seed 可重复 `--seed 42 --seed 43`。缩小 batch、开启 AMP 或短轮数试跑，写到单独 overrides JSON，用 `plan --overrides FILE` 并使用新输出目录；这些变化会保存在计划中。不要把小 batch 的结果当成原 batch=64 的消融，CDFP 依赖批内正负组成，梯度累积也不能等价恢复该组成。

中断训练按 epoch 恢复，不重新初始化模块，也不再要求原始 CLIP 或源初始化文件：

```bash
python tools/experiments.py check --plan experiments/paper/experiments.json --job seed42-finetune-rstpreid --resume
python tools/experiments.py run --plan experiments/paper/experiments.json --job seed42-finetune-rstpreid --resume
```

实验目录内 run/result 路径相对 plan，可整体移动；原数据和增强路径按新机器重新设置。精确续训的标注、身份映射、增强摘要和训练计划必须保持一致。

## 汇总实际结果

```bash
python tools/collect_results.py --root experiments/paper --plan experiments/paper/experiments.json --output_json experiments/paper/summary.json --output_csv experiments/paper/summary.csv
```

每项未测保留 pending；计划配置和实测配置分别标明，指标缺失留空。检查来源 checkpoint 目录与预期任务一致，重复结果保留并标记冲突，不按 test 分数取最高。每项结果含权重/标注 SHA、初始化来源、选模规则、seed、开关与检索指标。runner 还记录耗时、环境及 CUDA 时 PyTorch allocator 的显存峰值；CPU 不生成虚假 GPU 数值。整机 GPU 占用与完整扩散开销需要服务器另测。

## 本轮验证和下一步

全套 **260 项检查通过**。微型真实 CLIP 检查已覆盖三个目标直接迁移零训练、源权重不变、同/异类数微调均重置分类器和 optimizer、严格续训；`python tools/validate_experiments.py` 已完成计划→源预训练→全模块/双扩充微调→测试→汇总的 CPU 样例，临时权重自动清理。报告 `.recovery/stage5_validation/pipeline_smoke.json` 明确为测试样例；没有把数值当论文结果。

后续在服务器可先运行上述 CPU smoke 确认依赖和流程，再用已有 RSTPReid 做一次真实 CUDA 单步（batch=4，保留默认模型尺寸，不跑完整 epoch）：

```bash
python tools/validate_stage2.py --variant joint --use_nac --clip_weights .recovery/model_cache/ViT-B-16.pt --device cuda --output experiments/cuda_smoke.json
```

之后再实测正式 batch=64 的显存并运行完整配置。本机没运行服务器训练；尚无完整 SYNTH/三目标正式指标。阶段 4 真实生成及额外论文分析仍按进度文件继续。
