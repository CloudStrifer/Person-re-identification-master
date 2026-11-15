# 恢复进度（中断后先读本文件）

更新：2026-11-10。工作目录：项目根目录。用户要求以功能实现为主，**不再创建新快照**。

## 已完成

- 阶段 1：CLIP + SDM + ID、单数据集训练/保存/恢复/独立评估。
- 阶段 2：A2DFP、CDFP 独立/联合开关；82 项检查通过；三个变体真实 RSTPReid 小批量 CPU 更新通过。
- 正式训练在 Ubuntu 服务器；本机测试环境为 `.venv/Scripts/python.exe`（torch 2.7.1+cpu）。RSTPReid 和 BPE 在 `data/`，官方 CLIP 在 `.recovery/model_cache/ViT-B-16.pt`。

## 当前：阶段 1–4 本地代码完成；阶段 5 配置与运行流程已接通

实现约定沿用 `docs/NCHNET_STAGE0_DECISIONS.md` D01–D03：

1. 用图像局部 patch 与文本 token 的余弦产生检测掩码 M，阈值 0.7。
2. 人工描述构造同 BPE 长度的普通属性替换，保存干净目标和人工噪声真值；生成 CE 只在 M 与有效目标的交集求均值，不能直接用人工噪声真值代替 M。
3. 文本/图像自注意力、跨注意力、CMT、6 层生成器预测；替换控制 token 以外的位置，保留 EOT。
4. 纠正重编码停止梯度；有噪样本选用纠正序列，之后才进行 A2DFP、SDM/ID/CDFP。生成损失权重 0.5。
5. 默认训练期 NAC；独立检索编码不依赖匹配图像。无对齐干净目标的生成文本不伪造逐位置监督。

当前任务：

- [x] `model/nac_module.py`、`model/build_finetune.py`：NAC 与下游连接；`tests/test_nac_model.py`。NAC 7 项及基线模型 11 项通过。
- [x] `datasets/nac_augment.py`、`datasets/bases.py`、`datasets/build.py`：可对齐的属性噪声和目标；`tests/test_nac_data.py`。NAC 数据 17 项及基础数据 17 项通过。
- [x] 参数/训练入口/权重初始化允许 NAC；3 项集成检查通过，包括联合训练的 epoch 边界恢复后参数逐位一致。
- [x] 完整 109 项检查通过；真实 RSTPReid + 官方 CLIP 的 NAC/A2DFP/CDFP 联合 batch=4 CPU 更新通过。
- [x] 本文件、README 和恢复计划已更新；没有创建新快照。

参数/入口已解除 NAC 限制；新增 `--nac_corruption_rate`；权重显式初始化已允许完整缺失的 `nac.` 分支。腐蚀对每个可替换属性跨度独立抽样，不强制每句有噪声。自注意力和跨注意力按论文式 (9)–(11) 实现，外部不额外添加残差，Transformer 内部残差保留。

最终检查记录：`.recovery/stage3_validation/pytest.txt`（109 passed）；真实单步记录：`.recovery/stage3_validation/joint_real_step.json`。真实检查使用默认 CMT=4/生成器=6、图像384×128、文本77；为覆盖替换路径，检查时腐蚀率设为1（正式默认0.15）。生成损失有效，NAC 词表投影/跨注意力、图文编码器、分类器及两增强分支均有有限非零梯度并更新。小型 val 只作流程检查。

生成器尚未经过完整训练，初始视觉检测存在较多误报；本轮确认功能和梯度，不声明纠错准确率、收敛或论文指标。Ubuntu/CUDA/AMP 和完整训练未执行。

## 阶段 4：完成内容与真实环境待办

- [x] `augmentation/text.py`、`augmentation/templates.py`：30 模板、视觉属性缓存/Qwen 提取、两条描述；48 项离线检查通过，无真实 API 调用。
- [x] `augmentation/images.py`：旧 LoRA/词嵌入加载、20 候选、逐原图 Top-10 筛选；15 项离线检查通过，无真实扩散运行。
- [x] `datasets/augmentation.py` 及加载器：清单合并到 train，来源/PID/图片/两条自身描述校验，增强描述不伪造 NAC 干净目标。
- [x] 参数/入口允许 `--augmentation_manifest` 配合 `--use_llm_augment`、`--use_generated`；checkpoint 校验增强清单摘要，评估忽略增强。
- [x] `augmentation/workflow.py`、`tools/augment.py`：catalog / text / import-captions / images / filter / caption-images / merge 七个命令，原子保存、相同参数续接、身份绑定校验。完整 catalog 配合少量候选图时仅筛选实际包含的源图。
- [x] 测试生成器完成每原图 20 候选→Top-10→生成图自身描述→合并→真实数据类加载；微型真实 CLIP 完成三类来源联合训练、保存/恢复与独立评估。清单变更会拒绝精确恢复；移走增强文件后独立评估仍可运行。
- [x] 全套 223 项检查通过（25.39 秒），日志 `.recovery/stage4_validation/pytest.txt`；README、阶段 4 说明和恢复计划已更新，无新快照。

已生成可供训练入口读取的真实缓存清单 `.recovery/stage4_validation/text_cached.json`：18,394 张训练图新增 36,788 条描述；111 张的描述重复/与人工描述相同，已跳过，明细在清单 `report` 中。合并原样本后 73,798 条训练描述、3,701 身份，val/test 各 1,000 图不变；记录见 `full_cache_load.json`。来源模型未知，标记 `legacy-unspecified`，不据此宣称缓存描述质量已验证。

`tools/validate_augmentation.py` 使用两身份、2 条人工 + 2 条缓存描述、官方 CLIP 与 NAC/A2DFP/CDFP 完成真实 CPU Adam 单步；损失有限，骨干/NAC/分类器均有有限非零梯度并更新，增强描述有效干净目标数为 0。记录 `real_cached_text.json`；该检查腐蚀率为 1，正式默认仍为 0.15。固定官方 CLIP 筛选入口也已运行，使用明确的图片处理测试样本，见 `clip_filter.json`，不把它们当作扩散训练数据。

统一训练清单：`{"version":1,"dataset":"RSTPReid","records":[...]}`。text 行指向原训练源图并有两条新描述；image 行有 `selected:true`、清单目录内相对图片路径和生成图自身两条描述。原始身份、源图和 provenance 保留。原始描述缓存可导入；禁止用原图描述冒充生成图描述。

真实 Qwen/扩散运行仍待服务凭据、本地 SD1.5 和训练后的 LoRA 权重；本机缺少 diffusers/transformers，未执行真实 API、真实扩散或全量生成。旧训练器依赖 deprecated LoRA API 的运行环境还未验证，不能把旧脚本直接视作已验证训练入口。下一次服务器验证先按 `docs/NCHNET_STAGE4_AUGMENTATION.md` 跑 1–2 张源图，再决定批量规模。

阶段 4 的本地代码收尾已完成；真实服务/扩散的质量、显存与耗时待资源具备后验证，不把流程检查当作论文实验完成。

## 阶段 5：已接通，完整实验待执行

用户确认服务器为 **Ubuntu + RTX 3090 24GB**；其它数据之后放项目 `data/`，不以当前未提供具体绝对路径作为代码工作的阻塞。默认布局、官方下载格式依据和全部命令见 `docs/NCHNET_STAGE5_EXPERIMENTS.md`。

- [x] 重新核对论文 §4.1–4.5、Table 5/6/7/8/12 与 Fig.6/7，精简说明和 paper_reference 数值见 `docs/NCHNET_STAGE5_PAPER_PROTOCOL.md`，不将论文数值写成实测结果。
- [x] `configs/experiments/paper.json`、`paths.example.json`：完整 SYNTH 源训练→三目标零训练直接迁移→三目标分别微调/测试；Table 5 六行、Table 8 扩充消融和四组敏感性范围。所有比较固定 identity/K=4，默认 seed=42/batch=64/30 epochs。
- [x] `tools/experiments.py`：plan/show/check/run，逐项执行，原子保存结果、严格完成依赖、epoch 续训、不覆盖完成项；源/CLIP 初始化文件移走后仍可精确续训。生成计划不启动计算。全部支持配置共 67 项任务，四个默认参数点复用相同配置评估。
- [x] `processor/baseline.py`/`test.py`：跨集严格加载源架构、直接检索不训练；微调始终重置身份分类器与 optimizer。checkpoint/结果记录初始化来源 SHA、源/目标数据集、协议、选模规则、权重/测试标注 SHA；runner 记录耗时、环境及 CUDA allocator 显存峰值（仅实际 CUDA 执行时）。
- [x] `tools/collect_results.py`：按 plan 汇总 JSON/CSV；未运行 pending，计划/实际配置分开，校验来源目录，重复及冲突保留，不用 test 取最高，不混入明确 smoke/subset 结果。
- [x] 按原作者公开格式核对 CUHK/ICFG/SYNTH；ICFG 兼容 `ICFG-PEDES.json` 和 `ICFG_PEDES.json`，加载与资源检查一致，冲突明确报错。
- [x] 新增 `docs/UBUNTU_COMMANDS.md`：Ubuntu 环境、资源/单步检查、训练/续训/评估、扩充、实验计划和后台运行命令速查；命令参数和任务 ID 已核对，本次仅改文档。
- [x] 全套 **260 passed / 30.48s**，日志 `.recovery/stage5_validation/pytest.txt`；`tools/validate_experiments.py` 完成微型真实 CLIP 计划→源预训练→NAC/A2DFP/CDFP+双扩充微调→独立测试→汇总，临时权重已清理，报告 `pipeline_smoke.json`。

补充实验约定（论文未唯一说明）：SYNTH 自动描述没有对齐干净目标，主源预训练默认 A2DFP+CDFP、NAC off，使用完整发布标注且不额外扩充源；目标 full 新增 NAC 并使用双扩充。核心消融另用同一 baseline-only SYNTH checkpoint 初始化，避免静默删除已训练 full 分支；细节随 plan 保存。

本机仅实际具备 RSTPReid 原始数据、BPE、官方 CLIP 和文本增强缓存；默认目录及缺失资源见 `resources.json`。这不妨碍配置/代码准备，但完整实验还未执行，正式汇总当前全部 pending。服务器没有连接或运行，未验证 CUDA/AMP/批量显存；没有创建新快照。

尚待后续：阶段 4 真实生成；3090 CUDA 单步与正式 batch=64 试跑；全量源/三个目标正式训练和评估；Table 9 的特殊 NAC 分支、Table 12 static average、其它生成质量/复杂度分析尚未实现或未实测，不能把核心配置可运行等同整篇实验完成。

## 继续执行

- 先检查上述文件及本文件最新状态，继续阶段 5 的服务器验证/正式实验待办；不要重复建快照或重做阶段 1–4。
- PowerShell 测试前：`$env:TEMP=(Resolve-Path .recovery/tmp).Path; $env:TMP=$env:TEMP`。
- 阶段 3 已通过，不必重复实现或无故重跑全部测试。后续按改动运行必要检查，不启动完整训练，不将小样例指标称为论文结果。
- 如需重现 NAC 真实单步：`.venv/Scripts/python.exe tools/validate_stage2.py --variant joint --use_nac --nac_corruption_rate 1 --clip_weights .recovery/model_cache/ViT-B-16.pt`。
- 阶段 4 混合批次：`.venv/Scripts/python.exe tools/validate_augmentation.py --augmentation_manifest .recovery/stage4_validation/text_small.json --clip_weights .recovery/model_cache/ViT-B-16.pt`。仅用已有全部文本缓存训练时，指定 `--augmentation_manifest .recovery/stage4_validation/text_cached.json --use_llm_augment`。
- 阶段 5 CPU 全流程：`.venv/Scripts/python.exe tools/validate_experiments.py`；服务器命令先读 `docs/NCHNET_STAGE5_EXPERIMENTS.md`。用户已确认按 data 默认布局，不必再追问数据集绝对路径。
- 用户额度恢复后可直接要求“读取 docs/RECOVERY_PROGRESS.md，从未完成的阶段继续”。
