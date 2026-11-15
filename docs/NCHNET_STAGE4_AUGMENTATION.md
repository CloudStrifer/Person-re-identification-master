# 阶段 4：离线图文扩充

统一入口为 `tools/augment.py`，训练读取准备好的增强清单；训练时不调用 Qwen 或扩散模型。先从少量训练源图开始，检查后再扩大范围。正式扩散生成放在 Ubuntu 服务器。

## 1. 导出原训练源图

以下命令在项目根目录执行；Windows 可将 `python` 换为 `.venv/Scripts/python.exe`。所有产物放在同一目录下，便于图片路径随清单一起搬到服务器。

```bash
python tools/augment.py catalog --dataset RSTPReid --root_dir data --output augmentation_runs/rstp/catalog.json
```

目录仅包含原 train 的身份、源图路径和人工描述，排除 val/test。它记录当前机器的绝对原图路径，换机器后重新导出；训练清单中的生成图片使用相对路径。

## 2. 准备新增文本

已有的两条描述缓存可以直接导入，不需要重新调用服务：

```bash
python tools/augment.py import-captions --catalog augmentation_runs/rstp/catalog.json --captions captions/pedestrian_captions_results.json --output augmentation_runs/rstp/text.json --limit 2
```

`--limit` 限制源图数量，默认不限量；`--source_id` 可以重复指定确切的原标注路径，先缩小范围，再按 catalog 顺序取前 N 张。示例中的 2 是试跑范围，调用真实服务或扩散生成时也应显式保留小范围参数。默认跳过缺失、重复、与原人工描述相同或错误内容，并在输出 `report` 中记录源图及原因；`--strict` 改为遇错退出。不会凭文件名猜测身份，也不会导入非训练图。已有缓存的模型信息不完整，来源记为 `legacy-unspecified`。

重新生成文本时，先提取视觉属性，再从 30 个条件模板选出两条描述。属性缓存格式示例（键须与 catalog 完全一致）：

```json
{"0000_c14_0031.jpg": {"upper_clothing": "red coat", "lower_clothing": "black trousers"}}
```

```bash
python tools/augment.py text --catalog augmentation_runs/rstp/catalog.json --attributes attributes.json --output augmentation_runs/rstp/text_new.json --limit 2 --seed 42
```

也可显式 `--provider qwen --model MODEL_NAME`，凭据来自 `DASHSCOPE_API_KEY`，通过 `--base_url` 指定实际区域的兼容端点。模型名和服务可用性应以账户为准。仅明确使用该 provider 时才请求服务，异常文本不会写成描述；本轮没有执行真实 API 请求。

## 3. 生成候选图并逐原图筛选

需要本地 SD1.5 pipeline 目录，以及旧训练器保存的 `unet_lora.pt`、`embeddings.pt`、`metadata.json`。生成环境另需 diffusers/transformers；本机未安装，也未验证其版本组合。新入口可读取旧 adapter，将 LoRA 增量合入 UNet，并按 token 字符串载入学习到的词嵌入，不依赖旧的 attention processor 推理接口。

```bash
python tools/augment.py images --catalog augmentation_runs/rstp/catalog.json --local_model_dir /models/sd15 --checkpoint_dir /models/reid_lora/final_model --output augmentation_runs/rstp/candidates.json --num_candidates 20 --limit 2 --seed 42 --device cuda --dtype float16
python tools/augment.py filter --catalog augmentation_runs/rstp/catalog.json --candidates augmentation_runs/rstp/candidates.json --clip_weights /models/ViT-B-16.pt --output augmentation_runs/rstp/selected.json --keep 10 --device cuda
```

默认逐张生成，保存 seed、prompt、adapter 来源与图片 sidecar。身份使用训练时的 placeholder token 和 random ID。原 PID 与 metadata subject key 须精确匹配；若原 PID 为 `10` 而训练目录名为 `0010`，用 `--subject_map subject_map.json` 显式提供 `{"10":"0010"}`。

筛选用固定 CLIP 图像编码器，分别计算每张原图与其候选图的余弦，保留 Top-10，不用身份均值代替原图。只处理候选清单中实际包含的源图，因此小范围生成可以使用完整 catalog。某张源图的候选少于 keep 时会报错，应完成生成或显式减小 `--keep`。仅筛选的图片还不能进入训练，必须补自身描述。

## 4. 为保留的生成图标注，再合并

```bash
python tools/augment.py caption-images --catalog augmentation_runs/rstp/catalog.json --selected augmentation_runs/rstp/selected.json --attributes generated_attributes.json --output augmentation_runs/rstp/image_captions.json --seed 42
python tools/augment.py merge --catalog augmentation_runs/rstp/catalog.json --inputs augmentation_runs/rstp/text.json augmentation_runs/rstp/image_captions.json --output augmentation_runs/rstp/train_manifest.json
```

`generated_attributes.json` 的键是 `selected.json` 中生成图的 `image_path`，属性必须来自该生成图；也可显式使用 Qwen provider 读取生成图。禁止复制原图描述作为生成图标注。所有清单放在生成图片的共同父目录中，合并会重写相对路径，不复制或移动原图。

## 5. 训练接入及中断恢复

```bash
python train.py --root_dir data --clip_weights /models/ViT-B-16.pt --augmentation_manifest augmentation_runs/rstp/train_manifest.json --use_llm_augment --use_generated --use_nac --use_a2dfp --use_cdfp --device cuda --output_dir logs/rstp_augmented
```

两个增强开关可独立启用。原训练样本保留，新增样本继承原身份；val/test 不读取增强清单。LLM 和生成图描述没有对齐的干净目标，不构造伪 NAC 监督，仍参与检索损失和训练期纠正。

生成命令使用相同输入、参数和输出文件再次执行即可续接，已完成图文会验证并复用；输入或参数改变时应使用新的输出文件。API 描述逐源图保存，图片逐候选保存，采用原子写入。已有描述缓存导入可直接重算。清单异常不会被静默当成完成。

训练按 epoch 恢复：`python train.py --resume logs/rstp_augmented/last.pth`。checkpoint 保存增强清单 SHA-256 和使用开关，内容改变会拒绝精确续训；迁移机器时可显式指定新的 `--root_dir`、`--augmentation_manifest` 路径，清单内容须相同。独立 `test.py` 不需要增强文件。

## 本轮验证边界

模块与临时全流程检查覆盖属性/模板、旧权重矩阵与词嵌入加载、20 候选/Top-10、生成图自身标注、清单合并、NAC 目标边界、联合训练/续训/独立评估。图片流水线使用明确的测试生成器验证功能，没有运行真实扩散。

全套 223 项检查通过。已有缓存已导出至 `.recovery/stage4_validation/text_cached.json`，包含 18,394 张训练图的 36,788 条新增描述；111 张重复描述被跳过，详情在清单 `report` 中。合并后共有 73,798 条训练描述，保持 3,701 个身份及原 val/test 划分。仅使用这份文本扩充时，训练参数添加 `--augmentation_manifest .recovery/stage4_validation/text_cached.json --use_llm_augment`，不需要图像生成资源。

真实 RSTPReid 描述缓存的小样例已通过官方 CLIP 混合批次检查，结果见 `.recovery/stage4_validation/`。可复查：

```bash
python tools/validate_augmentation.py --augmentation_manifest .recovery/stage4_validation/text_small.json --clip_weights .recovery/model_cache/ViT-B-16.pt
```

真实 Qwen、SD1.5/训练后 LoRA 的生成效果和显存仍待服务器资源验证；旧 DreamBooth/LoRA 训练脚本的 deprecated API 环境也未验证。本轮不声明生成质量、完整训练或论文指标。最新状态见 `RECOVERY_PROGRESS.md`。
