import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, Image as DatasetImage  # 修正导入
from transformers import CLIPTextModel, CLIPTokenizer, TrainingArguments, Trainer
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline
)
from diffusers.utils import make_image_grid
from accelerate import Accelerator
import torch.nn.functional as F


def load_caption_data(json_path, image_dir):
    """加载RSTPReid数据集的图片-描述配对数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)

    data = []
    for item in caption_data:
        img_path = os.path.join(image_dir, item["img_path"])
        if os.path.exists(img_path):
            core_captions = [cap.replace("futuristic", "").replace("glowing", "")
                             for cap in item["captions"] if cap.strip()]
            if core_captions:
                data.append({
                    "image_path": img_path,
                    "caption": " ".join(list(set(core_captions)))
                })
    print(f"已加载 {len(data)} 条有效图片-描述配对数据")
    return data


def create_finetune_dataset(data):
    """创建适配训练的Dataset"""

    def load_image(sample):
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((512, 512))
        return {"image": image, "text": sample["caption"]}  # 增加text字段用于训练

    dataset = Dataset.from_list(data)
    dataset = dataset.map(load_image, remove_columns=["image_path"])
    dataset = dataset.cast_column("image", DatasetImage())  # 修正Image类型使用
    return dataset


def finetune_diffusion_model(
        json_path,
        image_dir,
        finetuned_model_dir="./finetuned_rstpreid_model",
        base_model_name="runwayml/stable-diffusion-v1-5",  # 改用更稳定的v1-5版本
        train_batch_size=4,
        learning_rate=2e-6,
        num_train_epochs=3,
):
    """使用官方推荐的Trainer微调模型（替代StableDiffusionTrainer）"""
    os.makedirs(finetuned_model_dir, exist_ok=True)

    # 1. 加载数据
    data = load_caption_data(json_path, image_dir)
    dataset = create_finetune_dataset(data)
    train_dataset = dataset.train_test_split(test_size=0.05)["train"]

    # 2. 加载模型组件
    tokenizer = CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model_name, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler")

    # 冻结不需要微调的组件（仅微调UNet，节省显存）
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 3. 定义训练参数和损失函数
    training_args = TrainingArguments(
        output_dir=finetuned_model_dir,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_dir=f"{finetuned_model_dir}/logs",
        logging_steps=10,
        remove_unused_columns=False,
    )

    # 4. 定义训练循环
    def training_loop(model, train_dataloader, optimizer, lr_scheduler):
        model.train()
        for epoch in range(training_args.num_train_epochs):
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
            for batch in train_dataloader:
                # 处理输入
                images = batch["image"].to(dtype=torch.float32) / 127.5 - 1.0  # 归一化到[-1, 1]
                captions = tokenizer(
                    batch["text"], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
                ).input_ids.to(model.device)

                # 生成噪声
                noise = torch.randn(images.shape, device=model.device)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (images.shape[0],),
                                          device=model.device)

                # 加噪图像
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

                # 预测噪声
                model_output = model(noisy_images, timesteps, captions).sample
                loss = F.mse_loss(model_output, noise)  # MSE损失

                # 反向传播
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
            progress_bar.close()
            # 保存每轮模型
            model.save_pretrained(f"{finetuned_model_dir}/epoch_{epoch + 1}")

    # 5. 初始化优化器和调度器
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataset) // training_args.per_device_train_batch_size
    )

    # 6. 开始训练（使用Accelerator适配多GPU）
    accelerator = Accelerator()
    unet, optimizer, train_dataset, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataset, lr_scheduler
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True
    )

    print(f"开始在RSTPReid数据集上微调模型，输出目录：{finetuned_model_dir}")
    training_loop(unet, train_dataloader, optimizer, lr_scheduler)

    # 保存完整模型（用于生成）
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_name,
        unet=accelerator.unwrap_model(unet),
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer
    )
    pipeline.save_pretrained(finetuned_model_dir)
    print(f"模型微调完成，已保存至：{finetuned_model_dir}")
    return finetuned_model_dir


def generate_similar_images(
        json_path,
        image_dir,
        output_dir,
        finetuned_model_dir="./finetuned_rstpreid_model",
        num_poses_per_image=3,
        strength=0.4,
        guidance_scale=7.0
):
    """用微调后的模型生成类似图片（复用原有逻辑，确保兼容性）"""
    os.makedirs(output_dir, exist_ok=True)
    caption_data = load_caption_data(json_path, image_dir)

    pose_variations = [
        "facing opposite direction, same clothing style as original, RSTPReid scene background, natural lighting",
        "side view profile, realistic human posture, consistent with RSTPReid dataset style, no artistic distortion",
        "upper body shot (chest upwards), same scene elements as original, sharp details, real-life proportions"
    ]

    # 加载微调后的模型
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        finetuned_model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("使用GPU加速生成")

    pipe.safety_checker = lambda images, clip_input: (images, [False for _ in images])

    rstpreid_base = (
        "Photorealistic, consistent with RSTPReid dataset style, high resolution, "
        "natural lighting, no filters, no digital art, sharp details"
    )

    for idx, item in enumerate(tqdm(caption_data, desc="生成图片")):
        img_filename = os.path.basename(item["image_path"])
        img_path = item["image_path"]
        core_caption = item["caption"]

        try:
            with Image.open(img_path).convert("RGB") as image:
                original_size = image.size
                for pose_idx in range(min(num_poses_per_image, len(pose_variations))):
                    pose_prompt = (
                        f"{rstpreid_base}, {core_caption}, {pose_variations[pose_idx]}"
                    )

                    result = pipe(
                        prompt=pose_prompt,
                        image=image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=100,
                        width=original_size[0],
                        height=original_size[1],
                        negative_prompt="digital art, painting, cartoon, fantasy, low resolution"
                    )

                    generated_image = result.images[0]
                    base_name = os.path.splitext(img_filename)[0]
                    pose_label = pose_variations[pose_idx].split(",")[0].replace(" ", "_")
                    save_path = os.path.join(output_dir, f"{base_name}_style_{pose_label}_{pose_idx + 1}.jpg")
                    generated_image.save(save_path)

        except Exception as e:
            print(f"处理图片 {img_filename} 时出错: {str(e)}")
            continue

    print(f"生成完成，保存至：{output_dir}")


if __name__ == "__main__":
    JSON_FILE_PATH = "../data/RSTPReid/data_captions.json"
    IMAGE_DIRECTORY = "../data/RSTPReid/imgs"
    FINETUNED_MODEL_DIR = "./finetuned_rstpreid_model"
    OUTPUT_DIRECTORY = "./generated_rstpreid_similar"

    # 步骤1：微调模型（仅需运行一次）
    finetune_diffusion_model(
        json_path=JSON_FILE_PATH,
        image_dir=IMAGE_DIRECTORY,
        finetuned_model_dir=FINETUNED_MODEL_DIR,
        train_batch_size=4,
        num_train_epochs=3
    )

    # 步骤2：生成图片
    generate_similar_images(
        json_path=JSON_FILE_PATH,
        image_dir=IMAGE_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        finetuned_model_dir=FINETUNED_MODEL_DIR
    )
