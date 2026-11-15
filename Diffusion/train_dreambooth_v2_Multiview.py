import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse


class PedestrianDataset(Dataset):
    """行人图片数据集"""

    def __init__(self, image_dir, width=256, height=512):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
        self.width = width
        self.height = height
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return {"pixel_values": img}


class PriorDataset(Dataset):
    """先验保留数据集 - 通用行人图像"""

    def __init__(self, image_dir, width=256, height=512):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
        self.width = width
        self.height = height
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx % len(self.image_paths)]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return {"pixel_values": img}


def train_dreambooth(
        pedestrian_id,
        image_dir,
        prior_dataloader,
        output_dir,
        pretrained_model="runwayml/stable-diffusion-v1-5",
        width=256,
        height=512,
        train_batch_size=1,
        num_train_epochs=100,
        learning_rate=5e-6,
        max_train_steps=400,
        prior_loss_weight=1.0,
):
    """训练单个行人的 Dreambooth 模型，带先验保留损失"""

    # 创建输出目录
    model_output_dir = os.path.join(output_dir, f"model_{pedestrian_id}")
    os.makedirs(model_output_dir, exist_ok=True)

    # 初始化加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )

    # 加载预训练模型
    print(f"加载预训练模型: {pretrained_model}")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")

    # 冻结 VAE 和 text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 只训练 UNet
    unet.train()

    # 创建目标行人数据集和数据加载器
    dataset = PedestrianDataset(image_dir, width=256, height=512)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    # 优化器
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # 噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

    # 准备训练
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    if prior_dataloader is not None:
        prior_dataloader = accelerator.prepare(prior_dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # 生成 prompt
    prompt = f"a photo of sks person {pedestrian_id}"
    prior_prompt = "a photo of person"

    # 创建循环迭代器用于先验数据
    prior_iterator = None
    if prior_dataloader is not None:
        prior_iterator = iter(prior_dataloader)

    # 训练循环
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc=f"Training {pedestrian_id}")

    for epoch in range(num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # ==================== 目标行人损失 ====================
                # 转换图片到潜在空间
                latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 采样噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # 随机时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 获取文本嵌入
                text_inputs = tokenizer(
                    [prompt] * bsz,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

                # 预测噪声
                model_pred = unet(noisy_latents, timesteps, text_embeddings).sample

                # 计算目标行人损失
                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), noise.float(), reduction="mean"
                )

                # ==================== 先验保留损失 ====================
                if prior_dataloader is not None and prior_loss_weight > 0:
                    try:
                        prior_batch = next(prior_iterator)
                    except StopIteration:
                        # 重置迭代器
                        prior_iterator = iter(prior_dataloader)
                        prior_batch = next(prior_iterator)

                    # 转换先验图片到潜在空间
                    prior_latents = vae.encode(
                        prior_batch["pixel_values"].to(vae.dtype)
                    ).latent_dist.sample()
                    prior_latents = prior_latents * vae.config.scaling_factor

                    # 采样噪声
                    prior_noise = torch.randn_like(prior_latents)
                    prior_bsz = prior_latents.shape[0]

                    # 随机时间步
                    prior_timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (prior_bsz,), device=prior_latents.device
                    )
                    prior_timesteps = prior_timesteps.long()

                    # 添加噪声
                    prior_noisy_latents = noise_scheduler.add_noise(
                        prior_latents, prior_noise, prior_timesteps
                    )

                    # 获取先验文本嵌入
                    prior_text_inputs = tokenizer(
                        [prior_prompt] * prior_bsz,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    prior_text_embeddings = text_encoder(
                        prior_text_inputs.input_ids.to(accelerator.device)
                    )[0]

                    # 预测噪声（先验）
                    prior_model_pred = unet(
                        prior_noisy_latents, prior_timesteps, prior_text_embeddings
                    ).sample

                    # 计算先验保留损失
                    prior_loss = torch.nn.functional.mse_loss(
                        prior_model_pred.float(), prior_noise.float(), reduction="mean"
                    )

                    # 合并损失
                    loss = loss + prior_loss_weight * prior_loss

                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    # 保存模型
    print(f"保存模型到: {model_output_dir}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(model_output_dir)

    return model_output_dir


def generate_images(
        pedestrian_id,
        model_dir,
        output_dir,
        num_images=20,
        width=256,
        height=512,
        guidance_scale=7.5,
        num_inference_steps=50,
):
    """使用训练好的模型生成新图片"""

    print(f"加载模型: {model_dir}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")

    # 创建输出目录
    person_output_dir = os.path.join(output_dir, pedestrian_id)
    os.makedirs(person_output_dir, exist_ok=True)

    # 生成图片
    prompt = f"a photo of sks person {pedestrian_id}"

    print(f"生成 {num_images} 张图片...")
    for i in tqdm(range(num_images), desc=f"Generating {pedestrian_id}"):
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]

        # 保存图片
        image.save(os.path.join(person_output_dir, f"{pedestrian_id}_generated_{i:04d}.png"))

    print(f"图片已保存到: {person_output_dir}")

    # 清理显存
    del pipeline
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Dreambooth 行人图片扩容（带先验保留损失，高效批处理）")
    parser.add_argument("--data_dir", type=str, default="Data/Organize_Pedestrians-all", help="目标行人数据目录")
    parser.add_argument("--prior_data_dir", type=str, default="Data/Prior_Images",
                        help="先验保留数据目录（通用行人图像）")
    parser.add_argument("--output_dir", type=str, default="Output", help="生成图片输出目录")
    parser.add_argument("--model_output_dir", type=str, default="Models", help="模型保存目录")
    parser.add_argument("--start_id", type=int, default='4100', help="开始的行人ID")
    parser.add_argument("--end_id", type=int, default='4101', help="结束的行人ID")
    parser.add_argument("--num_generated_images", type=int, default=20, help="每个行人生成的图片数量")
    parser.add_argument("--pretrained_model", type=str, default="/workspace/GuangjinOuyang/DiffusionStable/stable-diffusion",
                        help="预训练模型")
    parser.add_argument("--max_train_steps", type=int, default=450, help="最大训练步数")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="先验保留损失权重")
    parser.add_argument("--train_batch_size", type=int, default=1, help="训练批大小")
    parser.add_argument("--prior_batch_size", type=int, default=1, help="先验保留批大小")
    parser.add_argument("--width", type=int, default=256, help="生成图片宽度")
    parser.add_argument("--height", type=int, default=512, help="生成图片高度")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练,只生成图片")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_output_dir, exist_ok=True)

    # ==================== 一次性加载先验保留数据 ====================
    prior_dataloader = None
    if args.prior_data_dir and os.path.exists(args.prior_data_dir):
        print(f"\n{'=' * 50}")
        print("加载先验保留数据集（仅一次）...")
        print(f"{'=' * 50}")
        prior_dataset = PriorDataset(args.prior_data_dir, width=args.width, height=args.height)
        prior_dataloader = DataLoader(
            prior_dataset,
            batch_size=args.prior_batch_size,
            shuffle=True,
            num_workers=4  # 使用多进程加速数据加载
        )
        print(f"✓ 加载完成: {len(prior_dataset)} 张先验图像")
    else:
        print("⚠ 警告: 未提供先验保留图像目录，生成的图像可能会过度拟合")

    # ==================== 批量训练所有行人 ====================
    for person_id in range(args.start_id, args.end_id):
        pedestrian_id = f"{person_id:04d}"
        image_dir = os.path.join(args.data_dir, pedestrian_id)

        if not os.path.exists(image_dir):
            print(f"目录不存在,跳过: {image_dir}")
            continue

        print(f"\n{'=' * 50}")
        print(f"处理行人 ID: {pedestrian_id}")
        print(f"{'=' * 50}")

        model_dir = os.path.join(args.model_output_dir, f"model_{pedestrian_id}")

        # 训练模型
        if not args.skip_training or not os.path.exists(model_dir):
            try:
                model_dir = train_dreambooth(
                    pedestrian_id=pedestrian_id,
                    image_dir=image_dir,
                    prior_dataloader=prior_dataloader,  # 共享的先验数据加载器
                    output_dir=args.model_output_dir,
                    pretrained_model=args.pretrained_model,
                    width=args.width,
                    height=args.height,
                    train_batch_size=args.train_batch_size,
                    max_train_steps=args.max_train_steps,
                    learning_rate=args.learning_rate,
                    prior_loss_weight=args.prior_loss_weight,
                )
            except Exception as e:
                print(f"✗ 训练失败: {e}")
                continue

        # 生成新图片
        try:
            generate_images(
                pedestrian_id=pedestrian_id,
                model_dir=model_dir,
                output_dir=args.output_dir,
                num_images=args.num_generated_images,
                width=args.width,
                height=args.height,
            )
        except Exception as e:
            print(f"✗ 生成图片失败: {e}")
            continue

    print(f"\n{'=' * 50}")
    print("✓ 所有行人处理完成！")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
