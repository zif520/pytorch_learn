#!/usr/bin/env python3
"""
Stable Diffusion 1.5 训练脚本 - 初学者版本
适用于 macOS 系统
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import argparse
from pathlib import Path
import torchvision
from torchvision import datasets, transforms

class SimpleDataset(Dataset):
    """简单的数据集类，用于加载图片和文本描述"""
    
    def __init__(self, data_dir, tokenizer, image_size=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        
        # 读取 captions.txt 文件
        captions_file = self.data_dir / "captions.txt"
        if not captions_file.exists():
            raise FileNotFoundError(f"找不到 captions.txt 文件: {captions_file}")
        
        self.samples = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    img_name, caption = line.split('\t', 1)
                    img_path = self.data_dir / img_name
                    if img_path.exists():
                        self.samples.append((img_path, caption))
        
        print(f"加载了 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        
        # 加载和预处理图片
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # 编码文本
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokenized.input_ids[0]
        }

class CIFAR10CaptionDataset(Dataset):
    """CIFAR-10数据集包装，生成简单caption"""
    def __init__(self, train=True, image_size=32):
        self.dataset = datasets.CIFAR10(root="./data", train=train, download=True)
        self.image_size = image_size
        self.labels = self.dataset.targets
        self.label_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        caption = self.label_names[label]
        return image, caption

def load_models(model_id="runwayml/stable-diffusion-v1-5"):
    """加载预训练模型"""
    print(f"正在加载模型: {model_id}")
    
    # 加载 tokenizer 和 text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    
    # 加载 UNet
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    
    # 加载 scheduler
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    return tokenizer, text_encoder, unet, vae, scheduler

def train_step(batch, unet, text_encoder, vae, scheduler, optimizer, device):
    """单步训练"""
    # 将数据移到设备上
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    
    # 冻结 text encoder 和 vae
    with torch.no_grad():
        # 编码文本
        text_embeddings = text_encoder(input_ids)[0]
        
        # 编码图片
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215  # 缩放因子
    
    # 添加噪声
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=latents.device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    # 预测噪声
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion 1.5 训练脚本")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录路径（如不指定则用CIFAR-10）")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--image_size", type=int, default=32, help="图片尺寸（CIFAR-10为32）")
    args = parser.parse_args()
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 CUDA")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载模型
    tokenizer, text_encoder, unet, vae, scheduler = load_models()
    
    # 将模型移到设备上
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    vae = vae.to(device)
    
    # 冻结 text encoder 和 vae
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # 创建数据集和数据加载器
    if args.data_dir is None:
        print("使用CIFAR-10数据集")
        dataset = CIFAR10CaptionDataset(train=True, image_size=args.image_size)
        # 用CLIP tokenizer编码caption
        class CIFAR10SDWrapper(Dataset):
            def __init__(self, cifar_dataset, tokenizer):
                self.cifar_dataset = cifar_dataset
                self.tokenizer = tokenizer
            def __len__(self):
                return len(self.cifar_dataset)
            def __getitem__(self, idx):
                image, caption = self.cifar_dataset[idx]
                tokenized = self.tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )
                return {
                    "pixel_values": image,
                    "input_ids": tokenized.input_ids[0]
                }
        dataset = CIFAR10SDWrapper(dataset, tokenizer)
    else:
        print(f"使用自定义数据集: {args.data_dir}")
        dataset = SimpleDataset(args.data_dir, tokenizer, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    
    # 训练循环
    print(f"开始训练，共 {args.epochs} 轮")
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n第 {epoch + 1}/{args.epochs} 轮训练")
        
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            loss = train_step(batch, unet, text_encoder, vae, scheduler, optimizer, device)
            epoch_losses.append(loss)
            global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            
            # 定期保存模型
            if global_step % args.save_steps == 0:
                save_path = output_dir / f"unet_step_{global_step}"
                unet.save_pretrained(save_path)
                print(f"\n模型已保存到: {save_path}")
        
        # 每轮结束后保存
        avg_loss = np.mean(epoch_losses)
        print(f"第 {epoch + 1} 轮平均损失: {avg_loss:.4f}")
        
        save_path = output_dir / f"unet_epoch_{epoch + 1}"
        unet.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")
    
    # 保存最终模型
    final_save_path = output_dir / "unet_final"
    unet.save_pretrained(final_save_path)
    print(f"\n训练完成！最终模型已保存到: {final_save_path}")

if __name__ == "__main__":
    main() 