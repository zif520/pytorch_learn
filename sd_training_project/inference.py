#!/usr/bin/env python3
"""
Stable Diffusion 1.5 推理脚本 - 测试训练好的模型
"""

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
from pathlib import Path

def load_trained_model(model_path, base_model_id="runwayml/stable-diffusion-v1-5"):
    """加载训练好的模型"""
    print(f"加载训练好的模型: {model_path}")
    
    # 加载基础模型组件
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    
    # 加载训练好的 UNet
    unet = UNet2DConditionModel.from_pretrained(model_path)
    
    # 创建 pipeline
    pipe = StableDiffusionPipeline(
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    return pipe

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion 1.5 推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape", help="生成提示词")
    parser.add_argument("--output_path", type=str, default="generated_image.png", help="输出图片路径")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导强度")
    
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
    
    # 加载模型
    pipe = load_trained_model(args.model_path)
    pipe = pipe.to(device)
    
    # 生成图片
    print(f"正在生成图片，提示词: {args.prompt}")
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    ).images[0]
    
    # 保存图片
    image.save(args.output_path)
    print(f"图片已保存到: {args.output_path}")

if __name__ == "__main__":
    main() 