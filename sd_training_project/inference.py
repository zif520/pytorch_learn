#!/usr/bin/env python3
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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
        feature_extractor=None,
        requires_safety_checker=False
    )
    
    return pipe

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion 1.5 推理脚本")
    parser.add_argument("--model_path", type=str, default="output/unet_final", help="训练好的模型路径")
    parser.add_argument("--prompt", type=str, default=None, help="生成提示词")
    parser.add_argument("--prompt_file", type=str, default=None, help="批量生成时的提示词文件（每行一个prompt）")
    parser.add_argument("--output_path", type=str, default="generated_image.png", help="输出图片路径或前缀")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导强度")
    parser.add_argument("--height", type=int, default=512, help="生成图片高度")
    parser.add_argument("--width", type=int, default=512, help="生成图片宽度")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
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
    try:
        pipe = load_trained_model(args.model_path)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 设置种子
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    else:
        generator = None

    # 获取 prompt 列表
    prompts = []
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"无法读取 prompt 文件: {e}")
            return
    elif args.prompt:
        prompts = [args.prompt]
    else:
        print("请通过 --prompt 或 --prompt_file 指定生成提示词！")
        return

    # 批量生成
    for idx, prompt in enumerate(prompts):
        try:
            print(f"生成图片 {idx+1}/{len(prompts)}，提示词: {prompt}")
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator
            ).images[0]
            # 输出路径
            if len(prompts) == 1:
                out_path = args.output_path
            else:
                suffix = Path(args.output_path).suffix or ".png"
                prefix = args.output_path.replace(suffix, "")
                out_path = f"{prefix}_{idx+1}{suffix}"
            image.save(out_path)
            print(f"图片已保存到: {out_path}")
        except Exception as e:
            print(f"生成失败: {e}")

if __name__ == "__main__":
    main() 