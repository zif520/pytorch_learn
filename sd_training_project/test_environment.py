#!/usr/bin/env python3
"""
环境测试脚本 - 检查所有依赖是否正确安装
"""

import sys
import torch

def test_imports():
    """测试所有必要的包是否能正确导入"""
    print("🔍 测试包导入...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch 导入失败: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision 导入失败: {e}")
        return False
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ Diffusers 导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers 导入失败: {e}")
        return False
    
    try:
        import accelerate
        print(f"✅ Accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"❌ Accelerate 导入失败: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets 导入失败: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow (PIL)")
    except ImportError as e:
        print(f"❌ Pillow 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy 导入失败: {e}")
        return False
    
    try:
        import tqdm
        print(f"✅ TQDM: {tqdm.__version__}")
    except ImportError as e:
        print(f"❌ TQDM 导入失败: {e}")
        return False
    
    return True

def test_device():
    """测试可用的计算设备"""
    print("\n🔍 测试计算设备...")
    
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) 可用")
        device = torch.device("mps")
        try:
            # 测试 MPS 是否真的可用
            test_tensor = torch.randn(2, 2).to(device)
            print("✅ MPS 设备测试通过")
        except Exception as e:
            print(f"❌ MPS 设备测试失败: {e}")
            device = torch.device("cpu")
            print("⚠️  回退到 CPU")
    elif torch.cuda.is_available():
        print("✅ CUDA 可用")
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  仅 CPU 可用")
        device = torch.device("cpu")
    
    return device

def test_model_loading():
    """测试模型加载（不下载，只测试连接）"""
    print("\n🔍 测试模型加载...")
    
    try:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        print("✅ CLIP Tokenizer 加载成功")
    except Exception as e:
        print(f"❌ CLIP Tokenizer 加载失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 Stable Diffusion 训练环境测试")
    print("=" * 50)
    
    # 测试 Python 版本
    print(f"Python 版本: {sys.version}")
    
    # 测试包导入
    if not test_imports():
        print("\n❌ 包导入测试失败，请检查依赖安装")
        return False
    
    # 测试设备
    device = test_device()
    
    # 测试模型加载
    if not test_model_loading():
        print("\n❌ 模型加载测试失败，请检查网络连接")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 环境测试完成！")
    print(f"推荐设备: {device}")
    print("\n📝 下一步:")
    print("1. 准备训练数据")
    print("2. 编辑 data/captions.txt")
    print("3. 运行 python train_sd.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 