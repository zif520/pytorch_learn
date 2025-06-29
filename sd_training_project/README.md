# Stable Diffusion 1.5 训练项目 - 初学者版本

这是一个完整的 Stable Diffusion 1.5 训练项目，专为初学者设计，支持 macOS 系统。

## 📁 项目结构

```
sd_training_project/
├── train_sd.py          # 主训练脚本
├── inference.py         # 推理脚本
├── requirements.txt     # 依赖包列表
├── data/               # 数据目录
│   └── captions.txt    # 图片描述文件
├── output/             # 输出目录（训练时自动创建）
└── README.md          # 说明文档
```

## 🚀 快速开始

### 1. 环境准备

首先激活你的 conda 环境：

```bash
conda activate sd_training
```

如果没有环境，创建一个：

```bash
conda create -n sd_training python=3.10 -y
conda activate sd_training
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据

1. 将你的训练图片放在 `data/` 目录下
2. 编辑 `data/captions.txt` 文件，按照以下格式添加描述：

```
image1.jpg	a beautiful sunset over the ocean
image2.jpg	a cute cat sitting on a windowsill
image3.jpg	a modern city skyline at night
```

**格式说明：**
- 图片文件名 + 制表符（Tab）+ 描述文本
- 每行一个图片-描述对
- 图片格式：jpg, jpeg, png, bmp
- 建议图片尺寸：512x512 或更大

### 4. 开始训练

```bash
python train_sd.py --data_dir data --epochs 10 --batch_size 1
```

**参数说明：**
- `--data_dir`: 数据目录路径（默认：data）
- `--output_dir`: 输出目录（默认：output）
- `--epochs`: 训练轮数（默认：10）
- `--batch_size`: 批次大小（默认：1）
- `--learning_rate`: 学习率（默认：1e-5）
- `--save_steps`: 保存步数（默认：100）

### 5. 测试模型

训练完成后，使用推理脚本测试：

```bash
python inference.py --model_path output/unet_final --prompt "a beautiful landscape" --output_path test_image.png
```

**参数说明：**
- `--model_path`: 训练好的模型路径
- `--prompt`: 生成提示词
- `--output_path`: 输出图片路径
- `--num_inference_steps`: 推理步数（默认：50）
- `--guidance_scale`: 引导强度（默认：7.5）

## 🔧 系统要求

- **操作系统**: macOS 10.15+
- **Python**: 3.8+
- **内存**: 至少 8GB RAM
- **存储**: 至少 10GB 可用空间

## 💡 使用建议

### 数据集准备
- **图片数量**: 建议 10-50 张图片开始
- **图片质量**: 清晰、高质量、主题一致
- **描述质量**: 详细、准确、包含关键特征
- **图片尺寸**: 统一为 512x512 或更大

### 训练参数
- **学习率**: 1e-5 适合大多数情况
- **批次大小**: 1-2（受内存限制）
- **训练轮数**: 10-50 轮（根据数据集大小调整）

### 性能优化
- **Apple Silicon Mac**: 自动使用 MPS 加速
- **内存不足**: 减小批次大小或图片尺寸
- **训练速度**: 可以先用小数据集测试

## 🐛 常见问题

### 1. 内存不足
```
解决方案：
- 减小 batch_size（如：--batch_size 1）
- 减小图片尺寸（修改 train_sd.py 中的 image_size）
- 关闭其他应用程序释放内存
```

### 2. 模型下载失败
```
解决方案：
- 确保网络连接正常
- 使用 VPN 或代理
- 手动下载模型文件
```

### 3. 训练效果不好
```
解决方案：
- 检查数据质量
- 调整学习率
- 增加训练轮数
- 优化图片描述
```

## 📚 进阶使用

### 自定义模型
你可以修改 `train_sd.py` 中的 `model_id` 来使用其他预训练模型：

```python
# 使用其他模型
model_id = "CompVis/stable-diffusion-v1-4"  # SD 1.4
model_id = "stabilityai/stable-diffusion-2-1"  # SD 2.1
```

### 保存完整模型
训练完成后，你可以将训练好的 UNet 与原始模型的其他组件组合，创建完整的 Stable Diffusion pipeline。

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如果遇到问题，请：
1. 查看常见问题部分
2. 检查错误日志
3. 提交 Issue 描述问题

---

**注意**: 这是一个教学项目，适合学习和实验。生产环境使用请参考官方文档和最佳实践。 