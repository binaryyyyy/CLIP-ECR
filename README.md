# CLIP-ECR: 对食管癌CT图像进行分类的CLIP模型

这个项目使用CLIP (Contrastive Language-Image Pre-training) 模型对食管癌CT图像进行分类和标签生成。

## 项目结构

```
CLIP-ECR/
├── clipec/                  # 主要代码目录
│   ├── models/              # 模型定义
│   │   ├── image_encoders/  # 图像编码器
│   │   └── text_encoders/   # 文本编码器
│   ├── data/                # 数据处理
│   ├── training/            # 训练代码
│   └── utils/               # 工具函数
├── config/                  # 配置文件
├── scripts/                 # 脚本文件
└── requirements.txt         # 项目依赖
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：确保CT图像（.nrrd格式）和对应的标签数据（Excel表格）已经放置在正确的位置。
2. 运行训练脚本：

```bash
python scripts/train.py
```

3. 查看结果：

```bash
python scripts/evaluate.py
``` 