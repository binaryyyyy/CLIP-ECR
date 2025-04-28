# CLIP-ECR: 多切片食管癌CT图像分析模型

CLIP-ECR（Contrastive Language-Image Pre-training for Esophageal Cancer Radiography）是基于对比学习的食管癌CT影像分析模型，支持使用所有CT切片进行训练与推理。

## 功能特点

- 支持单一切片（中间切片）或全部CT切片的分析
- 基于CLIP模型架构，结合图像和文本信息进行对比学习
- 自动提取食管癌CT图像特征，与临床描述文本进行匹配
- 支持AJCC分期等多种临床指标的预测

## 安装

```bash
# 克隆代码库
git clone https://github.com/binaryyyyy/CLIP-ECR.git
cd CLIP-ECR

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练模型

训练模型支持两种切片选择模式：
- `middle`: 仅使用CT图像的中间切片（默认）
- `all`: 使用所有切片

```bash
# 使用中间切片进行训练（默认）
python scripts/train.py --image_dir path/to/images --label_file path/to/labels.xlsx

# 使用所有切片进行训练
python scripts/train.py --image_dir path/to/images --label_file path/to/labels.xlsx --slice_selection all
```

### 评估模型

评估模型时需要指定使用的切片选择模式与训练时保持一致：

```bash
# 评估使用中间切片训练的模型
python scripts/evaluate.py --image_dir path/to/images --label_file path/to/labels.xlsx --model_path path/to/checkpoint.pth

# 评估使用所有切片训练的模型
python scripts/evaluate.py --image_dir C:/Users/vipuser/Downloads/image --label_file C:/Users/vipuser/Downloads/table_info.xlsx --slice_selection all
```

### 单个病例演示

单个病例演示支持三种切片选择模式：
- `middle`: 使用CT图像的中间切片（默认）
- `specific`: 使用指定索引的切片
- `all`: 使用所有切片并汇总结果

```bash
# 使用中间切片
python scripts/demo.py --image_path path/to/image.nrrd --model_path path/to/checkpoint.pth

# 使用指定索引的切片
python scripts/demo.py --image_path path/to/image.nrrd --model_path path/to/checkpoint.pth --slice_selection specific --slice_idx 50

# 使用所有切片
python scripts/demo.py --image_path path/to/image.nrrd --model_path path/to/checkpoint.pth --slice_selection all
```

## 数据格式

- 图像数据：NRRD格式的CT图像
- 标签数据：Excel表格，包含患者ID和对应临床信息
  
## 模型架构

CLIP-ECR由以下组件构成：
- 图像编码器：基于ResNet50/ResNet152提取CT图像特征
- 文本编码器：基于BERT或自定义文本编码器提取临床描述特征
- 对比学习头：计算图像和文本特征之间的相似度

## 许可证

[MIT License](LICENSE)
