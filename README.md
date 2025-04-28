## 项目结构

项目的主要组件如下：

1. **数据处理**（`clipec/data/dataset.py`）：
   - 读取和预处理.nrrd格式的CT图像
   - 从Excel表格中读取对应的标签信息
   - 提供训练、验证和测试的数据加载器

2. **模型**：
   - 图像编码器（ResNet50）(`clipec/models/image_encoders/resnet.py`)
   - 文本编码器（简单嵌入或Transformer）(`clipec/models/text_encoders/transformer.py`)
   - CLIP模型（`clipec/models/clip_model.py`）：结合图像和文本编码器，计算相似度矩阵

3. **训练**（`clipec/training/trainer.py`）：
   - 包含训练循环、验证和测试功能
   - 支持模型保存和加载
   - 提供预测和评估功能

4. **工具函数**（`clipec/utils/helpers.py`）：
   - 随机种子设置
   - 模型保存和加载
   - 结果可视化等

5. **脚本**：
   - 训练脚本（`scripts/train.py`）
   - 评估脚本（`scripts/evaluate.py`）
   - 演示脚本（`scripts/demo.py`）
   - 一键运行脚本（`run.py`）

## 如何运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 开启虚拟环境
查看环境名
conda env list
开启虚拟环境
conda activate clip-ecr

### 训练模型

```bash
python run.py --mode train --image_dir C:/Users/vipuser/Downloads/image --label_file C:/Users/vipuser/Downloads/table_info.xlsx --batch_size 32 --epochs 10
```

```bash
python run.py --mode train --image_dir F:/1_ML/data/image --label_file F:/1_ML/data/table_info.xlsx --batch_size 32 --epochs 10
```

或直接使用默认参数：

```bash
python run.py
```

### 评估模型

```bash
python run.py --mode evaluate --model_path CLIP-ECR/checkpoints/best_model.pt
```

### 对单张图像进行预测

```bash
python run.py --mode demo --model_path CLIP-ECR/checkpoints/best_model.pt --image_path F:/1_ML/data/image/14002076.nrrd
```

## 注意事项

1. 该项目使用了ResNet50作为图像编码器，并利用预训练权重。这在您的计算机上应该能够高效运行。

2. 数据加载和预处理部分被设计为从表格中筛选出与图像匹配的条目，所以即使有些图像没有对应的标签信息也不会出错。

3. 简单文本编码器（SimpleTextEncoder）使用了预定义的词汇表，更适合您的标签分类任务，且计算开销较小。

4. 默认batch_size设置为8，您可以根据显存大小进行调整。

5. 整个项目设计得尽可能简单，您可以根据需要逐步迭代和扩展功能。

这个项目提供了一个基础框架，您可以在此基础上进行进一步的改进和定制，如增加更复杂的模型架构、添加更多的数据增强方法、优化训练策略等。
