"""默认配置文件"""

# 数据相关配置
DATA_CONFIG = {
    'image_dir': 'F:/1_ML/data/image',  # NRRD图像文件目录
    'label_file': 'F:/1_ML/data/table_info.xlsx',  # 标签Excel文件路径
    'batch_size': 8,  # 批量大小
    'num_workers': 2,  # 数据加载器工作进程数
    'train_ratio': 0.8,  # 训练集比例
    'val_ratio': 0.1,  # 验证集比例
}

# 模型相关配置
MODEL_CONFIG = {
    'image_encoder': 'resnet50',  # 图像编码器类型
    'text_encoder': 'simple',  # 文本编码器类型
    'embedding_dim': 512,  # 嵌入维度
    'temperature': 0.07,  # 温度系数
}

# 训练相关配置
TRAIN_CONFIG = {
    'lr': 1e-4,  # 学习率
    'weight_decay': 1e-4,  # 权重衰减
    'epochs': 50,  # 训练轮数
    'save_dir': 'CLIP-ECR/checkpoints',  # 模型保存目录
    'seed': 42,  # 随机种子
}

# 评估相关配置
EVAL_CONFIG = {
    'output_dir': 'CLIP-ECR/results',  # 结果保存目录
}

# 演示相关配置
DEMO_CONFIG = {
    'output_dir': 'CLIP-ECR/demo_results',  # 结果保存目录
}

# 标签相关配置
LABEL_CONFIG = {
    # AJCC分期
    'ajcc_stages': ['1', '2a', '2b', '3a', '3b', '4'],
    
    # T分期
    't_stages': ['T1', 'T2', 'T3', 'T4'],
    
    # N分期
    'n_stages': ['N0', 'N1', 'N2', 'N3'],
    
    # M分期
    'm_stages': ['M0', 'M1'],
    
    # 食管位置
    'locations': ['Upper', 'Middle', 'Lower'],
    
    # 生存状态
    'survival': ['alive', 'dead'],
} 