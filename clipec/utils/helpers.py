import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(force_gpu=False):
    """获取可用的设备（CPU或CUDA）"""
    if force_gpu:
        # 强制使用GPU，如果没有可用的GPU则抛出错误
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            return device
        else:
            raise RuntimeError("已指定强制使用GPU，但未检测到可用的CUDA设备！")
    else:
        # 自动选择设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("未检测到GPU，使用CPU进行计算")
        return device


def visualize_slice(slice_img, title=None, save_path=None):
    """可视化CT图像切片"""
    plt.figure(figsize=(10, 8))
    plt.imshow(slice_img, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def create_timestamp():
    """创建时间戳字符串，用于标记保存的模型和日志"""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def save_model(model, optimizer, epoch, loss, accuracy, save_dir, filename=None):
    """保存模型检查点"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_{create_timestamp()}.pt"
    
    path = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, path)
    
    return path


def load_model(model, optimizer, checkpoint_path):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', None)
    accuracy = checkpoint.get('accuracy', None)
    
    return model, optimizer, epoch, loss, accuracy


def compute_similarity_matrix(image_features, text_features, temperature=0.07):
    """计算图像和文本特征之间的相似度矩阵"""
    # 归一化特征向量
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算内积相似度
    similarity = torch.matmul(image_features, text_features.t()) / temperature
    
    return similarity 