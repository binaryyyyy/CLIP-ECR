import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clipec.data.dataset import get_data_loaders
from clipec.models.clip_model import CLIPModel
from clipec.training.trainer import CLIPTrainer
from clipec.utils.helpers import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='训练CLIP-ECR模型')
    
    # 数据相关参数
    parser.add_argument('--image_dir', type=str, default='F:/1_ML/data/image',
                        help='NRRD图像文件目录')
    parser.add_argument('--label_file', type=str, default='F:/1_ML/data/table_info.xlsx',
                        help='标签Excel文件路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    
    # 模型相关参数
    parser.add_argument('--image_encoder', type=str, default='resnet50',
                        help='图像编码器类型')
    parser.add_argument('--text_encoder', type=str, default='transformer',
                        help='文本编码器类型')
    parser.add_argument('--embedding_dim', type=int, default=1024,
                        help='嵌入维度')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='温度系数')
    
    # 训练相关参数
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='CLIP-ECR/checkpoints',
                        help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', action='store_true', 
                        help='强制使用GPU，如无可用GPU则报错')
    
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取数据加载器
    print("加载数据...")
    train_loader, val_loader, test_loader, text_labels = get_data_loaders(
        args.image_dir,
        args.label_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"数据加载完成。训练集: {len(train_loader.dataset)}个样本, "
          f"验证集: {len(val_loader.dataset)}个样本, "
          f"测试集: {len(test_loader.dataset)}个样本")
    
    # 创建模型
    print(f"创建模型：图像编码器={args.image_encoder}, 文本编码器={args.text_encoder}, "
          f"嵌入维度={args.embedding_dim}")
    model = CLIPModel(
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature
    )
    
    # 创建训练器，如果指定--gpu参数则强制使用GPU
    trainer = CLIPTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        force_gpu=args.gpu
    )
    
    # 开始训练
    print(f"开始训练，总共{args.epochs}个epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_best=True
    )
    
    print(f"训练完成！最佳验证损失: {history['best_val_loss']:.4f}")
    
    # 可以在这里添加测试代码
    print("训练完成.")


if __name__ == '__main__':
    main() 