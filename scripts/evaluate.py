import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clipec.data.dataset import get_data_loaders
from clipec.models.clip_model import CLIPModel
from clipec.training.trainer import CLIPTrainer
from clipec.utils.helpers import set_seed, load_model


def parse_args():
    parser = argparse.ArgumentParser(description='评估CLIP-ECR模型')
    
    # 数据相关参数
    parser.add_argument('--image_dir', type=str, default='F:/1_ML/data/image',
                        help='NRRD图像文件目录')
    parser.add_argument('--label_file', type=str, default='F:/1_ML/data/table_info.xlsx',
                        help='标签Excel文件路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--slice_selection', type=str, default='middle',
                        choices=['middle', 'all'],
                        help='CT切片选择方式: middle-仅中间切片, all-所有切片')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--image_encoder', type=str, default='resnet50',
                        help='图像编码器类型')
    parser.add_argument('--text_encoder', type=str, default='simple',
                        help='文本编码器类型')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='嵌入维度')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='温度系数')
    
    # 评估相关参数
    parser.add_argument('--output_dir', type=str, default='CLIP-ECR/results',
                        help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', action='store_true',
                        help='强制使用GPU，如无可用GPU则报错')
    
    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """绘制混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 设置图像大小
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    # 保存图像
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def evaluate_model(model, data_loader, text_labels, output_dir, slice_selection='middle', force_gpu=False):
    """评估模型性能"""
    # 创建训练器（只用于预测，不进行训练）
    trainer = CLIPTrainer(model=model, save_dir=output_dir, force_gpu=force_gpu)
    
    # 进行预测
    predictions, true_labels, patient_ids = trainer.predict(data_loader, text_labels)
    
    # 获取AJCC8th标签用于评估
    y_true = []
    y_pred = []
    patient_list = []
    slice_indices = []
    
    for i, label_dict in enumerate(true_labels):
        # 如果有AJCC8th标签
        if 'AJCC8th' in label_dict and label_dict['AJCC8th'] is not None:
            # 真实标签
            true_label = str(label_dict['AJCC8th'])
            y_true.append(true_label)
            
            # 预测标签（从文本中提取）
            pred_text = predictions[i]
            pred_label = pred_text.split('分期')[-1] if '分期' in pred_text else pred_text
            y_pred.append(pred_label)
            
            # 保存患者ID
            patient_list.append(patient_ids[i])
            
            # 如果是all模式，记录切片索引
            if slice_selection == 'all' and hasattr(data_loader.dataset, 'dataset'):
                # 对于Subset类型的dataset，需要获取原始dataset
                orig_dataset = data_loader.dataset.dataset
                idx = data_loader.dataset.indices[i]
                if hasattr(orig_dataset, 'slices_info'):
                    _, _, slice_idx = orig_dataset.slices_info[idx]
                    slice_indices.append(slice_idx)
                else:
                    slice_indices.append(None)
            else:
                slice_indices.append(None)
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 生成分类报告
    unique_labels = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=unique_labels)
    print("分类报告:")
    print(report)
    
    # 保存分类报告到文件
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"准确率: {accuracy:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(report)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        unique_labels,
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'patient_id': patient_list,
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': [t == p for t, p in zip(y_true, y_pred)]
    })
    
    # 如果是all模式，添加切片索引
    if slice_selection == 'all':
        results_df['slice_idx'] = slice_indices
    
    # 保存结果到CSV
    results_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"评估结果已保存到 {results_path}")
    
    return accuracy, results_df


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取数据加载器
    print("加载数据...")
    print(f"切片选择模式: {args.slice_selection}")
    _, _, test_loader, text_labels = get_data_loaders(
        args.image_dir,
        args.label_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slice_selection=args.slice_selection
    )
    print(f"测试集: {len(test_loader.dataset)}个样本")
    
    # 创建模型
    print(f"创建模型: {args.image_encoder} + {args.text_encoder}")
    model = CLIPModel(
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature
    )
    
    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建用于测试的文本标签
    # 这里使用AJCC分期作为示例
    test_text_labels = [f"食管癌AJCC分期{stage}" for stage in 
                        ['1', '2a', '2b', '3a', '3b', '4', '未知分期']]
    
    # 评估模型
    print("开始评估...")
    accuracy, results_df = evaluate_model(
        model=model,
        data_loader=test_loader,
        text_labels=test_text_labels,
        output_dir=args.output_dir,
        slice_selection=args.slice_selection,
        force_gpu=args.gpu
    )
    
    print(f"评估完成！准确率: {accuracy:.4f}")


if __name__ == '__main__':
    main() 