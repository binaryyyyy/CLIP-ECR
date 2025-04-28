import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from collections import Counter
import math

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clipec.models.clip_model import CLIPModel
from clipec.utils.helpers import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-ECR模型演示')
    
    # 输入文件
    parser.add_argument('--image_path', type=str, required=True,
                        help='输入NRRD文件路径')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--image_encoder', type=str, default='resnet50',
                        help='图像编码器类型')
    parser.add_argument('--text_encoder', type=str, default='simple',
                        help='文本编码器类型')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='嵌入维度')
    
    # 切片选择参数
    parser.add_argument('--slice_selection', type=str, default='middle',
                        choices=['middle', 'specific', 'grid'],
                        help='切片选择方式: middle-中间切片, specific-指定切片索引, grid-网格拼接切片')
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='当slice_selection为specific时使用的切片索引')
    parser.add_argument('--grid_size', type=int, default=16,
                        help='grid模式下选择的切片数量')
    parser.add_argument('--grid_layout', type=str, default='4,4',
                        help='grid模式下的网格布局，格式为"行,列"，例如"4,4"')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='CLIP-ECR/demo_results',
                        help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', action='store_true',
                        help='强制使用GPU，如无可用GPU则报错')
    
    return parser.parse_args()


def create_grid_image(img_array, slice_indices, grid_layout=(4, 4)):
    """
    从CT体积中提取指定的切片并创建网格布局的2D图像
    
    Args:
        img_array (numpy.ndarray): CT体积数据
        slice_indices (list): 要提取的切片索引列表
        grid_layout (tuple): 网格布局 (行, 列)
        
    Returns:
        tuple: (处理后的网格图像张量, 原始网格图像数组, 切片索引列表)
    """
    # 确保切片索引列表的长度等于指定的网格大小
    assert len(slice_indices) == grid_layout[0] * grid_layout[1], "切片索引数量必须等于网格大小"
    
    # 提取并预处理每个切片
    processed_slices = []
    original_slices = []
    for idx in slice_indices:
        # 获取切片
        slice_img = img_array[idx, :, :]
        original_slices.append(slice_img.copy())
        
        # 调整窗宽窗位并归一化
        slice_img = np.clip(slice_img, -100, 400)  # 食管癌常用窗宽窗位
        slice_img = (slice_img - (-100)) / 500  # 归一化到[0,1]
        
        processed_slices.append(slice_img)
    
    # 获取切片的高度和宽度
    slice_height, slice_width = processed_slices[0].shape
    
    # 创建网格布局
    rows, cols = grid_layout
    
    # 创建空网格
    grid = np.zeros((rows * slice_height, cols * slice_width))
    
    # 填充网格
    for i, slice_img in enumerate(processed_slices):
        row_idx = i // cols
        col_idx = i % cols
        
        # 计算在网格中的位置
        start_row = row_idx * slice_height
        start_col = col_idx * slice_width
        
        # 放置切片
        grid[start_row:start_row+slice_height, start_col:start_col+slice_width] = slice_img
    
    # 将网格转换为3通道图像
    grid_3ch = np.stack([grid] * 3, axis=0)
    
    # 转换为tensor
    return torch.FloatTensor(grid_3ch), grid, slice_indices


def preprocess_image(image_path, slice_selection='middle', specific_slice_idx=None, grid_size=16, grid_layout=(4, 4)):
    """预处理NRRD图像
    
    Args:
        image_path (str): NRRD文件路径
        slice_selection (str): 切片选择方式，'middle', 'specific' 或 'grid'
        specific_slice_idx (int, optional): 当slice_selection为'specific'时的切片索引
        grid_size (int): 'grid'模式下要选择的切片数量
        grid_layout (tuple): 'grid'模式下的网格布局 (行, 列)
        
    Returns:
        tuple: (图像张量, 原始切片数据, 切片索引列表)
    """
    # 读取NRRD文件
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    
    # 准备变换
    from torchvision import transforms
    
    # 根据切片选择方式处理
    if slice_selection == 'middle':
        # 选择中间切片
        slice_idx = img_array.shape[0] // 2
        
        # 获取并预处理切片
        slice_img = img_array[slice_idx, :, :]
        original_slice = slice_img.copy()
        
        # 数据预处理
        slice_img = np.clip(slice_img, -100, 400)  # 食管癌常用窗宽窗位
        slice_img = (slice_img - (-100)) / 500  # 归一化到[0,1]
        
        # 创建3通道图像
        slice_img_3ch = np.stack([slice_img] * 3, axis=0)
        
        # 转为Tensor
        image_tensor = torch.FloatTensor(slice_img_3ch)
        
        # 调整大小
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image_tensor)
        
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_slice, slice_idx
    
    elif slice_selection == 'specific' and specific_slice_idx is not None:
        # 选择指定切片
        if specific_slice_idx >= 0 and specific_slice_idx < img_array.shape[0]:
            # 获取并预处理切片
            slice_img = img_array[specific_slice_idx, :, :]
            original_slice = slice_img.copy()
            
            # 数据预处理
            slice_img = np.clip(slice_img, -100, 400)  # 食管癌常用窗宽窗位
            slice_img = (slice_img - (-100)) / 500  # 归一化到[0,1]
            
            # 创建3通道图像
            slice_img_3ch = np.stack([slice_img] * 3, axis=0)
            
            # 转为Tensor
            image_tensor = torch.FloatTensor(slice_img_3ch)
            
            # 调整大小
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            image_tensor = transform(image_tensor)
            
            # 添加批次维度
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor, original_slice, specific_slice_idx
        else:
            raise ValueError(
                f"指定的切片索引 {specific_slice_idx} 超出范围 [0, {img_array.shape[0] - 1}]"
            )
    
    elif slice_selection == 'grid':
        # 均匀选择切片
        num_slices = img_array.shape[0]
        grid_size = grid_layout[0] * grid_layout[1]  # 计算网格大小
        
        if num_slices >= grid_size:
            # 如果切片数量足够，均匀选择切片
            step = max(1, num_slices // grid_size)
            start = max(0, (num_slices - (grid_size-1) * step) // 2)
            slice_indices = [start + i * step for i in range(grid_size)]
            
            # 确保不超出范围
            slice_indices = [min(idx, num_slices-1) for idx in slice_indices]
        else:
            # 如果切片数量不足，将现有切片重复使用
            slice_indices = []
            for i in range(grid_size):
                slice_idx = (i * num_slices) // grid_size
                slice_indices.append(slice_idx)
        
        # 创建网格图像
        grid_tensor, grid_img, _ = create_grid_image(img_array, slice_indices, grid_layout)
        
        # 调整大小
        rows, cols = grid_layout
        transform = transforms.Compose([
            transforms.Resize((rows*224, cols*224)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        grid_tensor = transform(grid_tensor)
        
        # 添加批次维度
        grid_tensor = grid_tensor.unsqueeze(0)
        
        return grid_tensor, grid_img, slice_indices
    
    else:
        raise ValueError(f"不支持的slice_selection值: {slice_selection}")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析网格布局
    if args.slice_selection == 'grid':
        grid_layout = tuple(map(int, args.grid_layout.split(',')))
        if len(grid_layout) != 2:
            raise ValueError("grid_layout必须是形如'行,列'的字符串，例如'4,4'")
    else:
        grid_layout = (4, 4)  # 默认值
    
    # 加载和预处理图像
    print(f"加载图像: {args.image_path}")
    if args.slice_selection == 'grid':
        print(f"使用网格模式, 网格大小: {args.grid_size}, 布局: {grid_layout[0]}x{grid_layout[1]}")
        image_tensor, original_image, slice_indices = preprocess_image(
            args.image_path, 
            slice_selection=args.slice_selection,
            grid_size=args.grid_size,
            grid_layout=grid_layout
        )
        print(f"选择了 {len(slice_indices)} 个切片进行网格拼接")
    else:
        image_tensor, original_image, slice_idx = preprocess_image(
            args.image_path, 
            slice_selection=args.slice_selection,
            specific_slice_idx=args.slice_idx
        )
        if args.slice_selection == 'specific':
            print(f"使用指定切片: {slice_idx}")
        else:
            print(f"使用中间切片: {slice_idx}")
    
    # 创建模型
    print("加载模型...")
    model = CLIPModel(
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embedding_dim=args.embedding_dim
    )
    
    # 确定设备
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    elif args.gpu and not torch.cuda.is_available():
        raise RuntimeError("已指定强制使用GPU，但未检测到可用的CUDA设备！")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("未检测到GPU，使用CPU进行计算")
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型已加载到设备: {device}")
    # 打印模型大小
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 准备文本标签
    text_labels = [f"食管癌AJCC分期{stage}" for stage in 
                  ['1', '2a', '2b', '3a', '3b', '4']]
    
    # 进行预测
    print("预测中...")
    with torch.no_grad():
        # 将图像移到相同设备
        image_tensor = image_tensor.to(device)
        
        # 获取图像特征
        image_features = model.encode_image(image_tensor)
        
        # 获取所有文本特征
        text_features = model.encode_text(text_labels)
        
        # 计算相似度
        similarity = torch.matmul(image_features, text_features.t())
        
        # 获取最大相似度的索引
        max_idx = torch.argmax(similarity, dim=1).item()
        
        # 获取相似度分数
        scores = torch.nn.functional.softmax(similarity[0], dim=0)
    
    # 显示预测结果
    predicted_class = text_labels[max_idx]
    confidence = scores[max_idx].item()
    
    print(f"\n预测结果: {predicted_class}")
    print(f"置信度: {confidence:.4f}\n")
    
    # 显示所有分类的置信度
    print("各分类置信度:")
    for i, (label, score) in enumerate(zip(text_labels, scores)):
        print(f"{label}: {score.item():.4f}")
    
    # 可视化结果
    if args.slice_selection == 'grid':
        # 创建一个大图，展示网格和预测结果
        plt.figure(figsize=(15, 10))
        
        # 绘制网格图像
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f"CT切片网格 ({grid_layout[0]}x{grid_layout[1]})")
        
        # 在网格上添加切片索引标注
        rows, cols = grid_layout
        for i in range(len(slice_indices)):
            row_idx = i // cols
            col_idx = i % cols
            plt.text(
                col_idx * original_image.shape[1] / cols + 10,
                row_idx * original_image.shape[0] / rows + 20,
                f"#{slice_indices[i]}",
                color='white', fontsize=8, weight='bold'
            )
        
        plt.axis('off')
        
        # 右侧显示置信度条形图
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(text_labels))
        plt.barh(y_pos, scores.cpu().numpy())
        plt.yticks(y_pos, [label.split('分期')[-1] for label in text_labels])
        plt.xlabel('置信度')
        plt.title(f"分类置信度 (预测: {predicted_class.split('分期')[-1]}, 置信度: {confidence:.4f})")
        
        # 保存和显示
        result_path = os.path.join(args.output_dir, 'grid_prediction_result.png')
        plt.tight_layout()
        plt.savefig(result_path)
        
        print(f"\n结果可视化已保存到: {result_path}")
        
    else:
        # 单一切片的可视化
        plt.figure(figsize=(12, 6))
        
        # 左侧显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        if args.slice_selection == 'specific':
            plt.title(f"CT切片 (索引: {args.slice_idx})")
        else:
            plt.title(f"CT切片 (中间)")
        plt.axis('off')
        
        # 右侧显示置信度条形图
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(text_labels))
        plt.barh(y_pos, scores.cpu().numpy())
        plt.yticks(y_pos, [label.split('分期')[-1] for label in text_labels])
        plt.xlabel('置信度')
        plt.title('分类置信度')
        
        # 保存和显示
        result_path = os.path.join(args.output_dir, 'prediction_result.png')
        plt.tight_layout()
        plt.savefig(result_path)
        
        print(f"\n结果可视化已保存到: {result_path}")


if __name__ == '__main__':
    main() 