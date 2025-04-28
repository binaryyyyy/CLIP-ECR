import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clipec.models.clip_model import CLIPModel
from clipec.utils.helpers import set_seed, visualize_slice


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
                        choices=['middle', 'all', 'specific'],
                        help='切片选择方式: middle-中间切片, all-所有切片, specific-指定切片索引')
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='当slice_selection为specific时使用的切片索引')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='CLIP-ECR/demo_results',
                        help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', action='store_true',
                        help='强制使用GPU，如无可用GPU则报错')
    
    return parser.parse_args()


def preprocess_image(image_path, slice_selection='middle', specific_slice_idx=None):
    """预处理NRRD图像
    
    Args:
        image_path (str): NRRD文件路径
        slice_selection (str): 切片选择方式，'middle', 'all' 或 'specific'
        specific_slice_idx (int, optional): 当slice_selection为'specific'时的切片索引
        
    Returns:
        tuple: (图像张量, 原始切片数据, 切片索引列表)
    """
    # 读取NRRD文件
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    
    # 准备变换
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 根据切片选择方式处理
    if slice_selection == 'middle':
        # 选择中间切片
        slice_idx = img_array.shape[0] // 2
        return process_single_slice(img_array, slice_idx, transform)
    elif slice_selection == 'specific' and specific_slice_idx is not None:
        # 选择指定切片
        if specific_slice_idx >= 0 and specific_slice_idx < img_array.shape[0]:
            return process_single_slice(img_array, specific_slice_idx, transform)
        else:
            raise ValueError(f"指定的切片索引 {specific_slice_idx} 超出范围 [0, {img_array.shape[0] - 1}]")
    elif slice_selection == 'all':
        # 处理所有切片
        slices = []
        original_slices = []
        slice_indices = []
        
        for slice_idx in range(img_array.shape[0]):
            image_tensor, original_slice = process_single_slice(img_array, slice_idx, transform)
            slices.append(image_tensor)
            original_slices.append(original_slice)
            slice_indices.append(slice_idx)
        
        # 合并所有切片为一个批次
        batch_tensor = torch.cat(slices, dim=0)
        return batch_tensor, original_slices, slice_indices
    else:
        raise ValueError(f"不支持的slice_selection值: {slice_selection}")


def process_single_slice(img_array, slice_idx, transform):
    """处理单个切片
    
    Args:
        img_array (numpy.ndarray): 图像数组
        slice_idx (int): 切片索引
        transform (callable): 图像变换
        
    Returns:
        tuple: (图像张量, 原始切片数据)
    """
    # 获取指定切片
    slice_img = img_array[slice_idx, :, :]
    
    # 图像预处理
    # 窗宽窗位调整
    slice_img = np.clip(slice_img, -100, 400)
    slice_img = (slice_img - (-100)) / 500
    
    # 创建3通道图像
    slice_img_3ch = np.stack([slice_img] * 3, axis=0)
    
    # 转为Tensor
    image_tensor = torch.FloatTensor(slice_img_3ch)
    
    # 应用变换
    image_tensor = transform(image_tensor)
    
    # 添加批次维度
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, slice_img


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载和预处理图像
    print(f"加载图像: {args.image_path}")
    if args.slice_selection == 'all':
        image_tensors, original_slices, slice_indices = preprocess_image(
            args.image_path, 
            slice_selection=args.slice_selection
        )
        print(f"处理了 {len(slice_indices)} 个切片")
    else:
        image_tensor, original_slice = preprocess_image(
            args.image_path, 
            slice_selection=args.slice_selection,
            specific_slice_idx=args.slice_idx
        )
        image_tensors = image_tensor
        original_slices = [original_slice]
        slice_indices = [args.slice_idx if args.slice_selection == 'specific' 
                         else image_tensors.shape[0] // 2]
    
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
        image_tensors = image_tensors.to(device)
        
        # 获取图像特征
        image_features = model.encode_image(image_tensors)
        
        # 获取所有文本特征
        text_features = model.encode_text(text_labels)
        
        # 计算相似度
        similarity = torch.matmul(image_features, text_features.t())
        
        # 获取最大相似度的索引
        max_indices = torch.argmax(similarity, dim=1)
        
        # 获取相似度分数
        scores = torch.nn.functional.softmax(similarity, dim=1)
    
    # 显示预测结果
    if args.slice_selection == 'all':
        # 对所有切片的预测结果进行汇总
        predictions = []
        confidences = []
        
        for i, (max_idx, score) in enumerate(zip(max_indices, scores)):
            predicted_class = text_labels[max_idx]
            confidence = score[max_idx].item()
            predictions.append(predicted_class)
            confidences.append(confidence)
        
        # 创建汇总表格
        summary_path = os.path.join(args.output_dir, 'all_slices_predictions.csv')
        import pandas as pd
        df = pd.DataFrame({
            'slice_idx': slice_indices,
            'prediction': predictions,
            'confidence': confidences
        })
        df.to_csv(summary_path, index=False)
        print(f"所有切片的预测结果已保存到: {summary_path}")
        
        # 找出置信度最高的预测结果
        best_idx = np.argmax(confidences)
        best_prediction = predictions[best_idx]
        best_confidence = confidences[best_idx]
        best_slice_idx = slice_indices[best_idx]
        
        print(f"\n最佳预测结果 (切片 {best_slice_idx}):")
        print(f"预测: {best_prediction}")
        print(f"置信度: {best_confidence:.4f}")
        
        # 可视化最佳预测结果的切片
        plt.figure(figsize=(12, 6))
        
        # 左侧显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(original_slices[best_idx], cmap='gray')
        plt.title(f"CT切片 (索引: {best_slice_idx})")
        plt.axis('off')
        
        # 右侧显示置信度条形图
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(text_labels))
        plt.barh(y_pos, scores[best_idx].cpu().numpy())
        plt.yticks(y_pos, [label.split('分期')[-1] for label in text_labels])
        plt.xlabel('置信度')
        plt.title('分类置信度')
        
        # 保存和显示
        result_path = os.path.join(args.output_dir, 'best_prediction_result.png')
        plt.tight_layout()
        plt.savefig(result_path)
        print(f"最佳结果可视化已保存到: {result_path}")
        
        # 创建所有切片的预测结果可视化
        num_slices = min(len(slice_indices), 12)  # 限制为最多12个切片以避免图像过大
        rows = (num_slices + 2) // 3  # 每行最多3个切片
        
        plt.figure(figsize=(15, 5 * rows))
        for i in range(num_slices):
            plt.subplot(rows, 3, i + 1)
            plt.imshow(original_slices[i], cmap='gray')
            plt.title(f"切片 {slice_indices[i]}: {predictions[i].split('分期')[-1]} ({confidences[i]:.2f})")
            plt.axis('off')
        
        slices_result_path = os.path.join(args.output_dir, 'all_slices_preview.png')
        plt.tight_layout()
        plt.savefig(slices_result_path)
        print(f"所有切片预览已保存到: {slices_result_path}")
        
    else:
        # 单一切片的预测结果
        max_idx = max_indices.item()
        predicted_class = text_labels[max_idx]
        confidence = scores[0][max_idx].item()
        
        print(f"\n预测结果: {predicted_class}")
        print(f"置信度: {confidence:.4f}\n")
        
        # 显示所有分类的置信度
        print("各分类置信度:")
        for i, (label, score) in enumerate(zip(text_labels, scores[0])):
            print(f"{label}: {score.item():.4f}")
        
        # 可视化结果
        plt.figure(figsize=(12, 6))
        
        # 左侧显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(original_slices[0], cmap='gray')
        plt.title(f"CT切片 (索引: {slice_indices[0]})")
        plt.axis('off')
        
        # 右侧显示置信度条形图
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(text_labels))
        plt.barh(y_pos, scores[0].cpu().numpy())
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