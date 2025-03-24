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
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='CLIP-ECR/demo_results',
                        help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', action='store_true',
                        help='强制使用GPU，如无可用GPU则报错')
    
    return parser.parse_args()


def preprocess_image(image_path):
    """预处理NRRD图像"""
    # 读取NRRD文件
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    
    # 选择中间切片
    slice_idx = img_array.shape[0] // 2
    slice_img = img_array[slice_idx, :, :]
    
    # 图像预处理
    # 窗宽窗位调整
    slice_img = np.clip(slice_img, -100, 400)
    slice_img = (slice_img - (-100)) / 500
    
    # 创建3通道图像
    slice_img_3ch = np.stack([slice_img] * 3, axis=0)
    
    # 转为Tensor
    image_tensor = torch.FloatTensor(slice_img_3ch)
    
    # 调整大小以匹配模型输入
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
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
    image_tensor, original_slice = preprocess_image(args.image_path)
    
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
    plt.figure(figsize=(12, 6))
    
    # 左侧显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap='gray')
    plt.title("CT切片")
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