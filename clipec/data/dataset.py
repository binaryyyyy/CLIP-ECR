import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nrrd
import SimpleITK as sitk
from torchvision import transforms

class ESCCDataset(Dataset):
    """食管癌CT图像数据集"""
    
    def __init__(self, image_dir, label_file, transform=None, slice_selection='middle', text_labels=None):
        """
        初始化数据集
        
        Args:
            image_dir (str): NRRD文件所在目录
            label_file (str): 包含标签信息的Excel文件路径
            transform (callable, optional): 图像变换
            slice_selection (str): 选择哪些切片, 'middle'表示中间切片
            text_labels (list): 文本标签列表，例如['T1', 'T2', 'T3', 'T4']
        """
        self.image_dir = image_dir
        self.transform = transform
        self.slice_selection = slice_selection
        
        # 读取标签文件
        self.labels_df = pd.read_excel(label_file)
        
        # 获取有效文件列表（匹配标签文件中的住院号）
        self.valid_files = []  # 有效文件列表(image/nrrd文件)
        self.patient_ids = []  # 住院号
        
        for _, row in self.labels_df.iterrows():
            patient_id = str(row['住院号'])
            file_path = os.path.join(image_dir, f"{patient_id}.nrrd")
            if os.path.exists(file_path):
                self.valid_files.append(file_path)
                self.patient_ids.append(patient_id)
        
        print(f"找到 {len(self.valid_files)} 个有效文件")
        
        # 提取需要的标签
        self.text_labels = text_labels or [
            'Primary Site - labeled', 'AJCC8th', 
            'Derived AJCC T, 7th ed (2010-2015)', 
            'Derived AJCC N, 7th ed (2010-2015)', 
            'Derived AJCC M, 7th ed (2010-2015)', 
            'SEER cause-specific death classification，alive=0，dead=1'
            # 可指定或按默认的
        ]
        
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        # 读取.nrrd文件
        file_path = self.valid_files[idx]
        patient_id = self.patient_ids[idx]
        
        try:
            # 使用SimpleITK读取nrrd文件
            img = sitk.ReadImage(file_path)
            img_array = sitk.GetArrayFromImage(img)
            
            # 选择切片
            if self.slice_selection == 'middle':
                # 选择中间的切片
                slice_idx = img_array.shape[0] // 2
                slice_img = img_array[slice_idx, :, :]
            else:
                # TODO 可以实现其他切片选择策略
                slice_img = img_array[0, :, :]
            
            # 数据预处理
            # 1. 调整窗宽窗位（Windowing）- 针对CT图像的常见预处理
            # 这里暂时使用一个简单的线性变换
            # TODO: 未来需要根据运行效果进行调整
            slice_img = np.clip(slice_img, -100, 400)  # 食管癌常用窗宽窗位
            slice_img = (slice_img - (-100)) / 500  # 归一化到[0,1]
            
            # 2. 转换为3通道图像（模拟RGB）以匹配预训练模型的输入
            slice_img = np.stack([slice_img] * 3, axis=0)
            
            # 转为Tensor
            image = torch.FloatTensor(slice_img)
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
                
            # 获取标签
            label_row = self.labels_df[
                self.labels_df['住院号'] == int(patient_id)
            ]

            
            # 提取所有需要的标签
            labels = {}
            for label_name in self.text_labels:
                if label_name in label_row.columns:
                    value = label_row[label_name].values[0]
                    # 将值转换为字符串，处理NaN情况
                    if pd.notna(value):
                        labels[label_name] = str(value)
                    else:
                        labels[label_name] = ""
                else:
                    labels[label_name] = ""
            
            return {
                'image': image, 
                'labels': labels,
                'patient_id': patient_id
            }
            
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
            # 返回一个空的样本
            return {
                'image': torch.zeros((3, 512, 512)),
                'labels': {label: "" for label in self.text_labels},
                'patient_id': patient_id
            }

def get_data_loaders(
    image_dir, label_file, batch_size=8, train_ratio=0.75, 
    val_ratio=0.15, num_workers=4
):
    """创建训练、验证和测试数据加载器"""
    
    # 定义数据变换
    transform = transforms.Compose([
        # 注意: 输入已经是Tensor，所以我们不需要ToTensor变换
        # TODO：调整后的大小需要根据选择的模型进行调整
        # TODO：标准化需要根据实际的图像进行调整！
        # TODO：如果可以的话，使用数据增强
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 创建数据集
    dataset = ESCCDataset(image_dir, label_file, transform=transform)
    
    # 数据集大小
    dataset_size = len(dataset)
    
    # 计算分割点
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    # 分割数据集
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    val_to_train = val_indices[:len(val_indices)//2]
    test_to_train = test_indices[:len(test_indices)//2]
    
    train_indices = indices[:train_size] + val_to_train + test_to_train
    # val_indices = val_indices[len(val_indices)//2:]
    # test_indices = test_indices[len(test_indices)//2:]
    
    # 创建子集
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, dataset.text_labels 