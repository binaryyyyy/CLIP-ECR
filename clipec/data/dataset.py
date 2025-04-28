import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import SimpleITK as sitk
from torchvision import transforms

class ESCCDataset(Dataset):
    """食管癌CT图像数据集"""
    
    def __init__(self, image_dir, label_file, transform=None, 
                 slice_selection='middle', text_labels=None, 
                 grid_size=16, grid_layout=(4, 4)):
        """
        初始化数据集
        
        Args:
            image_dir (str): NRRD文件所在目录
            label_file (str): 包含标签信息的Excel文件路径
            transform (callable, optional): 图像变换
            slice_selection (str): 选择哪些切片, 'middle'表示中间切片, 
                                  'specific'表示指定切片,
                                  'grid'表示均匀选择切片并拼接成网格
            text_labels (list): 文本标签列表，例如['T1', 'T2', 'T3', 'T4']
            grid_size (int): 'grid'模式下要选择的切片数量
            grid_layout (tuple): 'grid'模式下的网格布局 (行, 列)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.slice_selection = slice_selection
        self.grid_size = grid_size
        
        # 解析网格布局
        if isinstance(grid_layout, str):
            parts = grid_layout.split(',')
            self.grid_layout = (int(parts[0]), int(parts[1]))
        else:
            self.grid_layout = grid_layout
            
        # 检查网格布局是否与尺寸匹配
        rows, cols = self.grid_layout
        if rows * cols != self.grid_size:
            print(f"警告：网格布局 {rows}x{cols} 与网格大小 {self.grid_size} 不匹配")
            # 调整网格大小以匹配布局
            self.grid_size = rows * cols
        
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
    
    def create_grid_image(self, img_array, slice_indices):
        """
        从CT体积中提取指定的切片并创建网格布局的2D图像
        
        Args:
            img_array (numpy.ndarray): CT体积数据
            slice_indices (list): 要提取的切片索引列表
            
        Returns:
            torch.Tensor: 4D张量，形状为[channels, rows*slice_height, cols*slice_width]
        """
        # 确保切片索引列表的长度等于指定的网格大小
        assert len(slice_indices) == self.grid_size, "切片索引数量必须等于网格大小"
        
        # 提取并预处理每个切片
        processed_slices = []
        for idx in slice_indices:
            # 获取切片
            slice_img = img_array[idx, :, :]
            
            # 调整窗宽窗位并归一化
            slice_img = np.clip(slice_img, -100, 400)  # 食管癌常用窗宽窗位
            slice_img = (slice_img - (-100)) / 500  # 归一化到[0,1]
            
            processed_slices.append(slice_img)
        
        # 获取切片的高度和宽度
        slice_height, slice_width = processed_slices[0].shape
        
        # 创建网格布局
        rows, cols = self.grid_layout
        
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
            grid[start_row:start_row+slice_height, 
                 start_col:start_col+slice_width] = slice_img
        
        # 将网格转换为3通道图像
        grid_3ch = np.stack([grid] * 3, axis=0)
        
        # 转换为tensor
        return torch.FloatTensor(grid_3ch)
    
    def __getitem__(self, idx):
        # 获取文件路径和患者ID
        file_path = self.valid_files[idx]
        patient_id = self.patient_ids[idx]
        
        try:
            # 使用SimpleITK读取nrrd文件
            img = sitk.ReadImage(file_path)
            img_array = sitk.GetArrayFromImage(img)
            num_slices = img_array.shape[0]
            
            # 选择切片
            if self.slice_selection == 'grid':
                # 均匀选择切片
                if num_slices >= self.grid_size:
                    # 如果切片数量足够，均匀选择切片
                    step = max(1, num_slices // self.grid_size)
                    start = max(0, (num_slices - (self.grid_size-1) * step) // 2)
                    slice_indices = [start + i * step for i in range(self.grid_size)]
                    
                    # 确保不超出范围
                    slice_indices = [min(idx, num_slices-1) for idx in slice_indices]
                else:
                    # 如果切片数量不足，将现有切片重复使用
                    slice_indices = []
                    for i in range(self.grid_size):
                        slice_idx = (i * num_slices) // self.grid_size
                        slice_indices.append(slice_idx)
                
                # 创建网格图像
                image = self.create_grid_image(img_array, slice_indices)
                slice_idx = slice_indices  # 保存所有使用的切片索引
            elif self.slice_selection == 'middle':
                # 选择中间的切片
                slice_idx = img_array.shape[0] // 2
                slice_img = img_array[slice_idx, :, :]
                
                # 数据预处理
                slice_img = np.clip(slice_img, -100, 400)  # 食管癌常用窗宽窗位
                slice_img = (slice_img - (-100)) / 500  # 归一化到[0,1]
                
                # 转换为3通道图像
                slice_img = np.stack([slice_img] * 3, axis=0)
                
                # 转为Tensor
                image = torch.FloatTensor(slice_img)
            else:  # 默认或specific模式
                # 默认选择第一个切片
                slice_idx = 0
                slice_img = img_array[slice_idx, :, :]
                
                # 数据预处理
                slice_img = np.clip(slice_img, -100, 400)  # 食管癌常用窗宽窗位
                slice_img = (slice_img - (-100)) / 500  # 归一化到[0,1]
                
                # 转换为3通道图像
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
                'patient_id': patient_id,
                'slice_idx': slice_idx
            }
            
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
            # 返回一个空的样本
            if self.slice_selection == 'grid':
                rows, cols = self.grid_layout
                empty_image = torch.zeros((3, rows*224, cols*224))
                slice_idx = [0] * self.grid_size
            else:
                empty_image = torch.zeros((3, 512, 512))
                slice_idx = 0
                
            return {
                'image': empty_image,
                'labels': {label: "" for label in self.text_labels},
                'patient_id': patient_id,
                'slice_idx': slice_idx
            }

def get_data_loaders(
    image_dir, label_file, batch_size=8, train_ratio=0.75, 
    val_ratio=0.15, num_workers=4, slice_selection='middle',
    grid_size=16, grid_layout=(4, 4), resize_grid=224
):
    """创建训练、验证和测试数据加载器"""
    
    # 解析网格布局
    if isinstance(grid_layout, str):
        parts = grid_layout.split(',')
        rows, cols = int(parts[0]), int(parts[1])
        grid_layout_tuple = (rows, cols)
    else:
        grid_layout_tuple = grid_layout
        rows, cols = grid_layout_tuple
    
    # 定义数据变换
    if slice_selection == 'grid':
        # 对于网格模式，需要调整输入大小
        transform = transforms.Compose([
            transforms.Resize((rows*resize_grid, cols*resize_grid)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    # 创建数据集
    dataset = ESCCDataset(
        image_dir, 
        label_file, 
        transform=transform, 
        slice_selection=slice_selection,
        grid_size=grid_size,
        grid_layout=grid_layout_tuple
    )
    
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
    
    # 创建子集
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